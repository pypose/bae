#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDADataType.h>
#include <ATen/cuda/detail/DeviceThreadHandles.h>
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpGEMM

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
                                                                               \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
                                                                               \
    }                                                                          \
}

class CuSparse
{
private:
    bool initialized_ = false;
    int64_t a_num_rows_ = 0;
    int64_t a_num_cols_ = 0;
    int64_t a_nnz_ = 0;
    int64_t b_num_rows_ = 0;
    int64_t b_num_cols_ = 0;
    int64_t b_nnz_ = 0;
    c10::ScalarType value_dtype_ = c10::ScalarType::Undefined;
    cusparseSpMatDescr_t matA = NULL;
    cusparseSpMatDescr_t matB = NULL;
    cusparseSpMatDescr_t matC = NULL;
    cusparseSpGEMMDescr_t spgemmDesc = NULL;
    torch::Tensor c_crow;
    torch::Tensor c_col;
    torch::Tensor c_val;
    cudaStream_t current_stream = nullptr;
    void *dBuffer4 = NULL;
    void *dBuffer5 = NULL;

    void reset_cached_state()
    {
        initialized_ = false;
        a_num_rows_ = 0;
        a_num_cols_ = 0;
        a_nnz_ = 0;
        b_num_rows_ = 0;
        b_num_cols_ = 0;
        b_nnz_ = 0;
        value_dtype_ = c10::ScalarType::Undefined;

        if (matA != NULL)
        {
            CHECK_CUSPARSE(cusparseDestroySpMat(matA))
            matA = NULL;
        }
        if (matB != NULL)
        {
            CHECK_CUSPARSE(cusparseDestroySpMat(matB))
            matB = NULL;
        }
        if (matC != NULL)
        {
            CHECK_CUSPARSE(cusparseDestroySpMat(matC))
            matC = NULL;
        }
        if (spgemmDesc != NULL)
        {
            CHECK_CUSPARSE(cusparseSpGEMM_destroyDescr(spgemmDesc))
            spgemmDesc = NULL;
        }
        if (dBuffer4 != NULL)
        {
            CHECK_CUDA(cudaFree(dBuffer4))
            dBuffer4 = NULL;
        }
        if (dBuffer5 != NULL)
        {
            CHECK_CUDA(cudaFree(dBuffer5))
            dBuffer5 = NULL;
        }

        c_crow = torch::Tensor();
        c_col = torch::Tensor();
        c_val = torch::Tensor();
    }

    static cudaDataType get_supported_cuda_dtype(const torch::Tensor &tensor)
    {
        const auto dtype = tensor.scalar_type();
        TORCH_CHECK(
            tensor.is_floating_point(),
            "CuSparse only supports floating-point CSR tensors, got ",
            tensor.scalar_type());
        TORCH_CHECK(
            dtype == torch::kFloat16 || dtype == torch::kBFloat16 ||
                dtype == torch::kFloat32 || dtype == torch::kFloat64,
            "CuSparse only supports float16, bfloat16, float32, and float64 tensors, got ",
            tensor.scalar_type());
        return at::cuda::ScalarTypeToCudaDataType(dtype);
    }

    bool needs_rebuild(
        const torch::Tensor &a,
        const torch::Tensor &b) const
    {
        return !initialized_ || value_dtype_ != a.scalar_type() ||
               a_num_rows_ != a.size(-2) || a_num_cols_ != a.size(-1) ||
               a_nnz_ != a._nnz() || b_num_rows_ != b.size(-2) ||
               b_num_cols_ != b.size(-1) || b_nnz_ != b._nnz();
    }

public:
    CuSparse()
    {
        CHECK_CUSPARSE(cusparseSpGEMM_createDescr(&spgemmDesc))
    }

    ~CuSparse()
    {
        reset_cached_state();
    }
    torch::Tensor
    cusparse_spgemm(const torch::Tensor a, const torch::Tensor b)
    {
        TORCH_CHECK(a.is_cuda(), "CuSparse expects `a` to be a CUDA tensor.");
        TORCH_CHECK(b.is_cuda(), "CuSparse expects `b` to be a CUDA tensor.");
        TORCH_CHECK(a.layout() == torch::kSparseCsr, "CuSparse expects `a` to be sparse CSR.");
        TORCH_CHECK(b.layout() == torch::kSparseCsr, "CuSparse expects `b` to be sparse CSR.");
        TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "CuSparse expects 2D sparse CSR tensors.");
        TORCH_CHECK(a.device() == b.device(), "CuSparse expects `a` and `b` on the same CUDA device.");
        TORCH_CHECK(a.scalar_type() == b.scalar_type(), "CuSparse expects `a` and `b` to share a dtype.");
        TORCH_CHECK(a.size(-1) == b.size(-2), "CuSparse expected matrix shapes to align for multiplication, got ",
                    a.sizes(), " and ", b.sizes(), ".");

// Host problem definition
#define A_NUM_ROWS 4 // C compatibility
        const int A_num_rows = a.size(-2);
        const int A_num_cols = a.size(-1);
        const int A_nnz = a._nnz();
        const int B_num_rows = b.size(-2);
        const int B_num_cols = b.size(-1);
        const int B_nnz = b._nnz();

        const cudaDataType valueType = get_supported_cuda_dtype(a);
        cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
        cusparseOperation_t opB = CUSPARSE_OPERATION_NON_TRANSPOSE;
        const cudaDataType computeType =
            (valueType == CUDA_R_16F || valueType == CUDA_R_16BF) ? CUDA_R_32F : valueType;

        float alpha_f = 1.0f;
        float beta_f = 0.0f;
        double alpha_d = 1.0;
        double beta_d = 0.0;
        const void *alpha = nullptr;
        const void *beta = nullptr;
        switch (computeType)
        {
        case CUDA_R_32F:
            alpha = &alpha_f;
            beta = &beta_f;
            break;
        case CUDA_R_64F:
            alpha = &alpha_d;
            beta = &beta_d;
            break;
        default:
            TORCH_CHECK(false, "CuSparse does not support computeType ", computeType, ".");
        }
        //--------------------------------------------------------------------------
        // Device memory management: Allocate and copy A, B
        int *dA_csrOffsets, *dA_columns, *dB_csrOffsets, *dB_columns,
            *dC_csrOffsets, *dC_columns;
        void *dA_values, *dB_values, *dC_values;
        // allocate A
        torch::Tensor a_crow = a.crow_indices();
        torch::Tensor b_crow = b.crow_indices();
        torch::Tensor a_col = a.col_indices();
        torch::Tensor b_col = b.col_indices();
        if (a_crow.dtype() != torch::kInt32)
        {
            a_crow = a_crow.to(torch::kInt32);
            a_col = a_col.to(torch::kInt32);
        }
        if (b_crow.dtype() != torch::kInt32)
        {
            b_crow = b_crow.to(torch::kInt32);
            b_col = b_col.to(torch::kInt32);
        }
        dA_csrOffsets = a_crow.data_ptr<int>();
        dA_columns = a_col.data_ptr<int>();
        dA_values = a.values().data_ptr();
        // allocate B
        dB_csrOffsets = b_crow.data_ptr<int>();
        dB_columns = b_col.data_ptr<int>();
        dB_values = b.values().data_ptr();

        //--------------------------------------------------------------------------
        // CUSPARSE APIs
        cusparseHandle_t handle = at::cuda::getCurrentCUDASparseHandle();
        current_stream = c10::cuda::getCurrentCUDAStream();
        if (needs_rebuild(a, b))
        {
            reset_cached_state();
            CHECK_CUSPARSE(cusparseSpGEMM_createDescr(&spgemmDesc))
            value_dtype_ = a.scalar_type();
            a_num_rows_ = A_num_rows;
            a_num_cols_ = A_num_cols;
            a_nnz_ = A_nnz;
            b_num_rows_ = B_num_rows;
            b_num_cols_ = B_num_cols;
            b_nnz_ = B_nnz;

            // allocate C offsets
            c_crow = torch::empty({A_num_rows + 1}, torch::dtype(torch::kInt32).device(torch::kCUDA));
            dC_csrOffsets = c_crow.data_ptr<int>();
            size_t bufferSize1 = 0;
            size_t bufferSize2 = 0;
            size_t bufferSize3 = 0;
            size_t bufferSize4 = 0;
            size_t bufferSize5 = 0;
            // CHECK_CUSPARSE(cusparseCreate(&handle))
            // CHECK_CUSPARSE(cusparseSetStream(handle, c10::cuda::getCurrentCUDAStream()))
            // Create sparse matrix A in CSR format
            CHECK_CUSPARSE(cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,
                                             dA_csrOffsets, dA_columns, dA_values,
                                             CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                             CUSPARSE_INDEX_BASE_ZERO, valueType))
            CHECK_CUSPARSE(cusparseCreateCsr(&matB, B_num_rows, B_num_cols, B_nnz,
                                             dB_csrOffsets, dB_columns, dB_values,
                                             CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                             CUSPARSE_INDEX_BASE_ZERO, valueType))
            CHECK_CUSPARSE(cusparseCreateCsr(&matC, A_num_rows, B_num_cols, 0,
                                             NULL, NULL, NULL,
                                             CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                             CUSPARSE_INDEX_BASE_ZERO, valueType))
            //==========================================================================
            // SpGEMM Computation
            //==========================================================================

            // ask bufferSize1 bytes for external memory
            CHECK_CUSPARSE(
                cusparseSpGEMMreuse_workEstimation(handle, opA, opB, matA, matB, matC,
                                                   CUSPARSE_SPGEMM_DEFAULT,
                                                   spgemmDesc, &bufferSize1, NULL))
            auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
            at::DataPtr dataPtr1 = allocator.allocate(bufferSize1);
            void *dBuffer1 = dataPtr1.get();
            // inspect the matrices A and B to understand the memory requirement for
            // the next step
            CHECK_CUSPARSE(
                cusparseSpGEMMreuse_workEstimation(handle, opA, opB, matA, matB, matC,
                                                   CUSPARSE_SPGEMM_DEFAULT,
                                                   spgemmDesc, &bufferSize1, dBuffer1))
            //--------------------------------------------------------------------------

            CHECK_CUSPARSE(
                cusparseSpGEMMreuse_nnz(handle, opA, opB, matA, matB,
                                        matC, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc,
                                        &bufferSize2, NULL, &bufferSize3, NULL,
                                        &bufferSize4, NULL))
            at::DataPtr dataPtr2 = allocator.allocate(bufferSize2);
            void *dBuffer2 = dataPtr2.get();
            at::DataPtr dataPtr3 = allocator.allocate(bufferSize3);
            void *dBuffer3 = dataPtr3.get();
            CHECK_CUDA(cudaMallocAsync((void **)&dBuffer4, bufferSize4, current_stream))

            CHECK_CUSPARSE(
                cusparseSpGEMMreuse_nnz(handle, opA, opB, matA, matB,
                                        matC, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc,
                                        &bufferSize2, dBuffer2, &bufferSize3, dBuffer3,
                                        &bufferSize4, dBuffer4))
            //--------------------------------------------------------------------------

            // get matrix C non-zero entries C_nnz1
            int64_t C_num_rows1, C_num_cols1, C_nnz1;
            CHECK_CUSPARSE(cusparseSpMatGetSize(matC, &C_num_rows1, &C_num_cols1,
                                                &C_nnz1))
            // allocate matrix C
            c_col = torch::empty({C_nnz1}, torch::dtype(torch::kInt32).device(torch::kCUDA));
            c_val = torch::zeros({C_nnz1}, a.options().layout(torch::kStrided));
            dC_columns = c_col.data_ptr<int>();
            dC_values = c_val.data_ptr();

            // fill dC_values if needed
            // update matC with the new pointers
            CHECK_CUSPARSE(
                cusparseCsrSetPointers(matC, dC_csrOffsets, dC_columns, dC_values))
            //--------------------------------------------------------------------------

            CHECK_CUSPARSE(
                cusparseSpGEMMreuse_copy(handle, opA, opB, matA, matB, matC,
                                         CUSPARSE_SPGEMM_DEFAULT, spgemmDesc,
                                         &bufferSize5, NULL))
            CHECK_CUDA(cudaMallocAsync((void **)&dBuffer5, bufferSize5, current_stream))
            CHECK_CUSPARSE(
                cusparseSpGEMMreuse_copy(handle, opA, opB, matA, matB, matC,
                                         CUSPARSE_SPGEMM_DEFAULT, spgemmDesc,
                                         &bufferSize5, dBuffer5))
            initialized_ = true;
        }
        else
        {
            // update matC with the new pointers
            CHECK_CUSPARSE(
                cusparseCsrSetPointers(matA, dA_csrOffsets, dA_columns, dA_values))
            CHECK_CUSPARSE(
                cusparseCsrSetPointers(matB, dB_csrOffsets, dB_columns, dB_values))
        }
        //--------------------------------------------------------------------------
        // first run
        CHECK_CUSPARSE(
            cusparseSpGEMMreuse_compute(handle, opA, opB, alpha, matA, matB, beta,
                                        matC, computeType, CUSPARSE_SPGEMM_DEFAULT,
                                        spgemmDesc))
        return at::_sparse_compressed_tensor_unsafe(c_crow, c_col, c_val, {A_num_rows, B_num_cols}, at::TensorOptions().dtype(c_val.dtype()).device(c_val.device()).layout(at::kSparseCsr));
    }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<CuSparse>(m, "CuSparse")
        .def(py::init<>())
        .def("__call__", &CuSparse::cusparse_spgemm, "");
}
