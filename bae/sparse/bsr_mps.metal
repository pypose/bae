// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Metal adaptation of NVIDIA Warp release-1.19 sparse matrix-vector kernels.
// Changes: Metal thread dispatch, SIMD reductions and float32 atomic accumulation.
// License text: third_party/warp/LICENSE.md.
//
// BAE tuples use compact BSR storage, so Warp's optional row_counts is absent:
// row_end = crow[row + 1], row_count = crow[row + 1] - crow[row]. Blocks are
// row-major. Python integer arguments use int64; alpha and beta use float32.
// All launches flatten Warp's LAST dimension fastest, including transpose.
#include <metal_stdlib>
using namespace metal;

// https://github.com/NVIDIA/warp/blob/release-1.19/warp/_src/sparse.py#L4884-L4923
template <typename Index, int BlockCols, int BlockRows = 0>
kernel void bsr_mv(
    const device Index* crow [[buffer(0)]],
    const device Index* col [[buffer(1)]],
    const device float* values [[buffer(2)]],
    const device float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    constant long& nrows [[buffer(5)]],
    constant long& br [[buffer(6)]],
    constant long& bc [[buffer(7)]],
    constant float& alpha [[buffer(8)]],
    constant float& beta [[buffer(9)]],
    uint tid [[thread_position_in_grid]]) {
    const long yi = tid;
    // Specializing common BA block dimensions avoids
    // runtime integer division when decoding Warp's flattened (row, subrow).
    const long height = BlockRows == 0 ? br : BlockRows;
    if (yi >= nrows * height) return;
    const long row = yi / height;
    const long subrow = yi % height;
    const long width = BlockCols == 0 ? bc : BlockCols;
    float v = 0.0f;

    // Preserve Warp's zero-scale semantics: unused operands may contain NaNs
    // or uninitialized memory, and therefore must not be read.
    if (alpha != 0.0f) {
        for (long block = crow[row]; block < crow[row + 1]; ++block) {
            const long xs = long(col[block]) * width;
            const long vs = (block * height + subrow) * width;
            if (BlockCols != 0) {
                #pragma unroll
                for (int c = 0; c < BlockCols; ++c) {
                    v += values[vs + c] * x[xs + c];
                }
            } else {
                for (long c = 0; c < width; ++c) {
                    v += values[vs + c] * x[xs + c];
                }
            }
        }
        v *= alpha;
    }
    if (beta != 0.0f) v += beta * y[yi];
    y[yi] = v;
}

// https://github.com/NVIDIA/warp/blob/release-1.19/warp/_src/sparse.py#L4925-L4974
template <typename Index, int TileSize, int BlockRows = 0, int BlockCols = 0>
kernel void bsr_mv_tiled(
    const device Index* crow [[buffer(0)]],
    const device Index* col [[buffer(1)]],
    const device float* values [[buffer(2)]],
    const device float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    constant long& nrows [[buffer(5)]],
    constant long& br [[buffer(6)]],
    constant long& bc [[buffer(7)]],
    constant float& alpha [[buffer(8)]],
    constant float& beta [[buffer(9)]],
    constant long& tile_size [[buffer(10)]],
    uint yi [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_width [[threads_per_simdgroup]]) {
    // Launch exactly nrows * br groups, with tile_size lanes in every group.
    // The generic variant supports any tile size up to 1024; fixed variants
    // make the lane stride and scratch size compile-time constants, as Warp
    // does when constructing make_bsr_mv_tiled_kernel(tile_size).
    const long lanes = TileSize == 0 ? tile_size : TileSize;
    const long height = BlockRows == 0 ? br : BlockRows;
    const long width = BlockCols == 0 ? bc : BlockCols;
    if (long(yi) >= nrows * height) return;
    const long row = long(yi) / height;
    const long subrow = long(yi) % height;
    threadgroup float partial[TileSize == 0 ? 1024 : TileSize];

    if (alpha != 0.0f) {
        const long block_beg = crow[row];
        const long col_count = (long(crow[row + 1]) - block_beg) * width;
        float lane_sum = 0.0f;
        for (long c = lane; c < col_count; c += lanes) {
            long block = c / width;
            const long block_col = c - block * width;
            block += block_beg;
            const float xi = x[long(col[block]) * width + block_col];
            lane_sum += values[(block * height + subrow) * width + block_col] * xi;
        }
        lane_sum *= alpha;

        // Warp's tile_sum reduces the lane sums once per (row, subrow).
        // Native SIMD reduction followed by one reduction of SIMD partials
        // implements that collective without a barrier at every tree level.
        const float simd_total = simd_sum(lane_sum);
        if (simd_lane == 0) partial[simd_group] = simd_total;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_group == 0) {
            const uint simd_count = (uint(lanes) + simd_width - 1) / simd_width;
            float total = simd_lane < simd_count ? partial[simd_lane] : 0.0f;
            total = simd_sum(total);
            if (simd_lane == 0) {
                const float initial = beta == 0.0f ? 0.0f : beta * y[yi];
                y[yi] = initial + total;
            }
        }
    } else if (lane == 0) {
        y[yi] = beta == 0.0f ? 0.0f : beta * y[yi];
    }
}

// https://github.com/NVIDIA/warp/blob/release-1.19/warp/_src/sparse.py#L4976-L5006
template <typename Index, int BlockRows>
kernel void bsr_mv_transpose(
    const device Index* crow [[buffer(0)]],
    const device Index* col [[buffer(1)]],
    const device float* values [[buffer(2)]],
    const device float* x [[buffer(3)]],
    device atomic_float* y [[buffer(4)]],
    constant long& nrows [[buffer(5)]],
    constant long& br [[buffer(6)]],
    constant long& bc [[buffer(7)]],
    constant float& alpha [[buffer(8)]],
    constant float& beta [[buffer(9)]],
    uint tid [[thread_position_in_grid]]) {
    // Warp's dim=(nnz, block_cols) has subcol as its fastest dimension.
    // A Metal grid with (nnz, block_cols) reverses that order; flatten it here.
    const long block = long(tid) / bc;
    const long subcol = long(tid) % bc;
    if (alpha == 0.0f || block >= long(crow[nrows])) return;

    // Exact compact-storage bsr_row_index: lower_bound(crow, block+1) - 1.
    // This includes leading and consecutive empty rows.
    // https://github.com/NVIDIA/warp/blob/release-1.19/warp/_src/sparse.py#L2090-L2130
    long lo = 0, hi = nrows + 1;
    while (lo < hi) {
        const long mid = lo + (hi - lo) / 2;
        if (long(crow[mid]) < block + 1) lo = mid + 1;
        else hi = mid;
    }
    const long row = lo - 1;
    if (row < 0) return;

    const long height = BlockRows == 0 ? br : BlockRows;
    float col_sum = 0.0f;
    if (BlockRows != 0) {
        #pragma unroll
        for (int r = 0; r < BlockRows; ++r) {
            col_sum += values[(block * height + r) * bc + subcol] * x[row * height + r];
        }
    } else {
        for (long r = 0; r < height; ++r) {
            col_sum += values[(block * height + r) * bc + subcol] * x[row * height + r];
        }
    }
    // As in Warp, the caller zeroes/scales y by beta before this scatter.
    atomic_fetch_add_explicit(
        y + long(col[block]) * bc + subcol, alpha * col_sum, memory_order_relaxed);
}

// https://github.com/NVIDIA/warp/blob/release-1.19/warp/_src/sparse.py#L3375-L3382
kernel void bsr_scale(
    constant float& beta [[buffer(0)]],
    device float* y [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    y[tid] = beta == 0.0f ? 0.0f : beta * y[tid];
}

#define SCALAR_ARGS(Index) \
    const device Index*, const device Index*, const device float*, \
    const device float*, device float*, constant long&, constant long&, constant long&, \
    constant float&, constant float&, uint

#define TRANSPOSE_ARGS(Index) \
    const device Index*, const device Index*, const device float*, \
    const device float*, device atomic_float*, constant long&, constant long&, constant long&, \
    constant float&, constant float&, uint

#define TILED_ARGS(Index) \
    const device Index*, const device Index*, const device float*, \
    const device float*, device float*, constant long&, constant long&, constant long&, \
    constant float&, constant float&, constant long&, uint, uint, uint, uint, uint

#define SPECIALIZE_BLOCK(Index, suffix, N) \
template [[host_name("bsr_mv_" #suffix "_c" #N)]] \
kernel void bsr_mv<Index, N>(SCALAR_ARGS(Index)); \
template [[host_name("bsr_mv_transpose_" #suffix "_r" #N)]] \
kernel void bsr_mv_transpose<Index, N>(TRANSPOSE_ARGS(Index));

#define SPECIALIZE_TILE(Index, suffix, N) \
template [[host_name("bsr_mv_tiled_" #suffix "_t" #N)]] \
kernel void bsr_mv_tiled<Index, N>(TILED_ARGS(Index));

#define SPECIALIZE_BA(Index, suffix, Rows, Columns) \
template [[host_name("bsr_mv_" #suffix "_r" #Rows "_c" #Columns)]] \
kernel void bsr_mv<Index, Columns, Rows>(SCALAR_ARGS(Index));

#define SPECIALIZE_BA_TILE(Index, suffix, Rows, Columns, N) \
template [[host_name("bsr_mv_tiled_" #suffix "_r" #Rows "_c" #Columns "_t" #N)]] \
kernel void bsr_mv_tiled<Index, N, Rows, Columns>(TILED_ARGS(Index));

#define INSTANTIATE(Index, suffix) \
template [[host_name("bsr_mv_" #suffix)]] \
kernel void bsr_mv<Index, 0>(SCALAR_ARGS(Index)); \
template [[host_name("bsr_mv_transpose_" #suffix)]] \
kernel void bsr_mv_transpose<Index, 0>(TRANSPOSE_ARGS(Index)); \
template [[host_name("bsr_mv_tiled_" #suffix)]] \
kernel void bsr_mv_tiled<Index, 0>(TILED_ARGS(Index)); \
SPECIALIZE_BLOCK(Index, suffix, 1) \
SPECIALIZE_BLOCK(Index, suffix, 2) \
SPECIALIZE_BLOCK(Index, suffix, 3) \
SPECIALIZE_BLOCK(Index, suffix, 4) \
SPECIALIZE_BLOCK(Index, suffix, 6) \
SPECIALIZE_BLOCK(Index, suffix, 7) \
SPECIALIZE_BLOCK(Index, suffix, 9) \
SPECIALIZE_BLOCK(Index, suffix, 10) \
SPECIALIZE_BLOCK(Index, suffix, 12) \
SPECIALIZE_BLOCK(Index, suffix, 16) \
SPECIALIZE_TILE(Index, suffix, 32) \
SPECIALIZE_TILE(Index, suffix, 64) \
SPECIALIZE_TILE(Index, suffix, 128) \
SPECIALIZE_TILE(Index, suffix, 256) \
SPECIALIZE_BA(Index, suffix, 2, 3) \
SPECIALIZE_BA(Index, suffix, 2, 9) \
SPECIALIZE_BA(Index, suffix, 3, 2) \
SPECIALIZE_BA_TILE(Index, suffix, 9, 2, 64)

INSTANTIATE(int, i32)
INSTANTIATE(long, i64)
