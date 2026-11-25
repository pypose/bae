import os
import platform
import site
import subprocess
import sys
import sysconfig
from importlib.metadata import PackageNotFoundError, version as get_installed_version
from pathlib import Path
from packaging.specifiers import SpecifierSet
from packaging.version import Version
from setuptools import Command, setup, find_packages
from setuptools.command.bdist_wheel import bdist_wheel
from setuptools.command.build import build
from setuptools.errors import CompileError
import torch
from torch.utils.cpp_extension import CUDA_HOME, CUDAExtension, BuildExtension

VERSION = "0.2.6"
SUPPORTED_CUDSS_SPECIFIER = SpecifierSet("<0.9")
METAL_SOURCE = Path("bae/sparse/bsr_mps.metal")
METAL_LIBRARY = METAL_SOURCE.with_suffix(".metallib")


def build_metal_enabled():
    return sys.platform == "darwin" and platform.machine() == "arm64"


def build_cuda_enabled():
    # A GPU need not be visible to build a CUDA wheel (for example, in CI).
    return not build_metal_enabled() and CUDA_HOME is not None and torch.version.cuda is not None


def metal_deployment_target():
    target = Version(os.environ.get("MACOSX_DEPLOYMENT_TARGET", "14.0"))
    target = max(target, Version("14.0"))
    return f"{target.major}.{target.minor}"


class BuildMetal(Command):
    """Build the Metal library for both regular and PEP 660 editable installs."""

    user_options = []
    editable_mode = False

    def initialize_options(self):
        self.build_lib = None
        self.build_temp = None

    def finalize_options(self):
        self.set_undefined_options(
            "build", ("build_lib", "build_lib"), ("build_temp", "build_temp")
        )

    def get_source_files(self):
        return [str(METAL_SOURCE)]

    def get_outputs(self):
        return [str(Path(self.build_lib) / METAL_LIBRARY)] if build_metal_enabled() else []

    def get_output_mapping(self):
        if self.editable_mode and build_metal_enabled():
            return {self.get_outputs()[0]: str(METAL_LIBRARY)}
        return {}

    def run(self):
        if not build_metal_enabled():
            return
        if Version(torch.__version__.split("+")[0]) < Version("2.13") or not hasattr(torch.mps, "load_metallib"):
            raise CompileError("Building BAE for MPS requires PyTorch >= 2.13 with torch.mps.load_metallib.")

        output = METAL_LIBRARY if self.editable_mode else Path(self.get_outputs()[0])
        intermediate = Path(self.build_temp) / "bsr_mps.air"
        if self.dry_run:
            return
        output.parent.mkdir(parents=True, exist_ok=True)
        intermediate.parent.mkdir(parents=True, exist_ok=True)
        commands = [
            [
                "xcrun", "-sdk", "macosx", "metal", "-std=metal3.0",
                f"-mmacosx-version-min={metal_deployment_target()}",
                "-c", str(METAL_SOURCE), "-o", str(intermediate),
            ],
            ["xcrun", "-sdk", "macosx", "metallib", str(intermediate), "-o", str(output)],
        ]
        try:
            for command in commands:
                subprocess.run(command, check=True)
        except (OSError, subprocess.CalledProcessError) as error:
            raise CompileError(
                "Could not compile BAE's Metal kernels. Install Xcode and its Metal Toolchain "
                "(`xcodebuild -downloadComponent MetalToolchain`), then reinstall BAE."
            ) from error


class BuildWithMetal(build):
    sub_commands = build.sub_commands + [("build_metal", None)]


class PlatformWheel(bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        if build_metal_enabled():
            self.root_is_pure = False

    def get_tag(self):
        if build_metal_enabled():
            target = metal_deployment_target().replace(".", "_")
            return "py3", "none", f"macosx_{target}_arm64"
        return super().get_tag()

def readme():
    """Read the README.md file for long description"""
    try:
        with open("README.md", "r", encoding="utf-8") as f:
            return f.read()
    except:
        return "PyTorch implementation of BA"

# Check if CUDSS should be enabled
USE_CUDSS = os.environ.get("USE_CUDSS", "1").lower() in ("1", "true", "yes", "y")
# Optionally get CUDSS directory from environment variable.
# Prefer an explicit CuDSS root when the environment already exposes one,
CUDSS_DIR = (
    os.environ.get("CUDSS_DIR")
    or os.environ.get("CUDSS_ROOT")
    or ""
)
def validate_cudss_packages(specifier):
    for package_name in ("nvidia-cudss-cu13", "nvidia-cudss-cu12"):
        try:
            installed_version = get_installed_version(package_name)
        except PackageNotFoundError:
            continue

        if Version(installed_version) not in specifier:
            raise RuntimeError(
                f"{package_name}=={installed_version} is not supported. "
                f"Install {package_name}{specifier} or disable CuDSS with USE_CUDSS=0."
            )


def find_cudss_root():
    if CUDSS_DIR:
        return CUDSS_DIR

    prefixes = []
    try:
        prefixes.extend(site.getsitepackages())
    except Exception:
        pass

    for key in ("purelib", "platlib"):
        path = sysconfig.get_paths().get(key)
        if path:
            prefixes.append(path)

    candidates = []
    for prefix in dict.fromkeys(prefixes):
        candidates.extend([
            os.path.join(prefix, "nvidia", "cu13"),
            os.path.join(prefix, "nvidia", "cu12"),
        ])

    for candidate in candidates:
        include_dir = os.path.join(candidate, "include")
        library_dir = os.path.join(candidate, "lib")
        header = os.path.join(include_dir, "cudss.h")
        if os.path.exists(header) and os.path.isdir(library_dir):
            return candidate

    return ""


def resolve_cudss_link_inputs(cudss_root):
    if not cudss_root:
        return ['cudss'], [], []

    lib_dir = os.path.join(cudss_root, "lib")
    shared_lib = os.path.join(lib_dir, "libcudss.so")
    versioned_shared_lib = os.path.join(lib_dir, "libcudss.so.0")
    static_lib = os.path.join(lib_dir, "libcudss_static.a")

    libraries = []
    extra_link_args = [f'-Wl,-rpath,{lib_dir}']
    nvcc_args = []

    if os.path.exists(shared_lib):
        libraries.append('cudss')
        nvcc_args.append('-lcudss')
    elif os.path.exists(versioned_shared_lib):
        extra_link_args.append(versioned_shared_lib)
    elif os.path.exists(static_lib):
        nvcc_args.append(f'-Xlinker={static_lib}')
    else:
        libraries.append('cudss')
        nvcc_args.append('-lcudss')

    return libraries, extra_link_args, nvcc_args

def cuda_extensions():
    if not build_cuda_enabled():
        return []

    if USE_CUDSS:
        validate_cudss_packages(SUPPORTED_CUDSS_SPECIFIER)

    cudss_root = find_cudss_root()

    # Common extensions
    ext_modules = [
        CUDAExtension(
            'bae.sparse.spgemm', 
            [os.path.join('bae', 'sparse', 'cusparse_wrapper.cpp')]
        ),
        CUDAExtension(
            'bae.sparse.conversion', 
            [os.path.join('bae', 'sparse', 'sparse_conversion.cu')]
        ),
    ]
    
    # Add CUDSS-dependent extension conditionally
    if USE_CUDSS:
        cudss_libraries, cudss_extra_link_args, cudss_nvcc_link_args = resolve_cudss_link_inputs(cudss_root)
        libraries = ['cusolver', 'cusparse', *cudss_libraries]
        extra_compile_args = {
            'nvcc': [
                '-lcusolver',
                '-lcusparse',
                *cudss_nvcc_link_args,
            ]
        }
        extra_link_args = cudss_extra_link_args
        
        include_dirs = []
        library_dirs = []

        # Add CUDSS directory paths if specified
        if cudss_root:
            include_dirs.append(os.path.join(cudss_root, 'include'))
            library_dirs.append(os.path.join(cudss_root, 'lib'))
        else:
            # Fall back to distro installs when no explicit CuDSS root is configured.
            # Prefer the CUDA-13 package on systems where both CUDA-12 and CUDA-13 CuDSS are present.
            include_dirs.extend([
                '/usr/include/libcudss/13/',
                '/usr/include/libcudss/12/',
            ])
            library_dirs.extend([
                '/usr/lib/x86_64-linux-gnu/libcudss/13/',
                '/usr/lib/x86_64-linux-gnu/libcudss/12/',
            ])
            
        ext_modules.append(
            CUDAExtension(
                'bae.sparse.solve',
                [os.path.join('bae', 'sparse', 'sparse_cusolve.cu')],
                libraries=libraries,
                extra_compile_args=extra_compile_args,
                extra_link_args=extra_link_args,
                include_dirs=include_dirs,
                library_dirs=library_dirs,
            )
        )
    return ext_modules


if __name__ == '__main__':
    setup(
        name = 'bae',
        version = VERSION,
        description = 'PyTorch implementation of BA',
        long_description = readme(),
        long_description_content_type = "text/markdown",
        license_files=["LICENSE", "third_party/warp/LICENSE.md"],
        python_requires = ">=3.8",
        install_requires=[
            'torch>=2.13; sys_platform == "darwin" and platform_machine == "arm64"',
            'torch; sys_platform != "darwin" or platform_machine != "arm64"',
            'numpy',
            'scipy',
            'matplotlib',
            'pillow',
            'warp-lang>=1.13.0; sys_platform != "darwin"',
        ],
        packages=find_packages(exclude=['./ba_example.py', 
                                        './setup.py', 
                                        './README.md',
                                        './data',
                                        './1dsfm_bal',
                                        './bal_data',
                                        './colmap_helpers',
                                        './datapipes',
                                        './examples',
                                        './tests',]),
        ext_modules=cuda_extensions(),
        package_data={'bae.sparse': ['*.metal']},
        cmdclass={
            'build': BuildWithMetal,
            'build_metal': BuildMetal,
            'build_ext': BuildExtension,
            'bdist_wheel': PlatformWheel,
        },
    )
