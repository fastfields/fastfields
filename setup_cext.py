import collections
from setuptools import Extension
from buildtools import *
import torch
import os
from glob import glob as glob_


# ~~~ library files
# Note that sources are identical between libfastfields_cpu and libfastfields_cuda.
# The same code is compiled in two different ways to generate native and cuda
# code. This trick allows to minimize code duplication.
# Finally, libfastfields links to both these sub-libraries and dispatches
# according to the Tensor's device type.

libcpu_sources = ['src/impl/*.cpp']
libcuda_sources = ['src/impl/*.cpp']
lib_sources = ['src/*.cpp']
ext_sources = ['bind.cpp']

libcpu_headers = ['**.h']
libcuda_headers = ['**.h']
lib_headers = ['**.h']
ext_headers = ['**.h']

# TODO
# . There is still quite a lot to do in setup and buildtools in order to make
#   things clean and work on multiple platforms.
# . I have to add abi checks and other smart tricks as in
#   torch.utils.cpp_extension


def prepare_extensions():
    # ~~~ checks
    cuda_home = cuda_check()
    cudnn_home = None

    build_extensions = []
    lib = []
    libext = []

    # ~~~ setup libraries
    libfastfields_cpu = SharedLibrary(
        name='lib.fastfields_cpu',
        sources=abspathC(libcpu_sources, glob=True),
        depends=abspathC(libcpu_headers, glob=True),
        libraries=torch_libraries(),
        library_dirs=torch_library_dirs(),
        include_dirs=torch_include_dirs(),
        extra_compile_args=common_flags() + torch_flags(),
        extra_link_args=common_links_flags() + torch_link_flags(),
        language='c++',
    )
    build_extensions += [libfastfields_cpu]
    libext += [libfastfields_cpu]
    lib += ['fastfields_cpu']
    if cuda_home:
        libfastfields_cuda = SharedLibrary(
            name='lib.fastfields_cuda',
            sources=abspathC(libcuda_sources, glob=True),
            depends=abspathC(libcuda_headers, glob=True),
            libraries=torch_libraries(cuda_home),
            library_dirs=torch_library_dirs(cuda_home, cudnn_home),
            include_dirs=torch_include_dirs(cuda_home, cudnn_home),
            extra_compile_args=cuda_flags() + torch_flags(cuda=True),
            extra_link_args=torch_link_flags(cuda=True),
            language='cuda',
        )
        build_extensions += [libfastfields_cuda]
        libext += [libfastfields_cuda]
        lib += ['fastfields_cuda']
    libfastfields = SharedLibrary(
        name='lib.fastfields',
        sources=abspathC(lib_sources, glob=True),
        depends=libext + abspathC(lib_headers, glob=True),
        libraries=torch_libraries() + lib,
        library_dirs=torch_library_dirs(),
        include_dirs=torch_include_dirs(),
        extra_compile_args=common_flags() + torch_flags() + (['-DNI_WITH_CUDA'] if cuda_home else []),
        extra_link_args=common_links_flags(),
        runtime_library_dirs=[link_relative('.')],
        language='c++',
    )
    build_extensions += [libfastfields]
    libext = [libfastfields]
    lib = ['fastfields']
    # ~~~ setup extensions
    python_library_dirs = [os.path.join(sys.exec_prefix, 'lib')]
    extfastfields = Extension(
        name='ext.bind',
        sources=abspathC(ext_sources, glob=True),
        depends=libext + abspathC(ext_headers, glob=True),
        libraries=torch_libraries(cuda_home) + lib,
        library_dirs=torch_library_dirs(cuda_home, cudnn_home) + python_library_dirs,
        include_dirs=torch_include_dirs(cuda_home, cudnn_home),
        extra_compile_args=common_flags() + torch_flags() + torch_extension_flags('bind'),
        extra_link_args=common_links_flags(),
        runtime_library_dirs=[link_relative(os.path.join('..', 'lib'))]
    )
    build_extensions += [extfastfields]
    return build_extensions


# ~~~ helpers
# Most of the helpers are in build tools. The remaining helpers defined
# here are specific to the version of pytorch that we compile against.


def torch_version(astuple=True):
    version = list(torch.__version__.split('+')[0].split('.'))
    # strip alpha tags
    for n, v in enumerate(version):
        for x in 'abcdefghijklmnopqrstuvwxy':
            if x in v:
                v = v[:v.index(x)]
        version[n] = v
    version = version[:3]
    version = tuple(int(v) for v in version)
    if len(version) == 2:
        version = version + (0,)
    if not astuple:
        version = version[0]*10000 + version[1]*100 + version[0]
    return version


def torch_cuda_version(astuple=True):
    if not torch.cuda.is_available():
        return None
    version = torch.version.cuda.split('.')
    version = tuple(int(v) for v in version)
    if len(version) == 2:
        version = version + (0,)
    if not astuple:
        version = version[0]*10000 + version[1]*100 + version[0]
    return version


def torch_cudnn_version(astuple=True):
    if not torch.cuda.is_available():
        return None
    version = torch.backends.cudnn.version()
    version = (version//1000, version//100 % 10, version % 100)
    if not astuple:
        version = version[0]*10000 + version[1]*100 + version[0]
    return version


def torch_parallel_backend():
    # check if set by user
    valid_backends = ('OPENMP', 'NATIVE', 'NATIVE_TBB')
    backend = os.environ.get('FF_PARALLEL_BACKEND', None)
    if backend:
        backend = 'AT_PARALLEL_' + backend
        if backend not in valid_backends:
            backend = None
        return backend

    # else, find backend used by pytorch
    match = re.search('^ATen parallel backend: (?P<backend>.*)$',
                      torch._C._parallel_info(), re.MULTILINE)
    if match is None:
        return None
    backend = match.group('backend')
    if backend == 'OpenMP':
        return 'AT_PARALLEL_OPENMP'
    elif backend == 'native thread pool':
        return 'AT_PARALLEL_NATIVE'
    elif backend == 'native thread pool and TBB':
        return 'AT_PARALLEL_NATIVE_TBB'
    else:
        return None


def torch_abi():
  return str(int(torch._C._GLIBCXX_USE_CXX11_ABI))


def torch_omp_lib():
    torch_dir = os.path.dirname(os.path.abspath(torch.__file__))
    torch_library_dir = os.path.join(torch_dir, 'lib')
    if is_darwin():
        libtorch = os.path.join(torch_library_dir, 'libtorch.dylib')
        linked_libs = os.popen('otool -L "{}"'.format(libtorch)).read()
        if 'libiomp5' in linked_libs:
            return 'iomp5'
        elif 'libomp' in linked_libs:
            return 'omp'
        else:
            return None


def torch_libraries(cuda_home=None):
    version = torch_version(astuple=False)
    if version < 10500:
        libraries = ['c10', 'torch', 'torch_python']
        if cuda_home:
            libraries += ['cudart', 'c10_cuda']
    else:
        libraries = ['c10', 'torch_cpu', 'torch_python', 'torch']
        if cuda_home:
            libraries += ['cudart', 'c10_cuda', 'torch_cuda']
    if not cuda_home and torch_parallel_backend() == 'AT_PARALLEL_OPENMP':
        libraries += omp_libraries()
    return libraries


def torch_library_dirs(cuda_home=None, cudnn_home=None):
    torch_dir = os.path.dirname(os.path.abspath(torch.__file__))
    torch_library_dir = os.path.join(torch_dir, 'lib')
    library_dirs = [torch_library_dir]
    if cuda_home:
        if is_windows():
            library_dirs += [os.path.join(cuda_home, 'lib/x64')]
        elif os.path.exists(os.path.join(cuda_home, 'lib64')):
            library_dirs += [os.path.join(cuda_home, 'lib64')]
        elif os.path.exists(os.path.join(cuda_home, 'lib')):
            library_dirs += [os.path.join(cuda_home, 'lib')]
    if cudnn_home:
        if is_windows():
            library_dirs += [os.path.join(cudnn_home, 'lib/x64')]
        elif os.path.exists(os.path.join(cudnn_home, 'lib64')):
            library_dirs += [os.path.join(cudnn_home, 'lib64')]
        elif os.path.exists(os.path.join(cudnn_home, 'lib')):
            library_dirs += [os.path.join(cudnn_home, 'lib')]
    if not cuda_home and torch_parallel_backend() == 'AT_PARALLEL_OPENMP':
        library_dirs += omp_library_dirs()
    return library_dirs


def torch_include_dirs(cuda_home=None, cudnn_home=None):
    torch_dir = os.path.dirname(os.path.abspath(torch.__file__))
    torch_include_dir = os.path.join(torch_dir, 'include')
    include_dirs = [torch_include_dir,
                    os.path.join(torch_include_dir, 'torch', 'csrc', 'api', 'include'),
                    os.path.join(torch_include_dir, 'TH'),
                    os.path.join(torch_include_dir, 'THC')]
    if cuda_home:
        cuda_include_dir = os.path.join(cuda_home, 'include')
        if cuda_include_dir != '/usr/include':
            include_dirs += [cuda_include_dir]
    if cudnn_home:
        include_dirs += [os.path.join(cudnn_home, 'include')]
    if not cuda_home and torch_parallel_backend() == 'AT_PARALLEL_OPENMP':
        include_dirs += omp_include_dirs()
    return include_dirs


def cuda_check():
    use_cuda = int(os.environ.get('FF_USE_CUDA', '-1'))
    if use_cuda == 0:
        return None

    if not torch.cuda.is_available():
        if use_cuda == 1:
            raise RuntimeError('PyTorch was not compiled with CUDA, '
                               'but FF_USE_CUDA=1. Unset FF_USE_CUDA to '
                               'compile for CPU only.')
        elif use_cuda == -1:
            print('PyTorch was not compiled with CUDA. Compiling for CPU only.')
        return None

    torch_version = torch_cuda_version()
    home = cuda_home(torch_version[:2])
    if not home:
        if use_cuda == 1:
            raise RuntimeError(f'CUDA not found. '
                               f'Unset FF_USE_CUDA to compile for CPU only.')
        elif use_cuda == -1:
            print(f'-- Compiling for CPU only.')
        return None

    return home


def cuda_arch_flags():
    """
    Determine CUDA arch flags to use.

    For an arch, say "6.1", the added compile flag will be
    ``-gencode=arch=compute_61,code=sm_61``.
    For an added "+PTX", an additional
    ``-gencode=arch=compute_xx,code=compute_xx`` is added.

    See select_compute_arch.cmake for corresponding named and supported arches
    when building with CMake.
    """

    # Note: keep combined names ("arch1+arch2") above single names, otherwise
    # string replacement may not do the right thing
    named_arches = collections.OrderedDict([
        ('Kepler+Tesla', '3.7'),
        ('Kepler', '3.5+PTX'),
        ('Maxwell+Tegra', '5.3'),
        ('Maxwell', '5.0;5.2+PTX'),
        ('Pascal', '6.0;6.1+PTX'),
        ('Volta', '7.0+PTX'),
        ('Turing', '7.5+PTX'),
        ('Ampere', '8.0;8.6+PTX'),
    ])

    supported_arches = ['3.5', '3.7', '5.0', '5.2', '5.3', '6.0', '6.1', '6.2',
                        '7.0', '7.2', '7.5', '8.0', '8.6']
    valid_arch_strings = supported_arches + [s + "+PTX" for s in supported_arches]

    # The default is sm_30 for CUDA 9.x and 10.x
    # First check for an env var (same as used by the main setup.py)
    # Can be one or more architectures, e.g. "6.1" or "3.5;5.2;6.0;6.1;7.0+PTX"
    # See cmake/Modules_CUDA_fix/upstream/FindCUDA/select_compute_arch.cmake
    arch_list = os.environ.get('FF_CUDA_ARCH_LIST', 'mine')

    # If not given, look into libtorch_cuda
    if not arch_list or arch_list.lower() == 'all':
        cuobjdump = os.path.join(cuda_home(), 'bin', 'cuobjdump')
        torchdir = os.path.dirname(os.path.abspath(torch.__file__))
        libtorch = os.path.join(torchdir, 'lib')
        if is_windows():
            libtorch = os.path.join(libtorch, 'torch_cuda.lib')
        else:
            assert not is_darwin()
            libtorch = os.path.join(libtorch, 'libtorch_cuda.so')
        arch_list = os.popen(cuobjdump + " '" + libtorch + \
                             "' -lelf | awk -F. '{print $3}' | " \
                             "grep sm | sort -u").read().split('\n')
        arch_list = [arch[3] + '.' + arch[4] for arch in arch_list if arch]
        ptx_list = os.popen(cuobjdump + " '" + libtorch + \
                             "' -lptx | awk -F. '{print $3}' | " \
                             "grep sm | sort -u").read().split('\n')
        ptx_list = [arch[3] + '.' + arch[4] for arch in ptx_list if arch]
        arch_list = [arch + '+PTX' if arch in ptx_list else arch
                     for arch in arch_list]
    elif arch_list == 'mine':
        #   this bit was in the torch extension util but I have replaced
        #   it with the bit above that looks into libtorch
        capability = torch.cuda.get_device_capability()
        arch_list = ['{}.{}'.format(capability[0], capability[1])]
    else:
        # Deal with lists that are ' ' separated (only deal with ';' after)
        arch_list = arch_list.replace(' ', ';')
        # Expand named arches
        for named_arch, archval in named_arches.items():
            arch_list = arch_list.replace(named_arch, archval)

        arch_list = arch_list.split(';')

    flags = []
    for arch in arch_list:
        if arch not in valid_arch_strings:
            raise ValueError("Unknown CUDA arch ({}) or GPU not supported".format(arch))
        else:
            num = arch[0] + arch[2]
            flags.append('-gencode=arch=compute_{},code=sm_{}'.format(num, num))
            if arch.endswith('+PTX'):
                flags.append('-gencode=arch=compute_{},code=compute_{}'.format(num, num))

    return list(set(flags))


def torch_extension_flags(name):
    return ['-DTORCH_EXTENSION_NAME={}'.format(name),
            '-DTORCH_API_INCLUDE_EXTENSION_H']


def gcc_clang_flags():
    flags = ['-fPIC', '-std=c++14']
    if is_darwin() and darwin_cc_type() == 'apple_clang':
        flags += ['-stdlib=libc++']
    return flags


def msvc_flags():
    return ['/MD', '/wd4819', '/EHsc']


def nvcc_flags():
    return [
      '-x=cu',
      '-D__CUDA_NO_HALF_OPERATORS__',
      '-D__CUDA_NO_HALF_CONVERSIONS__',
      '-D__CUDA_NO_HALF2_OPERATORS__',
      '--expt-relaxed-constexpr']


def darwin_cc_type():
    CC = os.environ.get('CC', 'clang')
    CC_name = os.popen(CC + ' --version').read().split(' ')[0]
    if CC_name == 'Apple':
        CC_type = 'apple_clang'
    elif CC_name == 'clang':
        CC_type = 'other_clang'
    else:
        CC_type = 'other'
    return CC_type


def find_omp_darwin():
    """Set the correct openmp flag on MacOS.

    LLVM's clang has both GCC's and Intel's implementations of OpenMP
    (gomp, omp5), which can be specifically used with the flag
    '-fopenmp=libiomp5' or '-fopenmp=libgomp'. If no implementation is
    specified, the behaviour depends on the version of LLVM
    (it used to be gomp, now I believe it is iomp5). Note that iomp5 is
    binary compatible with gomp (does that mean we can link with both
    without conflicts?).

    Apple's clang does not ship any OpenMP implementation, so the user
    needs to install one herself, which we need to find and reference
    properly.

    Return (cflag, lflag, lib_name, lib_dir)."""

    # TODO: LLVM's clang embeds OpenMP for version >= 3.8.0
    #       I need to add a special case for earlier versions.
    #       There are other ABI incompatibilities as well:
    #           https://openmp.llvm.org

    # Various references that helped me in this maze:
    #   https://iscinumpy.gitlab.io/post/omp-on-high-sierra/
    #   https://stackoverflow.com/questions/37362414/
    #   https://reviews.llvm.org/D2841

    def find_lib(names):
        for name in names:
            if not name:
                continue
            dirs = ['.', '/usr/', '/usr/local/', '/opt/local/', '/usr/local/opt/libomp/']
            if os.environ.get('LD_LIBRARY_PATH'):
                dirs += os.environ.get('LD_LIBRARY_PATH').split(':')
            for dir in dirs:
                if os.path.exists(os.path.join(dir, 'lib', 'lib' + name + '.dylib')):
                    return name, dir
        return None, None

    # First, check which clang we're dealing with
    # (gcc, apple clang or external clang)
    CC_type = darwin_cc_type()

    # If not apple clang: openmp should be packaged with the compiler:
    if CC_type != 'apple_clang':
        flag = '-fopenmp'
        if CC_type == 'other_clang':
            if torch_omp_lib() == 'iomp5':
                flag = '-fopenmp=libiomp5'
            elif torch_omp_lib() == 'omp':
                flag = '-fopenmp=libgomp'
        return [flag], [flag], [], None

    # Else, opnemp is no different than any other dependency

    # First, check if omp/iomp5 has been installed (e.g., using homebrew)
    lib_name, lib_dir = find_lib([torch_omp_lib(), 'iomp5', 'omp'])

    if lib_name is None:
        # OpenMP not found.
        # Let's just hope that the compiler knows what it's doing.
        return ['-fopenmp'], ['-fopenmp'], [], None
    else:
        return ['-Xpreprocessor', '-fopenmp'], [], [], lib_dir
        # return ['-Xpreprocessor', '-fopenmp'], [lib_name], [], lib_dir

        # It is super weird:
        # - precompiled torch wheels on mac link against libomp (or libiomp5)
        # - but openmp was not detected in their compilation chain, so
        #   it was actually compiled *without* openmp (pragmas were not used,
        #   parallel loops are actually sequential, (set/get)_num_threads
        #   is always 1)
        # - I can't get my mac to link against the correct omp lib (it
        #   links against libomp even though I ask it to link against
        #   libiomp5)
        # - If I don't link against openmp at all, things seem to work (!!)
        # - I really need to rewrite our compilation stuff anyway


def omp_flags():
    if is_windows():
        return ['/openmp']
    elif is_darwin():
        return find_omp_darwin()[0]
    else:
        return ['-fopenmp']


def omp_link_flags():
    if is_darwin():
        return find_omp_darwin()[1]
    else:
        return []


def omp_libraries():
    if is_darwin():
        return find_omp_darwin()[2]
    else:
        return []


def omp_library_dirs():
    if is_darwin():
        ompdir = find_omp_darwin()[3]
        return [os.path.join(ompdir, 'lib')] if ompdir else []
    else:
        return []


def omp_include_dirs():
    if is_darwin():
        ompdir = find_omp_darwin()[3]
        return [os.path.join(ompdir, 'include')] if ompdir else []
    else:
        return []


def common_flags():
    if is_windows():
        return msvc_flags()
    else:
        return gcc_clang_flags()


def torch_flags(cuda=False):
    version = torch_version()
    version = version[0]*10000+version[1]*100+version[2]
    flags = ['-DNI_TORCH_VERSION=' + str(version)]
    backend = torch_parallel_backend()
    flags += [
        '-D' + torch_parallel_backend() + '=1',
        '-D_GLIBCXX_USE_CXX11_ABI=' + torch_abi()]
    if not cuda and backend == 'AT_PARALLEL_OPENMP':
        flags += omp_flags()
    return flags


def torch_link_flags(cuda=False):
    backend = torch_parallel_backend()
    flags = []
    if not cuda and backend == 'AT_PARALLEL_OPENMP':
        flags += omp_link_flags()
    return flags


def common_links_flags():
    if is_darwin() and darwin_cc_type() == 'apple_clang':
        return ['-stdlib=libc++']
    return []


def cuda_flags():
    flags = nvcc_flags() + cuda_arch_flags()
    if is_windows():
        for flag in common_flags():
            # if flag == '-fPIC':
            #     continue
            flags = ['-Xcompiler', flag] + flags
    else:
        for flag in common_flags():
            # if flag == '-fPIC':
            #     continue
            flags += ['--compiler-options', flag]
    return flags


def abspathC(files, glob=False):
    scriptdir = os.path.abspath(os.path.dirname(__file__))
    sourcedir = os.path.join(scriptdir, 'fastfields', 'ext')
    files = [os.path.join(sourcedir, f) for f in files]
    if glob:
        files = [list(glob_(f)) for f in files]
        files = [f for file in files for f in file]
    return files


