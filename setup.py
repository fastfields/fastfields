"""
SETUP SCRIPT
============

Important environment variables:
CUDA_HOME
    Path to cuda toolkit (nvcc, etc.)
    By default, we try to find it in standard locations.
FF_USE_CUDA ({0, 1})
    Whether to compile with cuda support.
    By default, yes if a correct CUDA toolkit is found.
FF_CUDA_ARCH_LIST ({all, mine, Kepler, Maxwell, Pascal, Volta, Turing, Ampere})
    If "mine", we only compile for the architecture of the current GPU (default).
    If "all", compile for all architectures.
    If space-separated list of architectures, compile only for these archs.
FF_PARALLEL_BACKEND ({OPENMP, NATIVE, NATIVE_TBB})
    Parallel backend to use in the extension.
    By default, use the same as libtorch against which we compile.
FF_VERSION
    Overwrite the version number in the file VERSION
    Useful to distribute packages compiled against specific pytorch/cuda
"""
from setuptools import setup, find_packages
import os
from configparser import ConfigParser

SETUP_KWARGS = {}

config = ConfigParser()
rootdir = os.path.dirname(os.path.abspath(__file__))
config.read(os.path.join(rootdir, 'setup.cfg'))
INSTALL_REQUIRES = config['options']['install_requires']
INSTALL_REQUIRES = INSTALL_REQUIRES.split('\n')

from setup_cext import prepare_extensions, build_ext
from torch import __version__ as torch_version
torch_version = torch_version.split('.')
if '.'.join(torch_version[:2]) == '1.7':
    torch_version = '.'.join(torch_version[:3])  # we need the patch
else:
    torch_version = '.'.join(torch_version[:3])
SETUP_KWARGS['ext_package'] = 'fastfields'
SETUP_KWARGS['ext_modules'] = prepare_extensions()
INSTALL_REQUIRES += [f'torch=={torch_version}']
SETUP_KWARGS['install_requires'] = INSTALL_REQUIRES
SETUP_KWARGS['cmdclass'] = {'build_ext': build_ext}

version = os.environ.get('FF_VERSION', None)
vname = os.path.join(os.path.dirname(__file__), 'VERSION')
if version:
    with open(vname, 'wt') as f:
        f.write(version)
else:
    with open(vname, 'rt') as vfile:
        version = vfile.read().strip()

setup(
    packages=find_packages(),
    version=version,
    **SETUP_KWARGS
)
