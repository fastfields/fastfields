import torch as _  # Necessary for linking extensions

# set version number
import os
vfilename = os.path.join(os.path.dirname(__file__), '..', 'VERSION')
if not os.path.exists(vfilename):
    __version__ = None
else:
    with open(vfilename, 'rt') as vfile:
        __version__ = vfile.read().strip()
del vfilename

# The legacy jit executor (used by torchscript) was the default
# until v1.6 (included), but sometimes sets `requires_grad = True`
# on new variables even though it should not, making TS code
# virtually unusable. Here, we force the use of the 
# profiling executor (default from v1.7).
try:
    from torch._C import _jit_set_profiling_executor
    _jit_set_profiling_executor(True)
except ImportError:
    from warnings import warn
    warn('Could not use profiling executor. Parts may break.', RuntimeWarning)

# TODO:
# . check compatible cuda versions between torch and nitorch
#   (see torchvision.extension)

from .grid import *
from .resize import *
from .solve import *

