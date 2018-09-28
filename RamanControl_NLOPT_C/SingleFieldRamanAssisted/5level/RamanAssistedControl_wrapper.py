import os
import ctypes
from ctypes import c_int, c_double, POINTER, Structure


__doc__ = """
Python wrapper for RamanAssistedControl.c
Compile with:
gcc -O3 -shared -o RamanAssistedControl.so RamanAssistedControl.c -lm -fopenmp -fPIC
"""


class c_complex(Structure):
    """
    Complex double ctypes
    """
    _fields_ = [
        ('real', c_double),
        ('imag', c_double)
    ]


class Parameters(Structure):
    """
    Parameters structure ctypes
    """
    _fields_ = [
        ('time', POINTER(c_double)),
        ('rho_0', POINTER(c_complex)),

        ('A_R', c_double),
        ('width_R', c_double),
        ('t0_R', c_double),

        ('A_EE', c_double),
        ('width_EE', c_double),
        ('t0_EE', c_double),

        ('w_R', c_double),
        ('w_v1', c_double),
        ('w_v2', c_double),
        ('w_v3', c_double),
        ('w_EE1', c_double),
        ('w_EE2', c_double),
        ('w_EE3', c_double),

        ('nDIM', c_int),
        ('timeDIM', c_int),

        ('field_out', POINTER(c_complex)),
        ('field_grad_A_R', POINTER(c_complex)),
        ('field_grad_A_EE', POINTER(c_complex)),

        ('lower_bounds', POINTER(c_double)),
        ('upper_bounds', POINTER(c_double)),
        ('guess', POINTER(c_double)),
        ('MAX_EVAL', c_int)
    ]


class Molecule(Structure):
    """
    Parameters structure ctypes
    """
    _fields_ = [
        ('energies', POINTER(c_double)),
        ('gamma_decay', POINTER(c_double)),
        ('gamma_pure_dephasing', POINTER(c_double)),
        ('mu', POINTER(c_complex)),
        ('rho', POINTER(c_complex)),
        ('dyn_rho', POINTER(c_complex)),
        ('g_tau_t', POINTER(c_complex))
    ]


try:
    # Load the shared library assuming that it is in the same directory
    lib = ctypes.cdll.LoadLibrary(os.getcwd() + "/RamanAssistedControl_5lvl.so")
except OSError:
    raise NotImplementedError(
        """
        The library is absent. You must compile the C shared library using the commands:
        gcc -O3 -shared -o RamanAssistedControl.so RamanAssistedControl.c -lnlopt -lm -fopenmp -fPIC
        """
    )

#####################################################
#                                                   #
#          Declaring RamanControlFunction           #
#                                                   #
#####################################################

lib.RamanControlFunction.argtypes = (
    POINTER(Molecule),      # molecule molA
    POINTER(Molecule),      # molecule molB
    POINTER(Parameters),    # parameter field_params
)
lib.RamanControlFunction.restype = POINTER(c_complex)


def RamanControlFunction(molA, molB, func_params):
    return lib.RamanControlFunction(
        molA,
        molB,
        func_params
    )


lib.Propagate.argtypes = (
    POINTER(Molecule),      # molecule mol
    POINTER(Parameters),    # parameter field_params
)
lib.Propagate.restype = None


def PropagateFunction(mol, func_params):
    return lib.Propagate(
        mol,
        func_params
    )