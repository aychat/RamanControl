import os
import ctypes
from ctypes import c_int, c_double, POINTER, Structure


__doc__ = """
Python wrapper for PropagateOptimize.c
Compile with:
gcc -O3 -shared -o PropagateOptimize.so PropagateOptimize.c -lnlopt -lm -fopenmp -fPIC
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

        ('A_R_1', c_double),
        ('width_R_1', c_double),
        ('t0_R_1', c_double),

        ('A_R_2', c_double),
        ('width_R_2', c_double),
        ('t0_R_2', c_double),

        ('A_EE', c_double),
        ('width_EE', c_double),
        ('t0_EE', c_double),

        ('w_R_1', c_double),
        ('w_R_2', c_double),
        ('w_v_1', c_double),
        ('w_v_2', c_double),
        ('w_EE', c_double),

        ('nDIM', c_int),
        ('timeDIM', c_int),

        ('field_out', POINTER(c_complex)),

        ('lower_bounds', POINTER(c_double)),
        ('upper_bounds', POINTER(c_double)),
        ('guess', POINTER(c_double)),
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
    lib = ctypes.cdll.LoadLibrary(os.getcwd() + "/PropagateOptimize.so")
except OSError:
    raise NotImplementedError(
        """
        The library is absent. You must compile the C shared library using the commands:
        gcc -O3 -shared -o PropagateOptimize.so PropagateOptimize.c -lnlopt -lm -fopenmp -fPIC
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
