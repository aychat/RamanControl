import os
import ctypes
from ctypes import c_int, c_double, POINTER, Structure

__doc__ = """
Python wrapper for RamanControl.c
Compile with:
gcc -O3 -shared -o RamanControl.so RamanControl.c -lm -fopenmp -fPIC
"""


class c_complex(Structure):
    """
    Complex double ctypes
    """
    _fields_ = [
        ('real', c_double),
        ('imag', c_double)
    ]


class parameters(Structure):
    """
    Parameters structure ctypes
    """
    _fields_ = [
        ('gamma_decay', POINTER(c_double)),
        ('gamma_pure_dephasing', POINTER(c_double)),
        ('mu', POINTER(c_complex)),
        ('rho_0', POINTER(c_complex)),
        ('energies', POINTER(c_double)),

        ('time', POINTER(c_double)),

        ('A_R', c_double),
        ('width_R', c_double),
        ('t0_R', c_double),
        ('A_EE', c_double),
        ('width_EE', c_double),
        ('t0_EE', c_double),
        ('w_R', c_double),
        ('w_v', c_double),
        ('w_EE', c_double),

        ('nDIM', c_int),
        ('timeDIM', c_int)
    ]


try:
    # Load the shared library assuming that it is in the same directory
    lib = ctypes.cdll.LoadLibrary(os.getcwd() + "/RamanControl.so")
except OSError:
    raise NotImplementedError(
        """
        The library is absent. You must compile the C shared library using the commands:
        gcc -O3 -shared -o RamanControl.so RamanControl.c -lm -fopenmp -fPIC
        """
    )

#####################################################
#                                                   #
#        Declaring the function Propagate           #
#                                                   #
#####################################################

lib.Propagate.argtypes = (
    POINTER(c_complex),     # cmplx* out, Array to store L[Q]
    POINTER(c_complex),     # cmplx* dyn_rho,
    POINTER(c_complex),     # cmplx* dyn_coh,
    POINTER(c_complex),     # cmplx* field,
    POINTER(parameters),    # parameter field_params
)
lib.Propagate.restype = None


def Propagate(out, dyn_rho, dyn_coh, field, func_params):
    return lib.Propagate(
        out.ctypes.data_as(POINTER(c_complex)),
        dyn_rho.ctypes.data_as(POINTER(c_complex)),
        dyn_coh.ctypes.data_as(POINTER(c_complex)),
        field.ctypes.data_as(POINTER(c_complex)),
        func_params
    )


lib.CalculateField.argtypes = (
    POINTER(c_complex),     # cmplx* field, , Array to store E(t)
    POINTER(parameters),    # parameter field_params
)
lib.CalculateField.restype = None


def CalculateField(field, func_params):
    return lib.CalculateField(
        field.ctypes.data_as(POINTER(c_complex)),
        func_params
    )
