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
        ('nDIM', c_int),
        ('timeDIM', c_int)
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
        ('dyn_rho', POINTER(c_complex))
    ]


try:
    # Load the shared library assuming that it is in the same directory
    lib = ctypes.cdll.LoadLibrary(os.getcwd() + "/Propagate.so")
except OSError:
    raise NotImplementedError(
        """
        The library is absent. You must compile the C shared library using the commands:
        gcc -O3 -shared -o Propagate.so Propagate.c -lnlopt -lm -fopenmp -fPIC
        """
    )

#####################################################
#                                                   #
#          Declaring RamanControlFunction           #
#                                                   #
#####################################################

lib.Propagate.argtypes = (
    POINTER(Molecule),      # molecule mol
    POINTER(Parameters),    # parameter field_params
    POINTER(c_complex)      # c_complex field
)
lib.Propagate.restype = c_double


def PropagateFunction(mol, func_params, field):
    print("hi")
    return lib.Propagate(
        mol,
        func_params,
        field.ctypes.data_as(POINTER(c_complex))
    )