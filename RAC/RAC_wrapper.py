import os
import numpy as np
import ctypes
from ctypes import c_int, c_double, POINTER, Structure
from collections import namedtuple


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
        ('w_v', c_double),
        ('w_EE', c_double),
        ('nDIM', c_int),
        ('timeDIM', c_int),
        ('field_out', POINTER(c_complex)),
        ('field_grad_A_R', POINTER(c_complex)),
        ('field_grad_A_EE', POINTER(c_complex))
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
        ('g_t_t', POINTER(c_complex))
    ]


try:
    # Load the shared library assuming that it is in the same directory
    lib = ctypes.cdll.LoadLibrary(os.getcwd() + "/RAC.so")
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


def RamanControlFunction(params):

    molecules = namedtuple('molecules', ['molA', 'molB'])

    nDIM = len(params.energies_A)

    rhoA = np.zeros((nDIM, nDIM), dtype=np.complex)
    rhoB = np.zeros((nDIM, nDIM), dtype=np.complex)
    dyn_rhoA = np.zeros((int(nDIM * (nDIM + 3) / 2), params.timeDIM), dtype=np.complex)
    dyn_rhoB = np.zeros((int(nDIM * (nDIM + 3) / 2), params.timeDIM), dtype=np.complex)
    g_tau_t_A = np.zeros((nDIM, nDIM), dtype=np.complex)
    g_tau_t_B = np.zeros((nDIM, nDIM), dtype=np.complex)

    molecules.molA = Molecule(
        energies=params.energies_A.ctypes.data_as(POINTER(c_double)),
        gamma_decay=params.gamma_decay.ctypes.data_as(POINTER(c_double)),
        gamma_pure_dephasing=params.gamma_pure_dephasing.ctypes.data_as(POINTER(c_double)),
        mu=params.mu.ctypes.data_as(POINTER(c_complex)),
        rho=rhoA.ctypes.data_as(POINTER(c_complex)),
        dyn_rho=dyn_rhoA.ctypes.data_as(POINTER(c_complex)),
        g_t_t=g_tau_t_A.ctypes.data_as(POINTER(c_complex))
    )

    molecules.molB = Molecule(
        energies=params.energies_B.ctypes.data_as(POINTER(c_double)),
        gamma_decay=params.gamma_decay.ctypes.data_as(POINTER(c_double)),
        gamma_pure_dephasing=params.gamma_pure_dephasing.ctypes.data_as(POINTER(c_double)),
        mu=params.mu.ctypes.data_as(POINTER(c_complex)),
        rho=rhoB.ctypes.data_as(POINTER(c_complex)),
        dyn_rho=dyn_rhoB.ctypes.data_as(POINTER(c_complex)),
        g_t_t=g_tau_t_B.ctypes.data_as(POINTER(c_complex))
    )

    func_params = Parameters(
        time=np.linspace(-params.timeAMP, params.timeAMP, params.timeDIM).ctypes.data_as(POINTER(c_double)),
        rho_0=params.rho_0.ctypes.data_as(POINTER(c_complex)),

        A_R=params.A_R,
        width_R=params.width_R,
        t0_R=params.t0_R,

        A_EE=params.A_EE,
        width_EE=params.width_EE,
        t0_EE=params.t0_EE,

        w_R=params.w_R,
        w_v=params.w_v,
        w_EE=params.w_EE,

        nDIM=nDIM,
        timeDIM=params.timeDIM,

        field_out=params.field_out.ctypes.data_as(POINTER(c_complex)),
        field_grad_A_R=params.field_grad_A_R.ctypes.data_as(POINTER(c_complex)),
        field_grad_A_EE=params.field_grad_A_EE.ctypes.data_as(POINTER(c_complex))
    )


    return lib.RamanControlFunction(
        molecules.molA,
        molecules.molB,
        func_params
    )
