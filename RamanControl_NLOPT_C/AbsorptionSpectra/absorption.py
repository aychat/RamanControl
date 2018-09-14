import numpy as np
from types import MethodType, FunctionType
from wrapper import *
from ctypes import c_int, c_double, POINTER, Structure


class ADict(dict):
    """
    Dictionary where you can access keys as attributes
    """
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            dict.__getattribute__(self, item)


class RamanControl:
    """
    """

    def __init__(self, params, **kwargs):
        """
        __init__ function call to initialize variables from the
        parameters for the class instance provided in __main__ and
        add new variables for use in other functions in this class.
        """

        for name, value in kwargs.items():
            if isinstance(value, FunctionType):
                setattr(self, name, MethodType(value, self, self.__class__))
            else:
                setattr(self, name, value)

        self.time = np.linspace(-params.timeAMP, params.timeAMP, params.timeDIM)
        self.field_t = np.empty(params.timeDIM, dtype=np.complex)

        self.gamma_decay = np.ascontiguousarray(self.gamma_decay)
        self.gamma_pure_dephasing = np.ascontiguousarray(self.gamma_pure_dephasing)
        self.mu = np.ascontiguousarray(self.mu)
        self.rho_0 = np.ascontiguousarray(params.rho_0)
        self.rhoA = np.ascontiguousarray(params.rho_0.copy())
        self.rhoB = np.ascontiguousarray(params.rho_0.copy())
        self.energies_A = np.ascontiguousarray(self.energies_A)
        self.energies_B = np.ascontiguousarray(self.energies_B)

        N = len(self.energies_A)
        self.dyn_rhoA = np.ascontiguousarray(np.zeros((int(N * (N + 3) / 2), params.timeDIM), dtype=np.complex))
        self.dyn_rhoB = np.ascontiguousarray(np.zeros((int(N * (N + 3) / 2), params.timeDIM), dtype=np.complex))

    def create_molecules(self, molA, molB):
        molA.energies = self.energies_A.ctypes.data_as(POINTER(c_double))
        molB.energies = self.energies_B.ctypes.data_as(POINTER(c_double))
        molA.gamma_decay = self.gamma_decay.ctypes.data_as(POINTER(c_double))
        molB.gamma_decay = self.gamma_decay.ctypes.data_as(POINTER(c_double))
        molA.gamma_pure_dephasing = self.gamma_pure_dephasing.ctypes.data_as(POINTER(c_double))
        molB.gamma_pure_dephasing = self.gamma_pure_dephasing.ctypes.data_as(POINTER(c_double))
        molA.mu = self.mu.ctypes.data_as(POINTER(c_complex))
        molB.mu = self.mu.ctypes.data_as(POINTER(c_complex))
        molA.rho = self.rhoA.ctypes.data_as(POINTER(c_complex))
        molB.rho = self.rhoB.ctypes.data_as(POINTER(c_complex))
        molA.dyn_rho = self.dyn_rhoA.ctypes.data_as(POINTER(c_complex))
        molB.dyn_rho = self.dyn_rhoB.ctypes.data_as(POINTER(c_complex))

    def create_parameters(self, func_params):
        func_params.time = self.time.ctypes.data_as(POINTER(c_double))
        func_params.rho_0 = self.rho_0.ctypes.data_as(POINTER(c_complex))
        func_params.nDIM = len(self.energies_A)
        func_params.timeDIM = len(self.time)


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from itertools import *

    np.set_printoptions(precision=4)
    energy_factor = 1. / 27.211385
    time_factor = .02418

    energies_A = np.array((0.000, 0.07439, 1.94655, 2.02094)) * energy_factor
    energies_B = np.array((0.000, 0.07639, 1.92655, 2.02094)) * energy_factor
    rho_0 = np.zeros((len(energies_A), len(energies_A)), dtype=np.complex)
    rho_0[0, 0] = 1. + 0j
    mu = 4.97738*np.ones_like(rho_0)
    np.fill_diagonal(mu, 0j)

    pop_relax = 2.418884e-8
    electronic_dephasing = 2.418884e-4
    vibrational_dephasing = 1.422872e-5

    gamma_decay = np.ones((4, 4)) * pop_relax
    np.fill_diagonal(gamma_decay, 0.0)
    gamma_decay = np.tril(gamma_decay)
    gamma_pure_dephasing = np.ones_like(gamma_decay) * electronic_dephasing
    np.fill_diagonal(gamma_pure_dephasing, 0.0)
    gamma_pure_dephasing[1, 0] = vibrational_dephasing
    gamma_pure_dephasing[3, 2] = vibrational_dephasing
    gamma_pure_dephasing[0, 1] = vibrational_dephasing
    gamma_pure_dephasing[2, 3] = vibrational_dephasing
    timeAMP = 1000.

    params = ADict(
        energy_factor=energy_factor,
        time_factor=time_factor,
        timeDIM=40000,
        timeAMP=timeAMP,
        rho_0=rho_0
    )

    FourLevels = dict(
        energies_A=energies_A,
        energies_B=energies_B,
        gamma_decay=gamma_decay,
        gamma_pure_dephasing=gamma_pure_dephasing,
        mu=mu,
    )
    molecules = RamanControl(params, **FourLevels)

    N = 100
    omega = 1239.84193 / np.linspace(300., 800., 1000)
    t = molecules.time

    spectraA = np.zeros(N, dtype=np.complex)
    spectraB = np.zeros(N, dtype=np.complex)

    molA = Molecule()
    molB = Molecule()
    molecules.create_molecules(molA, molB)

    for i in range(N):
        func_params = Parameters()
        field = 0.0001 * np.exp(-(t**2)/(2.*200**2)) * np.cos(omega[i] * t) + 0j
        PropagateFunction(molA, func_params, field)
