import numpy as np
from types import MethodType, FunctionType
from wrapper import *
from ctypes import c_int, c_double, POINTER


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
        self.g_tau_t_A = np.ascontiguousarray(np.empty((4, 4)), dtype=np.complex)
        self.g_tau_t_B = np.ascontiguousarray(np.empty((4, 4)), dtype=np.complex)

        N = len(self.energies_A)
        self.dyn_rhoA = np.ascontiguousarray(np.zeros((int(N * (N + 3) / 2), params.timeDIM), dtype=np.complex))
        self.dyn_rhoB = np.ascontiguousarray(np.zeros((int(N * (N + 3) / 2), params.timeDIM), dtype=np.complex))

    def call_raman_control_function(self, molA, molB, params, guess):
        """

        """
        t0_R_1 = params.t0_R_1 * params.timeAMP
        t0_R_2 = params.t0_R_2 * params.timeAMP
        t0_EE = params.t0_EE * params.timeAMP
        width_R_1 = params.timeAMP / params.width_R_1
        width_R_2 = params.timeAMP / params.width_R_2
        width_EE = params.timeAMP / params.width_EE

        func_params = Parameters()

        func_params.time = self.time.ctypes.data_as(POINTER(c_double))
        func_params.rho_0 = self.rho_0.ctypes.data_as(POINTER(c_complex))

        func_params.A_R_1 = params.A_R_1
        func_params.width_R_1 = width_R_1
        func_params.t0_R_1 = t0_R_1

        func_params.A_R_2 = params.A_R_2
        func_params.width_R_2 = width_R_2
        func_params.t0_R_2 = t0_R_2

        func_params.A_EE = params.A_EE
        func_params.width_EE = width_EE
        func_params.t0_EE = t0_EE

        func_params.w_R_1 = params.w_R_1
        func_params.w_R_2 = params.w_R_2
        func_params.w_v_1 = params.w_v_1
        func_params.w_v_2 = params.w_v_2
        func_params.w_EE = params.w_EE

        func_params.nDIM = len(self.energies_A)
        func_params.timeDIM = len(self.time)

        func_params.field_out = self.field_t.ctypes.data_as(POINTER(c_complex))

        func_params.lower_bounds = params.lower_bounds.ctypes.data_as(POINTER(c_double))
        func_params.lower_bounds = params.lower_bounds.ctypes.data_as(POINTER(c_double))
        func_params.guess = guess.ctypes.data_as(POINTER(c_double))

        RamanControlFunction(molA, molB, func_params)

        return


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import time
    from multiprocessing import Pool
    from itertools import *
    from functools import partial

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
    timeAMP = 20000.

    lower_bounds = np.asarray([0.00015, 0.00015, 0.9*energies_A[1], 0.9*energies_A[1], 1.*energy_factor, 1.*energy_factor])
    upper_bounds = np.asarray([0.00027, 0.00027, 1.1*energies_A[1], 1.1*energies_A[1], 1.6*energy_factor, 1.6*energy_factor])

    guess = np.asarray([np.random.uniform(lower_bounds[i], upper_bounds[i]) for i in range(len(lower_bounds))])
    print(guess)

    params = ADict(
        energy_factor=energy_factor,
        time_factor=time_factor,
        timeDIM=40000,
        timeAMP=20000.,

        A_R_1=0.000576595,
        width_R_1=1.50,
        t0_R_1=-0.00,

        A_R_2=0.000576595,
        width_R_2=1.50,
        t0_R_2=-0.00,

        A_EE=0.000366972,
        width_EE=19.32,
        t0_EE=0.55,

        w_R_1=1.4 * energy_factor,
        w_R_2=1.4 * energy_factor,
        w_v_1=energies_A[1],
        w_v_2=energies_A[1],
        w_EE=(energies_A[2] - energies_A[1]),
        rho_0=rho_0,

        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        guess=guess
    )

    FourLevels = dict(
        energies_A=energies_A,
        energies_B=energies_B,
        gamma_decay=gamma_decay,
        gamma_pure_dephasing=gamma_pure_dephasing,
        mu=mu,
    )
    molecules = RamanControl(params, **FourLevels)

    molA = Molecule()
    molB = Molecule()

    molA.energies = molecules.energies_A.ctypes.data_as(POINTER(c_double))
    molB.energies = molecules.energies_B.ctypes.data_as(POINTER(c_double))
    molA.gamma_decay = molecules.gamma_decay.ctypes.data_as(POINTER(c_double))
    molB.gamma_decay = molecules.gamma_decay.ctypes.data_as(POINTER(c_double))
    molA.gamma_pure_dephasing = molecules.gamma_pure_dephasing.ctypes.data_as(POINTER(c_double))
    molB.gamma_pure_dephasing = molecules.gamma_pure_dephasing.ctypes.data_as(POINTER(c_double))
    molA.mu = molecules.mu.ctypes.data_as(POINTER(c_complex))
    molB.mu = molecules.mu.ctypes.data_as(POINTER(c_complex))
    molA.rho = molecules.rhoA.ctypes.data_as(POINTER(c_complex))
    molB.rho = molecules.rhoB.ctypes.data_as(POINTER(c_complex))
    molA.dyn_rho = molecules.dyn_rhoA.ctypes.data_as(POINTER(c_complex))
    molB.dyn_rho = molecules.dyn_rhoB.ctypes.data_as(POINTER(c_complex))
    molA.g_tau_t = molecules.g_tau_t_A.ctypes.data_as(POINTER(c_complex))
    molB.g_tau_t = molecules.g_tau_t_B.ctypes.data_as(POINTER(c_complex))

    molecules.call_raman_control_function(molA, molB, params, guess)

    fig1, axes = plt.subplots(nrows=5, ncols=1, sharex=True)
    axes[0].plot(molecules.time, molecules.field_t.real, 'r')

    axes[1].plot(molecules.time, molecules.dyn_rhoA[0, :], label='1')
    axes[1].plot(molecules.time, molecules.dyn_rhoA[1, :], label='2')
    axes[1].plot(molecules.time, molecules.dyn_rhoA[2, :], label='3')
    axes[1].plot(molecules.time, molecules.dyn_rhoA[3, :], label='4')
    axes[1].legend()

    axes[2].plot(molecules.time, molecules.dyn_rhoB[0, :], label='1')
    axes[2].plot(molecules.time, molecules.dyn_rhoB[1, :], label='2')
    axes[2].plot(molecules.time, molecules.dyn_rhoB[2, :], label='3')
    axes[2].plot(molecules.time, molecules.dyn_rhoB[3, :], label='4')
    axes[2].legend()

    axes[3].plot(molecules.time, molecules.dyn_rhoA[4, :])
    axes[3].plot(molecules.time, molecules.dyn_rhoA[5, :])
    axes[3].plot(molecules.time, molecules.dyn_rhoA[6, :])
    axes[3].plot(molecules.time, molecules.dyn_rhoA[7, :])
    axes[3].plot(molecules.time, molecules.dyn_rhoA[8, :])
    axes[3].plot(molecules.time, molecules.dyn_rhoA[9, :])

    axes[4].plot(molecules.time, molecules.dyn_rhoB[4, :])
    axes[4].plot(molecules.time, molecules.dyn_rhoB[5, :])
    axes[4].plot(molecules.time, molecules.dyn_rhoB[6, :])
    axes[4].plot(molecules.time, molecules.dyn_rhoB[7, :])
    axes[4].plot(molecules.time, molecules.dyn_rhoB[8, :])
    axes[4].plot(molecules.time, molecules.dyn_rhoB[9, :])

    print(np.diag(molecules.rhoA.real)[2:].sum() - np.diag(molecules.rhoB.real)[2:].sum())

    print(molecules.rhoA.real, "\n")
    print(molecules.rhoB.real, "\n")

    plt.show()
