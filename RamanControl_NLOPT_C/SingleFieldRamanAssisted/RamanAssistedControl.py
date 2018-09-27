import numpy as np
from types import MethodType, FunctionType
from RamanAssistedControl_wrapper import *
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
        self.field_grad_A_R = np.empty(params.timeDIM, dtype=np.complex)
        self.field_grad_A_EE = np.empty(params.timeDIM, dtype=np.complex)

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
        self.dyn_rhoA = np.ascontiguousarray(np.zeros((N+1, params.timeDIM), dtype=np.complex))
        self.dyn_rhoB = np.ascontiguousarray(np.zeros((N+1, params.timeDIM), dtype=np.complex))

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
        molA.g_tau_t = self.g_tau_t_A.ctypes.data_as(POINTER(c_complex))
        molB.g_tau_t = self.g_tau_t_B.ctypes.data_as(POINTER(c_complex))

    def create_parameters(self, func_params, params):
        func_params.time = self.time.ctypes.data_as(POINTER(c_double))
        func_params.rho_0 = self.rho_0.ctypes.data_as(POINTER(c_complex))

        t0_R = params.t0_R * params.timeAMP
        t0_EE = params.t0_EE * params.timeAMP
        width_R = params.timeAMP / params.width_R
        width_EE = params.timeAMP / params.width_EE

        func_params.A_R = params.A_R
        func_params.width_R = width_R
        func_params.t0_R = t0_R

        func_params.A_EE = params.A_EE
        func_params.width_EE = width_EE
        func_params.t0_EE = t0_EE

        func_params.w_R = params.w_R
        func_params.w_v = params.w_v
        func_params.w_EE = params.w_EE

        func_params.nDIM = len(self.energies_A)
        func_params.timeDIM = len(self.time)

        func_params.field_out = self.field_t.ctypes.data_as(POINTER(c_complex))
        func_params.field_grad_A_R = self.field_grad_A_R.ctypes.data_as(POINTER(c_complex))
        func_params.field_grad_A_EE = self.field_grad_A_EE.ctypes.data_as(POINTER(c_complex))

        func_params.lower_bounds = params.lower_bounds.ctypes.data_as(POINTER(c_double))
        func_params.lower_bounds = params.lower_bounds.ctypes.data_as(POINTER(c_double))
        func_params.guess = guess.ctypes.data_as(POINTER(c_double))

        func_params.MAX_EVAL = params.MAX_EVAL

    def call_raman_control_func(self, params):
        self.field_t = np.empty(params.timeDIM, dtype=np.complex)
        molA = Molecule()
        molB = Molecule()
        self.create_molecules(molA, molB)
        func_params = Parameters()
        self.create_parameters(func_params, params)
        RamanControlFunction(molA, molB, func_params)
        return


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    # from itertools import *

    np.set_printoptions(precision=4)
    energy_factor = 1. / 27.211385
    time_factor = .02418 / 1000

    energies_A = np.array((0.000, 0.16304, 0.20209, 1.87855)) * energy_factor
    energies_B = np.array((0.000, 0.15907, 0.19924, 1.77120)) * energy_factor
    N = len(energies_A)
    rho_0 = np.zeros((N, N), dtype=np.complex)
    rho_0[0, 0] = 1. + 0j

    mu = 4.97738 * np.ones_like(rho_0)
    np.fill_diagonal(mu, 0j)
    # mu[2, 1] = 0j
    # mu[1, 2] = 0j
    population_decay = 2.418884e-8
    electronic_dephasing = 2.418884e-4
    vibrational_dephasing = 0.5 * 2.418884e-5
    gamma_decay = np.ones((N, N)) * population_decay
    np.fill_diagonal(gamma_decay, 0.0)
    gamma_decay = np.tril(gamma_decay)
    gamma_decay[2, 1] = 0
    gamma_decay[1, 2] = 0
    gamma_pure_dephasing = np.ones_like(gamma_decay) * electronic_dephasing
    np.fill_diagonal(gamma_pure_dephasing, 0.0)
    gamma_pure_dephasing[1, 0] = vibrational_dephasing
    gamma_pure_dephasing[2, 0] = vibrational_dephasing
    gamma_pure_dephasing[0, 1] = vibrational_dephasing
    gamma_pure_dephasing[0, 2] = vibrational_dephasing
    gamma_pure_dephasing[1, 2] = vibrational_dephasing
    gamma_pure_dephasing[2, 1] = vibrational_dephasing
    # gamma_pure_dephasing[1, 2] = 0.
    # gamma_pure_dephasing[2, 1] = 0.

    print(gamma_decay)
    print(gamma_pure_dephasing)

    timeAMP = 60000
    timeDIM = 120000

    lower_bounds = np.asarray([0.00020, 3.0, 0.9*energies_A[1], 0.35*energy_factor])
    upper_bounds = np.asarray([0.00070, 10.0, 1.1*energies_A[1], 0.45*energy_factor])

    print(lower_bounds)
    # guess = np.asarray([np.random.uniform(lower_bounds[i], upper_bounds[i]) for i in range(len(lower_bounds))])
    guess = np.asarray([0.000543963, 6.02721, energies_A[1], 0.5*energy_factor])

    print(guess)
    params = ADict(
        energy_factor=energy_factor,
        time_factor=time_factor,
        timeDIM=timeDIM,
        timeAMP=timeAMP,

        A_R=0.000576595,
        width_R=5.017,
        t0_R=0.0,

        A_EE=0.000366972,
        width_EE=19.32,
        t0_EE=0.55,

        w_R=0.6 * energy_factor,
        w_v=energies_A[1],
        w_EE=(energies_A[3] - energies_A[1]),
        rho_0=rho_0,

        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        guess=guess,

        MAX_EVAL=100
    )

    FourLevels = dict(
        energies_A=energies_A,
        energies_B=energies_B,
        gamma_decay=gamma_decay,
        gamma_pure_dephasing=gamma_pure_dephasing,
        mu=mu,
    )

    def render_ticks(axes):
        axes.get_xaxis().set_tick_params(which='both', direction='in', width=1, labelrotation=0, labelsize='large')
        axes.get_yaxis().set_tick_params(which='both', direction='in', width=1, labelcolor='r', labelsize='large')
        axes.get_xaxis().set_ticks_position('both')
        axes.get_yaxis().set_ticks_position('both')
        axes.grid()

    molecules = RamanControl(params, **FourLevels)

    molecules.call_raman_control_func(params)

    fig, axes = plt.subplots(nrows=1, ncols=1)
    axes.plot(molecules.time*time_factor, molecules.field_t.real, 'r')

    fig1, axes = plt.subplots(nrows=2, ncols=1, sharex=True)

    axes[0].plot(molecules.time*time_factor, molecules.dyn_rhoA[0, :], label='11_A', linewidth=2.)
    axes[0].plot(molecules.time*time_factor, molecules.dyn_rhoA[1, :], label='22_A', linewidth=2.)
    axes[0].plot(molecules.time*time_factor, molecules.dyn_rhoA[2, :], label='33_A', linewidth=2.)
    axes[0].plot(molecules.time*time_factor, molecules.dyn_rhoA[3, :], label='44_A', linewidth=2.)
    axes[0].plot(molecules.time*time_factor, molecules.dyn_rhoA[4, :], 'k', label='Tr[$\\rho_A^2$]', linewidth=2.)

    axes[0].legend(loc=2)
    render_ticks(axes[0])

    axes[1].plot(molecules.time*time_factor, molecules.dyn_rhoB[0, :], label='11_B', linewidth=2.)
    axes[1].plot(molecules.time*time_factor, molecules.dyn_rhoB[1, :], label='22_B', linewidth=2.)
    axes[1].plot(molecules.time*time_factor, molecules.dyn_rhoB[2, :], label='33_B', linewidth=2.)
    axes[1].plot(molecules.time*time_factor, molecules.dyn_rhoB[3, :], label='44_B', linewidth=2.)
    axes[1].plot(molecules.time*time_factor, molecules.dyn_rhoA[4, :], 'k', label='Tr[$\\rho_A^2$]', linewidth=2.)

    axes[1].legend(loc=2)
    render_ticks(axes[1])

    print(np.diag(molecules.rhoA.real)[2:].sum() - np.diag(molecules.rhoB.real)[2:].sum())

    print(molecules.rhoA.real, "\n")
    print(molecules.rhoB.real, "\n")

    plt.show()

    # gamma = gamma_pure_dephasing.copy()
    # for i in range(4):
    #     for j in range(4):
    #         for k in range(4):
    #             if k > i:
    #                 gamma[i, j] += gamma_decay[k, i]
    #             if k > j:
    #                 gamma[i, j] += gamma_decay[k, j]
    #         gamma[i, j] *= 0.5
    # # gamma += gamma_decay
    #
    # print(gamma)
    #
    # N = 1000
    # omega = np.linspace(1.25, 3., N) * energy_factor
    # spectraA = np.zeros(N)
    # spectraB = np.zeros(N)
    #
    # for i in range(N):
    #     spectraA[i] *= 0.
    #     spectraB[i] *= 0.
    #     for j in range(1, 4):
    #         spectraA[i] += molecules.energies_A[j] * np.abs(molecules.mu[j, 0])**2 * gamma[j, 0] / ((molecules.energies_A[j] - omega[i])**2 + gamma[j, 0]**2)
    #         spectraB[i] += molecules.energies_B[j] * np.abs(molecules.mu[j, 0])**2 * gamma[j, 0] / ((molecules.energies_B[j] - omega[i])**2 + gamma[j, 0]**2)
    #
    # plt.plot(1239.84193/omega*energy_factor, spectraA / omega / (spectraA / omega).max(), 'r')
    # plt.plot(1239.84193/omega*energy_factor, spectraB / omega / (spectraA / omega).max(), 'k')
    # plt.plot(1239.84193 / omega * energy_factor, spectraA / spectraA.max(), 'r--')
    # plt.plot(1239.84193 / omega * energy_factor, spectraB / spectraA.max(), 'k--')
    # plt.show()

