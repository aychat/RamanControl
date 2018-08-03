import numpy as np
from types import MethodType, FunctionType
from RamanControl_wrapper import *
from numpy import ctypeslib


class ADict(dict):
    """
    Dictionary where you can access keys as attributes
    """
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            dict.__getattribute__(self, item)


class RhoPropagate:
    """
    Class for propagating the Lindblad Master equation. We calculate
    rho(T) and obtain NL-polarization by finding Trace[mu * rho(T)]
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

        self.time = np.linspace(-params.timeAMP, params.timeAMP, params.timeDIM)[:, np.newaxis]
        self.field_t = np.empty(params.timeDIM, dtype=np.complex)

        self.gamma_decay = np.ascontiguousarray(self.gamma_decay)
        self.gamma_pure_dephasing = np.ascontiguousarray(self.gamma_pure_dephasing)
        self.mu = np.ascontiguousarray(self.mu)
        self.rho_0 = np.ascontiguousarray(self.rho_0)
        self.energies = np.ascontiguousarray(self.energies)

        self.rho = np.ascontiguousarray(np.zeros_like(self.rho_0, dtype=np.complex))
        self.dyn_rho = np.ascontiguousarray(np.zeros((4, params.timeDIM), dtype=np.complex))
        self.dyn_coh = np.ascontiguousarray(np.zeros((6, params.timeDIM), dtype=np.complex))

    def propagate(self, A_R, width_R, A_EE, width_EE, w_R, w_v, w_EE):
        """
        PROPAGATOR TO CALCULATE DENSITY MATRIX DYNAMICS FOR INTERACTION OF 4-LEVEL SYSTEM WITH FIELD
        :param A_R: Amplitude of field leading to vibrational excitation of the ground state
        :param width_R: Width (gaussian envelope) of above field
        :param A_EE: Amplitude of field leading to electronic excitation from first vibrational state
        :param width_EE: Width (gaussian envelope) of above field
        :param w_R: Raman frequency for Raman excitation
        :param w_v: Vibrational frequency for vibrational excitation
        :param w_EE: Frequency for electronic excitation
        :return: rho(T) after interaction with E(t)
        """
        t0_R = - 0.35 * params.timeAMP
        t0_EE = 0.55 * params.timeAMP
        width_R = params.timeAMP / width_R
        width_EE = params.timeAMP / width_EE

        func_params = parameters()
        func_params.gamma_decay = self.gamma_decay.ctypes.data_as(POINTER(c_double))
        func_params.gamma_pure_dephasing = self.gamma_pure_dephasing.ctypes.data_as(POINTER(c_double))
        func_params.mu = self.mu.ctypes.data_as(POINTER(c_complex))
        func_params.rho_0 = self.rho_0.ctypes.data_as(POINTER(c_complex))
        func_params.energies = self.energies.ctypes.data_as(POINTER(c_double))
        func_params.time = self.time.ctypes.data_as(POINTER(c_double))
        func_params.A_R = A_R
        func_params.width_R = width_R
        func_params.t0_R = t0_R
        func_params.A_EE = A_EE
        func_params.width_EE = width_EE
        func_params.t0_EE = t0_EE
        func_params.w_R = w_R
        func_params.w_v = w_v
        func_params.w_EE = w_EE
        func_params.nDIM = len(self.energies)
        func_params.timeDIM = len(self.time)

        Propagate(self.rho, self.dyn_rho, self.dyn_coh, self.field_t, func_params)

        import matplotlib.pyplot as plt
        plt.plot(self.time, self.field_t.real, 'r')
        plt.show()
        return self.rho


if __name__ == '__main__':

    from RamanControlParameters import plot_Raman_Assisted_Control
    import nlopt
    import time

    np.set_printoptions(precision=4)
    energy_factor = 1. / 27.211385
    time_factor = .02418

    params = ADict(
        energy_factor=energy_factor,
        time_factor=time_factor,
        timeDIM=100000,
        timeAMP=20000.
    )

    energies = np.array((0.000, 0.07439, 1.94655, 2.02094)) * params.energy_factor
    rho_0 = np.zeros((len(energies), len(energies)), dtype=np.complex)
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

    FourLevel = dict(
        energies=energies,
        gamma_decay=gamma_decay,
        gamma_pure_dephasing=gamma_pure_dephasing,
        mu=mu,
        rho_0=rho_0,
    )

    molecule1 = RhoPropagate(params, **FourLevel)
    molecule1.propagate(5e-4, 5., 1e-3, 35., 1.4 * params.energy_factor,
                        0.07439 * params.energy_factor, (1.94655 - 0.07439) * params.energy_factor)

    print molecule1.rho.real
