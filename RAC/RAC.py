import numpy as np
from types import MethodType, FunctionType
from RAC_wrapper import *


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

    def __init__(self, **kwargs):
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

    def call_raman_control_function(self, python_params):

        RamanControlFunction(python_params)
        return


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import time
    np.set_printoptions(precision=4)
    energy_factor = 1. / 27.211385
    time_factor = .02418

    energies_A = np.array((0.000, 0.07439, 1.94655, 2.02094)) * energy_factor
    energies_B = np.array((0.000, 0.09439, 1.94655, 2.02094)) * energy_factor
    rho_0 = np.zeros((len(energies_A), len(energies_A)), dtype=np.complex)
    rho_0[0, 0] = 1. + 0j
    mu = 4.97738*np.ones_like(rho_0, dtype=np.complex)
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

    timeDIM = 100000
    field_t = np.empty(timeDIM, dtype=np.complex)
    field_grad_A_R = np.empty(timeDIM, dtype=np.complex)
    field_grad_A_EE = np.empty(timeDIM, dtype=np.complex)

    params = ADict(
        energy_factor=energy_factor,
        time_factor=time_factor,
        timeDIM=timeDIM,
        timeAMP=20000.,

        A_R=5e-4,
        width_R=5.,
        t0_R=-0.35,

        A_EE=1e-3,
        width_EE=35.,
        t0_EE=0.55,

        w_R=1.4 * energy_factor,
        w_v=energies_A[1],
        w_EE=(energies_A[2] - energies_A[1]),
        rho_0=rho_0,

        energies_A=energies_A,
        energies_B=energies_B,
        gamma_decay=gamma_decay,
        gamma_pure_dephasing=gamma_pure_dephasing,
        mu=mu,
        field_out=field_t,
        field_grad_A_R=field_grad_A_R,
        field_grad_A_EE=field_grad_A_EE
    )

    molecules = RamanControl(**params)
    molecules.call_raman_control_function(params)
