import numpy as np
from types import MethodType, FunctionType
from RamanControl_wrapper import Propagate


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
        self.dt = params.timeAMP * 2. / params.timeDIM
        self.field_t = np.empty(params.timeDIM, dtype=np.complex)
        self.gamma_decay = np.ascontiguousarray(self.gamma_decay)
        self.gamma_pure_dephasing = np.ascontiguousarray(self.gamma_pure_dephasing)
        self.mu = np.ascontiguousarray(self.mu)
        self.rho_0 = np.ascontiguousarray(self.rho_0)
        self.energies = np.ascontiguousarray(self.energies)
        self.rho = np.ascontiguousarray(np.zeros_like(self.rho_0, dtype=np.complex))
        self.dyn_rho = np.ascontiguousarray(np.zeros((4, params.timeDIM), dtype=np.complex))
        self.dyn_coh = np.ascontiguousarray(np.zeros((6, params.timeDIM), dtype=np.complex))

    def propagate(self, A_R, width_R, A_EE, width_EE, omega_vib, freq, params):
        """
        PROPAGATOR TO CALCULATE DENSITY MATRIX DYNAMICS FOR INTERACTION OF 4-LEVEL SYSTEM WITH FIELD
        :param A_R: Amplitude of field leading to vibrational excitation of the ground state
        :param width_R: Width (gaussian envelope) of above field
        :param A_EE: Amplitude of field leading to electronic excitation from first vibrational state
        :param width_EE: Width (gaussian envelope) of above field
        :param omega_vib: Vibrational frequency w_v for vibrational excitation
        :param freq: Frequency for electronic excitation
        :param params: Parameters ADict variable containing constants and field parameters
        :return: rho(T) after interaction with E(t)
        """
        t = self.time
        t0_R = - 0.35 * params.timeAMP
        t0_EE = 0.55 * params.timeAMP
        width_R = params.timeAMP / width_R
        width_EE = params.timeAMP / width_EE
        w_R = params.omega_Raman
        w_v = omega_vib

        self.field_t = np.ascontiguousarray(
            A_R * np.exp(-(t - t0_R) ** 2 / (2. * width_R ** 2))
            * (np.cos(w_R * t) + np.cos((w_R + w_v) * t))
            + A_EE * np.exp(-(t - t0_EE) ** 2 / (2. * width_EE ** 2))
            * (np.cos(freq * t))
            + 0j
        )

        Propagate(
            self.rho, self.dyn_rho, self.dyn_coh, self.field_t, self.gamma_decay, self.gamma_pure_dephasing,
            self.mu, self.rho_0, self.energies, params.timeDIM, self.dt
        )
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
        timeAMP=20000.,
        omega_Raman=1.4 * energy_factor
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
    molecule1.propagate(5e-4, 5., 1e-3, 35., 0.07439 * params.energy_factor, (1.94655 - 0.07439) * params.energy_factor, params)

    molecule1_excited_pop = np.diag(molecule1.rho.real)[2:].sum()
    molecule2 = RhoPropagate(params, **FourLevel)
    molecule2.energies = np.array((0.000, 0.09439, 1.94655, 2.02094)) * params.energy_factor
    molecule2.propagate(5e-4, 5., 1e-3, 35., 0.07439 * params.energy_factor, (1.94655 - 0.07439) * params.energy_factor, params)
    molecule2_excited_pop = np.diag(molecule2.rho.real)[2:].sum()

    print '\n', molecule1.rho.real
    print '\n', molecule2.rho.real
    print '\n', molecule1_excited_pop - molecule2_excited_pop

    plot_Raman_Assisted_Control(molecule1, molecule2, params)
    del molecule1
    del molecule2

    ########################################################################################################
    #                                                                                                      #
    #                               NLOPT optimization of field-parameters                                 #
    #                                                                                                      #
    ########################################################################################################

    start = time.time()

    def rho_cost_function(x, grad):
        """

        :array x: Optimization Variable
        :array grad: Gradient w.r.t. to each x_i; Only used for gradient dependent optimization algorithms
        """
        moleculeA = RhoPropagate(params, **FourLevel)
        moleculeA.energies = np.array((0.000, 0.07439, 1.94655, 2.02094)) * params.energy_factor
        moleculeB = RhoPropagate(params, **FourLevel)
        moleculeB.energies = np.array((0.000, 0.09439, 1.94655, 2.02094)) * params.energy_factor

        moleculeA.propagate(
            x[0], x[1], x[2], x[3],
            0.07439 * params.energy_factor,
            (1.94655 - 0.07439) * params.energy_factor,
            params
        )
        moleculeB.propagate(
            x[0], x[1], x[2], x[3],
            0.07439 * params.energy_factor,
            (1.94655 - 0.07439) * params.energy_factor,
            params
        )

        moleculeA_excited_pop = np.diag(moleculeA.rho.real)[2:].sum()
        moleculeB_excited_pop = np.diag(moleculeB.rho.real)[2:].sum()

        return moleculeA_excited_pop - moleculeB_excited_pop

    opt = nlopt.opt(nlopt.LN_COBYLA, 4)             # LN stands for Local No-derivative
    opt.set_lower_bounds([1e-5, 5., 1e-5, 30.])
    opt.set_upper_bounds([1e-3, 20., 1e-3, 75.])
    opt.set_max_objective(rho_cost_function)
    opt.set_xtol_rel(1e-6)
    x = opt.optimize([1e-3, 10., 1e-3, 32.5])
    maxf = opt.last_optimum_value()
    print("optimum at ", x[0], x[1], x[2], x[3])
    print("maximum value = ", maxf)
    print("result code = ", opt.last_optimize_result())

    print time.time() - start
