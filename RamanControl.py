import numpy as np
import matplotlib.pyplot as plt
from types import MethodType, FunctionType
from RamanControl_wrapper import Propagate


class RhoPropagate:
    """
    Class for propagating the Lindblad Master equation. We calculate 
    rho(T) and obtain NL-polarization by finding Trace[mu * rho(T)]
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

        self.time = np.linspace(-self.timeAMP, self.timeAMP, self.timeDIM)[:, np.newaxis]
        self.dt = self.timeAMP * 2. / self.timeDIM
        self.field_t = np.empty(self.timeDIM, dtype=np.complex)
        self.gamma_decay = np.ascontiguousarray(self.gamma_decay)
        self.gamma_pure_dephasing = np.ascontiguousarray(self.gamma_pure_dephasing)
        self.mu = np.ascontiguousarray(self.mu)
        self.rho_0 = np.ascontiguousarray(self.rho_0)
        self.energies = np.ascontiguousarray(self.energies)
        self.H0 = np.diag(self.energies)
        self.rho = np.ascontiguousarray(np.zeros_like(self.rho_0, dtype=np.complex))
        self.dyn_rho = np.ascontiguousarray(np.zeros((4, self.timeDIM), dtype=np.complex))
        self.dyn_coh = np.ascontiguousarray(np.zeros((6, self.timeDIM), dtype=np.complex))
        self.pol2 = np.ascontiguousarray(np.zeros_like(self.timeDIM, dtype=np.complex))

    def propagate(self, omega_vib, freq):

        self.field_t = np.ascontiguousarray(
            1e-3 * np.exp(-(self.time + 0.35 * self.timeAMP) ** 2 / (2. * (self.timeAMP / 7.5) ** 2))
            * (np.cos(self.omega_Raman * self.time) + np.cos((self.omega_Raman + omega_vib) * self.time))
            # + 5e-3 * np.exp(-(self.time - 0.55 * self.timeAMP) ** 2 / (2. * (self.timeAMP / 7.5) ** 2))
            # * (np.cos(freq * self.time))
            + 0j)

        if freq == 0.0:
            self.field_t *= 0.0

        Propagate(
            self.rho, self.dyn_rho, self.dyn_coh, self.field_t, self.gamma_decay, self.gamma_pure_dephasing,
            self.mu, self.rho_0, self.energies, self.timeDIM, self.dt, self.pol2
        )
        return self.rho


if __name__ == '__main__':

    np.set_printoptions(precision=4)

    energy_factor = 1. / 27.211385
    time_factor = .02418

    energies = np.array((0.000, 0.07439, 1.94655, 2.02094)) * energy_factor
    rho_0 = np.zeros((len(energies), len(energies)), dtype=np.complex)
    rho_0[0, 0] = 1. + 0j
    mu = 4.97738*np.ones_like(rho_0)
    np.fill_diagonal(mu, 0j)

    gamma_decay = np.ones((4, 4))*2.418884e-8
    gamma_decay = np.tril(gamma_decay)

    gamma_pure_dephasing = np.ones_like(gamma_decay)*2.418884e-4
    np.fill_diagonal(gamma_pure_dephasing, 0.0)
    gamma_pure_dephasing[1, 0] = 1.422872e-5
    gamma_pure_dephasing[3, 2] = 1.422872e-5
    gamma_pure_dephasing[0, 1] = 1.422872e-5
    gamma_pure_dephasing[2, 3] = 1.422872e-5
    # gamma_pure_dephasing = np.tril(gamma_pure_dephasing)

    print gamma_pure_dephasing
    ThreeLevel = dict(
        energies=energies,
        gamma_decay=gamma_decay,
        gamma_pure_dephasing=gamma_pure_dephasing,
        mu=mu,
        rho_0=rho_0,
        timeDIM=100000,
        timeAMP=20000.,
        omega_Raman=0.8 * energy_factor

    )

    molecule1 = RhoPropagate(**ThreeLevel)
    molecule1.propagate(energies[1], energies[2] - energies[1])

    print "Ground state population ", np.diag(molecule1.rho.real)[:2].sum()
    print "Excited state population ", np.diag(molecule1.rho.real)[2:].sum()
    print

    molecule1_exc_pop = np.diag(molecule1.rho.real)[2:].sum()

    fig, axes = plt.subplots(nrows=4, ncols=1)
    axes[0].plot(molecule1.time * .02418, molecule1.field_t.real, 'r')
    axes[0].set_ylabel("Field (a.u.)")
    axes[0].set_xlabel("Time (fs)")
    axes[0].ticklabel_format(axis='y', style='sci', scilimits=(-1, 1))

    axes[1].plot(molecule1.time * time_factor, molecule1.dyn_rho[0].real, label='$\\rho_{11}$')
    axes[1].plot(molecule1.time * time_factor, molecule1.dyn_rho[1].real, label='$\\rho_{22}$')
    axes[1].plot(molecule1.time * time_factor, molecule1.dyn_rho[2].real, label='$\\rho_{33}$')
    axes[1].plot(molecule1.time * time_factor, molecule1.dyn_rho[3].real, label='$\\rho_{44}$')
    axes[1].set_ylabel("Populations")
    axes[1].set_xlabel("Time (fs)")
    axes[1].legend(loc=2)

    axes[3].plot(molecule1.time * time_factor, molecule1.dyn_coh[0].real, label='$\\rho_{12}$')
    axes[3].plot(molecule1.time * time_factor, molecule1.dyn_coh[1].real, label='$\\rho_{13}$')
    axes[3].plot(molecule1.time * time_factor, molecule1.dyn_coh[2].real, label='$\\rho_{14}$')
    axes[3].plot(molecule1.time * time_factor, molecule1.dyn_coh[3].real, label='$\\rho_{23}$')
    axes[3].plot(molecule1.time * time_factor, molecule1.dyn_coh[4].real, label='$\\rho_{24}$')
    axes[3].plot(molecule1.time * time_factor, molecule1.dyn_coh[5].real, label='$\\rho_{34}$')
    axes[3].set_ylabel("Coherences")
    axes[3].set_xlabel("Time (fs)")
    axes[3].ticklabel_format(axis='y', style='sci', scilimits=(-1, 1))
    axes[3].legend(loc=2)

    molecule2 = RhoPropagate(**ThreeLevel)
    molecule2.energies = np.array((0.000, 0.08439, 1.94655, 2.02094)) * energy_factor
    molecule2.propagate(0.07439 * energy_factor, (1.94655 - 0.07439) * energy_factor)

    print "Ground state population ", np.diag(molecule2.rho.real)[:2].sum()
    print "Excited state population ", np.diag(molecule2.rho.real)[2:].sum()

    molecule2_exc_pop = np.diag(molecule2.rho.real)[2:].sum()
    print
    print molecule1_exc_pop - molecule2_exc_pop
    axes[2].plot(molecule2.time * time_factor, molecule2.dyn_rho[0].real, label='$\\rho_{11}$')
    axes[2].plot(molecule2.time * time_factor, molecule2.dyn_rho[1].real, label='$\\rho_{22}$')
    axes[2].plot(molecule2.time * time_factor, molecule2.dyn_rho[2].real, label='$\\rho_{33}$')
    axes[2].plot(molecule2.time * time_factor, molecule2.dyn_rho[3].real, label='$\\rho_{44}$')
    axes[2].set_ylabel("Populations")
    axes[2].set_xlabel("Time (fs)")
    axes[2].legend(loc=2)

    print
    print molecule1.rho.real
    print
    print molecule2.rho.real

    del molecule1
    del molecule2

    # Nfreq = 300
    # rho_excited1 = np.empty(Nfreq)
    # rho_excited2 = np.empty(Nfreq)
    # for w in range(1, Nfreq):
    #     molecule1 = RhoPropagate(**ThreeLevel)
    #     molecule1.energies = np.array((0.000, 0.07439, 1.94655, 2.02094)) * energy_factor
    #     molecule2 = RhoPropagate(**ThreeLevel)
    #     molecule2.energies = np.array((0.000, 0.08439, 1.94655, 2.12094)) * energy_factor
    #
    #     molecule1.propagate(0.07439 * energy_factor, w * 0.005 * (1.94655 - 0.07439) * energy_factor)
    #     molecule2.propagate(0.07439 * energy_factor, w * 0.005 * (1.94655 - 0.07439) * energy_factor)
    #     rho_excited1[w] = np.diag(molecule1.rho.real)[2:].sum()
    #     rho_excited2[w] = np.diag(molecule2.rho.real)[2:].sum()
    #     print w, rho_excited1[w], rho_excited2[w]
    #     del molecule1
    #     del molecule2
    #
    # fig, axes = plt.subplots(nrows=1, ncols=1)
    # axes.plot(rho_excited1[1:], 'k*-', label='molecule 1')
    # axes.plot(rho_excited2[1:], 'r*-', label='molecule 2')

    plt.show()

