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
            1e-3 * np.exp(-(self.time + 0.35 * self.timeAMP) ** 2 / (2. * (self.timeAMP / 8.) ** 2))
            * (np.cos(self.omega_Raman * self.time) + np.cos((self.omega_Raman + omega_vib) * self.time))
            + 1e-3 * np.exp(-(self.time - 0.55 * self.timeAMP) ** 2 / (2. * (self.timeAMP / 10.) ** 2))
            * (np.cos(freq * self.time)) + 0j)

        if freq == 0.0:
            self.field_t *= 0.0

        Propagate(
            self.rho, self.dyn_rho, self.dyn_coh, self.field_t, self.gamma_decay, self.gamma_pure_dephasing,
            self.mu, self.rho_0, self.energies, self.timeDIM, self.dt, self.pol2
        )
        return self.rho


if __name__ == '__main__':

    energy_factor = 1. / 27.211385
    energies = np.array((0.000, 0.07439, 1.94655, 2.02094)) * energy_factor
    rho_0 = np.zeros((len(energies), len(energies)), dtype=np.complex)
    rho_0[0, 0] = 1. + 0j
    mu = 4.97738*np.ones_like(rho_0)
    np.fill_diagonal(mu, 0j)

    gamma_decay = np.ones((4, 4))*2.418884e-8
    gamma_decay = np.tril(gamma_decay)

    gamma_pure_dephasing = np.ones_like(gamma_decay)*2.418884e-4
    gamma_pure_dephasing = np.tril(gamma_pure_dephasing)
    gamma_pure_dephasing[1, 0] = 1.422872e-5
    gamma_pure_dephasing[3, 2] = 1.422872e-5

    ThreeLevel = dict(
        energies=energies,
        gamma_decay=gamma_decay,
        gamma_pure_dephasing=gamma_pure_dephasing,
        mu=mu,
        rho_0=rho_0,
        timeDIM=100000,
        timeAMP=10000.,
        omega_Raman=1. * energy_factor

    )

    molecule = RhoPropagate(**ThreeLevel)
    molecule.propagate(energies[1], energies[2] - energies[1])

    np.set_printoptions(precision=4)

    print molecule.rho.real

    print "Ground state population ", np.diag(molecule.rho.real)[:2].sum()
    print "Excited state population ", np.diag(molecule.rho.real)[2:].sum()

    fig, axes = plt.subplots(nrows=3, ncols=1)
    axes[0].plot(molecule.time * .02418, molecule.field_t, 'r')
    axes[0].set_ylabel("Field (a.u.)")
    axes[0].set_xlabel("Time (fs)")
    axes[0].ticklabel_format(axis='y', style='sci', scilimits=(-1, 1))

    axes[1].plot(molecule.time * .02418, molecule.dyn_rho[0].real, label='$\\rho_{11}$')
    axes[1].plot(molecule.time * .02418, molecule.dyn_rho[1].real, label='$\\rho_{22}$')
    axes[1].plot(molecule.time * .02418, molecule.dyn_rho[2].real, label='$\\rho_{33}$')
    axes[1].plot(molecule.time * .02418, molecule.dyn_rho[3].real, label='$\\rho_{44}$')

    axes[1].set_ylabel("Populations")
    axes[1].set_xlabel("Time (fs)")
    axes[1].legend(loc=2)

    axes[2].plot(molecule.time * .02418, molecule.dyn_coh[0].real, label='$\\rho_{12}$')
    axes[2].plot(molecule.time * .02418, molecule.dyn_coh[1].real, label='$\\rho_{13}$')
    axes[2].plot(molecule.time * .02418, molecule.dyn_coh[2].real, label='$\\rho_{14}$')
    axes[2].plot(molecule.time * .02418, molecule.dyn_coh[3].real, label='$\\rho_{23}$')
    axes[2].plot(molecule.time * .02418, molecule.dyn_coh[4].real, label='$\\rho_{24}$')
    axes[2].plot(molecule.time * .02418, molecule.dyn_coh[5].real, label='$\\rho_{34}$')
    axes[2].set_ylabel("Coherences")
    axes[2].set_xlabel("Time (fs)")
    axes[2].ticklabel_format(axis='y', style='sci', scilimits=(-1, 1))
    axes[2].legend(loc=2)

    del molecule
    molecule = RhoPropagate(**ThreeLevel)
    energies = np.array((0.000, 0.08439, 1.94655, 2.02094)) * energy_factor
    molecule.propagate(0.07439 * energy_factor, energies[2] - energies[1])
    print molecule.rho.real

    print "Ground state population ", np.diag(molecule.rho.real)[:2].sum()
    print "Excited state population ", np.diag(molecule.rho.real)[2:].sum()
    axes[1].plot(molecule.time * .02418, molecule.dyn_rho[0].real, '-.')
    axes[1].plot(molecule.time * .02418, molecule.dyn_rho[1].real, '-.')
    axes[1].plot(molecule.time * .02418, molecule.dyn_rho[2].real, '-.')
    axes[1].plot(molecule.time * .02418, molecule.dyn_rho[3].real, '-.')

    # Nfreq = 130
    # rho_excited = np.empty(Nfreq)
    # for w in range(1, Nfreq):
    #     molecule = RhoPropagate(**ThreeLevel)
    #     molecule.propagate(0.05 * 0.15, w * 0.01 * 0.15)
    #     rho_excited[w] = np.diag(molecule.rho.real)[2:].sum()
    #     print w, rho_excited[w]
    #     del molecule
    #
    # fig, axes = plt.subplots(nrows=1, ncols=1)
    # axes.plot(rho_excited[1:], 'k*-')
    plt.show()

