import matplotlib.pyplot as plt
import numpy as np


def plot_Raman_Assisted_Control(molecule1, molecule2, params):
    fig, axes = plt.subplots(nrows=4, ncols=1)
    axes[0].plot(molecule1.time * .02418, molecule1.field_t.real, 'r')
    axes[0].set_ylabel("Field (a.u.)")
    axes[0].set_xlabel("Time (fs)")
    axes[0].ticklabel_format(axis='y', style='sci', scilimits=(-1, 1))

    axes[1].plot(molecule1.time * params.time_factor, molecule1.dyn_rho[0].real, label='$\\rho_{11}$')
    axes[1].plot(molecule1.time * params.time_factor, molecule1.dyn_rho[1].real, label='$\\rho_{22}$')
    axes[1].plot(molecule1.time * params.time_factor, molecule1.dyn_rho[2].real, label='$\\rho_{33}$')
    axes[1].plot(molecule1.time * params.time_factor, molecule1.dyn_rho[3].real, label='$\\rho_{44}$')
    axes[1].set_ylabel("Populations")
    axes[1].set_xlabel("Time (fs)")
    axes[1].legend(loc=2)

    axes[3].plot(molecule1.time * params.time_factor, molecule1.dyn_coh[0].real, label='$\\rho_{12}$')
    axes[3].plot(molecule1.time * params.time_factor, molecule1.dyn_coh[1].real, label='$\\rho_{13}$')
    axes[3].plot(molecule1.time * params.time_factor, molecule1.dyn_coh[2].real, label='$\\rho_{14}$')
    axes[3].plot(molecule1.time * params.time_factor, molecule1.dyn_coh[3].real, label='$\\rho_{23}$')
    axes[3].plot(molecule1.time * params.time_factor, molecule1.dyn_coh[4].real, label='$\\rho_{24}$')
    axes[3].plot(molecule1.time * params.time_factor, molecule1.dyn_coh[5].real, label='$\\rho_{34}$')
    axes[3].set_ylabel("Coherences")
    axes[3].set_xlabel("Time (fs)")
    axes[3].ticklabel_format(axis='y', style='sci', scilimits=(-1, 1))
    axes[3].legend(loc=2)

    axes[2].plot(molecule2.time * params.time_factor, molecule2.dyn_rho[0].real, label='$\\rho_{11}$')
    axes[2].plot(molecule2.time * params.time_factor, molecule2.dyn_rho[1].real, label='$\\rho_{22}$')
    axes[2].plot(molecule2.time * params.time_factor, molecule2.dyn_rho[2].real, label='$\\rho_{33}$')
    axes[2].plot(molecule2.time * params.time_factor, molecule2.dyn_rho[3].real, label='$\\rho_{44}$')
    axes[2].set_ylabel("Populations")
    axes[2].set_xlabel("Time (fs)")
    axes[2].legend(loc=2)

    plt.show()