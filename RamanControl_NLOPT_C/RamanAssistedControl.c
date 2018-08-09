#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <nlopt.h>

#include "RamanAssistedControl_utils.h"

typedef double complex cmplx;

typedef struct parameters{
    double* time;
    cmplx* rho_0;

    double A_R;
    double width_R;
    double t0_R;

    double A_EE;
    double width_EE;
    double t0_EE;

    double w_R;
    double w_v;
    double w_EE;

    int nDIM;
    int timeDIM;

    cmplx* field_out;
} parameters;

typedef struct molecule{
    double* energies;
    double* gamma_decay;
    double* gamma_pure_dephasing;
    cmplx* mu;

    cmplx* rho;
    cmplx* dyn_rho;
} molecule;

typedef struct mol_system{
    molecule* moleculeA;
    molecule* moleculeB;
    parameters* params;
} mol_system;


//====================================================================================================================//
//                                                                                                                    //
//                                 AUXILIARY FUNCTIONS FOR SIMPLE MATRIX FUNCTIONS                                    //
//                                                                                                                    //
//====================================================================================================================//





//====================================================================================================================//
//                                                                                                                    //
//                                         FUNCTIONS FOR PROPAGATION STEP                                             //
//                                                                                                                    //
//====================================================================================================================//

void CalculateField(cmplx* field, parameters* params)
//----------------------------------------------------//
//   RETURNS THE ENTIRE FIELD AS A FUNCTION OF TIME   //
//----------------------------------------------------//
{
    int i;
    int nDIM = params->nDIM;
    int timeDIM = params->timeDIM;

    double* t = params->time;

    double A_R = params->A_R;
    double width_R = params->width_R;
    double t0_R = params->t0_R;

    double A_EE = params->A_EE;
    double width_EE = params->width_EE;
    double t0_EE = params->t0_EE;

    double w_R = params->w_R;
    double w_v = params->w_v;
    double w_EE = params->w_EE;


    for(i=0; i<timeDIM; i++)
    {
        field[i] = A_R * exp(-pow(t[i] - t0_R, 2) / (2. * pow(width_R, 2))) * (cos((w_R + w_v) * t[i]) + cos(w_R * t[i]))
        + A_EE * exp(-pow(t[i] - t0_EE, 2) / (2. * pow(width_EE, 2))) * cos(w_EE * t[i]);
    }
}

void L_operate(cmplx* Qmat, const cmplx field_ti, molecule* mol, parameters* params)
//----------------------------------------------------//
// 	    RETURNS Q <-- L[Q] AT A PARTICULAR TIME (t)   //
//----------------------------------------------------//
{
    int m, n, k;
    int nDIM = params->nDIM;
    double* gamma_pure_dephasing = mol->gamma_pure_dephasing;
    double* gamma_decay = mol->gamma_decay;
    cmplx* mu = mol->mu;
    double* energies = mol->energies;

    cmplx* Lmat = (cmplx*)calloc(nDIM * nDIM,  sizeof(cmplx));

    for(m = 0; m < nDIM; m++)
        {
        for(n = 0; n < nDIM; n++)
            {
                Lmat[m * nDIM + n] += - I * (energies[m] - energies[n]) * Qmat[m * nDIM + n];
                for(k = 0; k < nDIM; k++)
                {
                    Lmat[m * nDIM + n] -= 0.5 * (gamma_decay[n * nDIM + k] + gamma_decay[m * nDIM + k]) * Qmat[m * nDIM + n];
                    Lmat[m * nDIM + n] += I * field_ti * (mu[m * nDIM + k] * Qmat[k * nDIM + n] - Qmat[m * nDIM + k] * mu[k * nDIM + n]);

                    if (m == n)
                    {
                        Lmat[m * nDIM + m] += gamma_decay[k * nDIM + m] * Qmat[k * nDIM + k];
                    }
                    else
                    {
                        Lmat[m * nDIM + n] -= gamma_pure_dephasing[m * nDIM + n] * Qmat[m * nDIM + n];
                    }


                }

            }

        }

    for(m = 0; m < nDIM; m++)
        {
        for(n = 0; n < nDIM; n++)
            {
                Qmat[m * nDIM + n] = Lmat[m * nDIM + n];
            }
        }
    free(Lmat);

}


void Propagate(molecule* mol, parameters* params)
//----------------------------------------------------------------//
//    GETTING rho(T)_{k=[3,4]} FROM rho(0) USING PROPAGATE FUNCTION     //
//----------------------------------------------------------------//
{
    int i, j, k;
    int nDIM = params->nDIM;
    int timeDIM = params->timeDIM;

    double *gamma_decay = mol->gamma_decay;
    double *gamma_pure_dephasing = mol->gamma_pure_dephasing;
    cmplx *mu = mol->mu;
    cmplx *rho_0 = params->rho_0;
    double *energies = mol->energies;
    double *time = params->time;
    cmplx* field = params->field_out;

    double dt = time[1] - time[0];

    cmplx* L_func = (cmplx*)calloc(nDIM * nDIM, sizeof(cmplx));
    copy_mat(rho_0, L_func, nDIM);
    copy_mat(rho_0, mol->rho, nDIM);

    for(i=0; i<timeDIM; i++)
    {
        j=0;
        do
        {
            L_operate(L_func, field[i], mol, params);
            scale_mat(L_func, dt/(j+1), nDIM);
            add_mat(L_func, mol->rho, nDIM);
            j+=1;
        }while(complex_max_element(L_func, nDIM) > 1.0E-8);

        for(k=0; k<nDIM; k++)
        {
            mol->dyn_rho[k * timeDIM + i] = mol->rho[k * nDIM + k];
        }

        mol->dyn_rho[4 * timeDIM + i] = mol->rho[0 * nDIM + 1];
        mol->dyn_rho[5 * timeDIM + i] = mol->rho[0 * nDIM + 2];
        mol->dyn_rho[6 * timeDIM + i] = mol->rho[0 * nDIM + 3];
        mol->dyn_rho[7 * timeDIM + i] = mol->rho[1 * nDIM + 2];
        mol->dyn_rho[8 * timeDIM + i] = mol->rho[1 * nDIM + 3];
        mol->dyn_rho[9 * timeDIM + i] = mol->rho[2 * nDIM + 3];

        copy_mat(mol->rho, L_func, nDIM);
    }

    free(L_func);
}


double calculateJ(molecule* molA, molecule* molB, int nDIM)
{
    double molA_excited_pop = creal(molA->rho[2*nDIM + 2] + molA->rho[3*nDIM + 3]);
    double molB_excited_pop = creal(molB->rho[2*nDIM + 2] + molB->rho[3*nDIM + 3]);

    return molA_excited_pop - molB_excited_pop;
}


double nloptJ(unsigned N, const double *opt_params, double *grad_J, void *nloptJ_params)
{

    mol_system** Ensemble = (mol_system**)nloptJ_params;

    parameters* params = (*Ensemble)->params;
    molecule* moleculeA = (*Ensemble)->moleculeA;
    molecule* moleculeB = (*Ensemble)->moleculeB;

    params->A_R = opt_params[0];
    params->A_EE = opt_params[1];

    CalculateField(params->field_out, params);
    Propagate(moleculeA, params);
    Propagate(moleculeB, params);

    int nDIM = params->nDIM;

    printf("%12.12lf %12.12lf %12.12lf \n", params->A_R, params->A_EE, calculateJ(moleculeA, moleculeB, nDIM));
    return calculateJ(moleculeA, moleculeB, nDIM);

    free(params);
    free(moleculeA);
    free(moleculeB);
    free(Ensemble);

}


cmplx* RamanControlFunction(molecule* molA, molecule* molB, parameters* func_params)
//------------------------------------------------------------//
//    GETTING rho(T) FROM rho(0) USING PROPAGATE FUNCTION     //
//------------------------------------------------------------//
{
    mol_system* Ensemble;
    Ensemble->moleculeA = molA;
    Ensemble->moleculeB = molB;
    Ensemble->params = func_params;

    nlopt_opt opt;

    double lower_bounds[2] = { 0.00005, 0.00005 };
    double upper_bounds[2] = { 0.00100, 0.00100 };

    opt = nlopt_create(NLOPT_LN_COBYLA, 2);
    nlopt_set_lower_bounds(opt, lower_bounds);
    nlopt_set_upper_bounds(opt, upper_bounds);
    nlopt_set_max_objective(opt, nloptJ, (void*)&Ensemble);
    nlopt_set_xtol_rel(opt, 1e-12);

    double x[2] = { 0.0005, 0.0005 };
    double maxf;

    if (nlopt_optimize(opt, x, &maxf) < 0) {
        printf("nlopt failed!\n");
    }
    else {
        printf("found minimum at f(%g,%g) = %0.10g\n", x[0], x[1], maxf);
    }

    nlopt_destroy(opt);
}
