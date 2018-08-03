#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

typedef double complex cmplx;

typedef struct parameters{
    double *gamma_decay;
    double *gamma_pure_dephasing;
    cmplx *mu;
    cmplx *rho_0;
    double *energies;

    double *time;

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
} parameters;

//====================================================================================================================//
//                                                                                                                    //
//                                 AUXILIARY FUNCTIONS FOR SIMPLE MATRIX FUNCTIONS                                    //
//                                                                                                                    //
//====================================================================================================================//


void print_complex_mat(cmplx *A, int nDIM)
//----------------------------------------------------//
// 	          PRINTS A COMPLEX MATRIX                 //
//----------------------------------------------------//
{
	int i,j;
	for(i=0; i<nDIM; i++)
	{
		for(j=0; j<nDIM; j++)
		{
			printf("%3.3e + %3.3eJ  ", creal(A[i * nDIM + j]), cimag(A[i * nDIM + j]));
		}
	    printf("\n");
	}
	printf("\n\n");
}

void print_double_mat(double *A, int nDIM)
//----------------------------------------------------//
// 	            PRINTS A REAL MATRIX                  //
//----------------------------------------------------//
{
	int i,j;
	for(i=0; i<nDIM; i++)
	{
		for(j=0; j<nDIM; j++)
		{
			printf("%3.3e  ", A[i * nDIM + j]);
		}
	    printf("\n");
	}
	printf("\n\n");
}

void copy_mat(const cmplx *A, cmplx *B, int nDIM)
//----------------------------------------------------//
// 	        COPIES MATRIX A ----> MATRIX B            //
//----------------------------------------------------//
{
    int i, j = 0;
    for(i=0; i<nDIM; i++)
    {
        for(j=0; j<nDIM; j++)
        {
            B[i * nDIM + j] = A[i * nDIM + j];
        }
    }
}

void add_mat(const cmplx *A, cmplx *B, int nDIM)
//----------------------------------------------------//
// 	        ADDS A to B ----> MATRIX B = A + B        //
//----------------------------------------------------//
{
    int i, j = 0;
    for(i=0; i<nDIM; i++)
    {
        for(j=0; j<nDIM; j++)
        {
            B[i * nDIM + j] += A[i * nDIM + j];
        }
    }
}

void scale_mat(cmplx *A, double factor, int nDIM)
//----------------------------------------------------//
// 	        ADDS A to B ----> MATRIX B = A + B        //
//----------------------------------------------------//
{
    for(int i=0; i<nDIM; i++)
    {
        for(int j=0; j<nDIM; j++)
        {
            A[i * nDIM + j] *= factor;
        }
    }
}

double complex_abs(cmplx z)
//----------------------------------------------------//
// 	            RETURNS ABSOLUTE VALUE OF Z           //
//----------------------------------------------------//
{

    return sqrt((creal(z)*creal(z) + cimag(z)*cimag(z)));
}

cmplx complex_trace(cmplx *A, int nDIM)
//----------------------------------------------------//
// 	                 RETURNS TRACE[A]                 //
//----------------------------------------------------//
{
    cmplx trace = 0.0 + I * 0.0;
    for(int i=0; i<nDIM; i++)
    {
        trace += A[i*nDIM + i];
    }
    printf("Trace = %3.3e + %3.3eJ  \n", creal(trace), cimag(trace));

    return trace;
}

void multiply_complex_mat(cmplx *product, const cmplx *A, const cmplx *B, const int nDIM)
//----------------------------------------------------//
// 	             RETURNS A*B MATRIX PRODUCT           //
//----------------------------------------------------//
{
    for (int i=0; i<nDIM; i++)
    {
        for (int j=0; j<nDIM; j++)
        {
            for (int k=0; k<nDIM; k++)
            {
                product[i*nDIM + j] += A[i*nDIM + k]*B[k*nDIM + j];
            }
        }
    }
}


double complex_max_element(cmplx *A, int nDIM)
//----------------------------------------------------//
// 	   RETURNS ELEMENT WITH MAX ABSOLUTE VALUE        //
//----------------------------------------------------//
{
    double max_el = 0.0;
    int i, j = 0;
    for(i=0; i<nDIM; i++)
    {
        for(j=0; j<nDIM; j++)
        {
            if(complex_abs(A[i * nDIM + j]) > max_el)
            {
                max_el = complex_abs(A[i * nDIM + j]);
            }
        }
    }
    return max_el;
}


void L_operate(cmplx* Qmat, const cmplx field_ti, const double* gamma_decay, const double* gamma_pure_dephasing,
                const cmplx* mu, const double* energies, int nDIM)
//----------------------------------------------------//
// 	    RETURNS Q <-- L[Q] AT A PARTICULAR TIME (t)   //
//----------------------------------------------------//
{
    int m, n, k;
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

void CalculateField(cmplx* field, struct parameters* func_params)
{
    int i;
    int nDIM = func_params->nDIM;
    int timeDIM = func_params->timeDIM;

    double* t = func_params->time;

    double A_R = func_params->A_R;
    double width_R = func_params->width_R;
    double t0_R = func_params->t0_R;

    double A_EE = func_params->A_EE;
    double width_EE = func_params->width_EE;
    double t0_EE = func_params->t0_EE;

    double w_R = func_params->w_R;
    double w_v = func_params->w_v;
    double w_EE = func_params->w_EE;

    for(i=0; i<timeDIM; i++)
    {
        field[i] = A_R * exp(-pow(t[i] - t0_R, 2) / (2. * pow(width_R, 2))) * (cos((w_R + w_v) * t[i]) + cos(w_R * t[i]))
        + A_EE * exp(-pow(t[i] - t0_EE, 2) / (2. * pow(width_EE, 2))) * cos(w_EE * t[i]);
    }
}

void Propagate(cmplx* out, cmplx* dyn_rho, cmplx* dyn_coh, cmplx* field, struct parameters* func_params)
//------------------------------------------------------------//
//    GETTING rho(T) FROM rho(0) USING PROPAGATE FUNCTION     //
//------------------------------------------------------------//
{
    int i, j, k;
    int nDIM = func_params->nDIM;
    int timeDIM = func_params->timeDIM;

    CalculateField(field, func_params);

    double *gamma_decay = func_params->gamma_decay;
    double *gamma_pure_dephasing = func_params->gamma_pure_dephasing;
    cmplx *mu = func_params->mu;
    cmplx *rho_0 = func_params->rho_0;
    double *energies = func_params->energies;
    double *time = func_params->energies;

    double dt = time[1] - time[0];

    cmplx* L_func = (cmplx*)calloc(nDIM * nDIM, sizeof(cmplx));
    copy_mat(rho_0, L_func, nDIM);
    copy_mat(rho_0, out, nDIM);

    printf("%d \n", timeDIM);

    for(i=0; i<timeDIM; i++)
    {
        printf("%5.5lf \n", creal(field[i]));

        j=0;
        do
        {
            L_operate(L_func, field[i], gamma_decay, gamma_pure_dephasing, mu, energies, nDIM);
            scale_mat(L_func, dt/(j+1), nDIM);
            add_mat(L_func, out, nDIM);
            j+=1;
        }while(complex_max_element(L_func, nDIM) > 1.0E-8);

        for(k=0; k<nDIM; k++)
        {
            dyn_rho[k*timeDIM + i] = out[k*nDIM + k];
        }

        dyn_coh[0*timeDIM + i] = out[0*nDIM + 1];
        dyn_coh[1*timeDIM + i] = out[0*nDIM + 2];
        dyn_coh[2*timeDIM + i] = out[0*nDIM + 3];
        dyn_coh[3*timeDIM + i] = out[1*nDIM + 2];
        dyn_coh[4*timeDIM + i] = out[1*nDIM + 3];
        dyn_coh[5*timeDIM + i] = out[2*nDIM + 3];

        copy_mat(out, L_func, nDIM);
    }

    free(L_func);
}