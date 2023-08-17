#include <stdlib.h>
#include "../../shared.h"
#include "Kokkos_Core.hpp"
#include "shared.hpp"

/*
 *		CONJUGATE GRADIENT SOLVER KERNEL
 */

// Initialises the CG solver
void cg_init(
        const int x,
        const int y,
        const int halo_depth,
        const int coefficient,
        double rx,
        double ry,
        double* rro,
        KView density,
        KView energy,
        KView u,
        KView p,
        KView r,
        KView w,
        KView kx,
        KView ky)
{
    if(coefficient != CONDUCTIVITY && coefficient != RECIP_CONDUCTIVITY)
    {
        die(__LINE__, __FILE__, "Coefficient %d is not valid.\n", coefficient);
    }




		Kokkos::parallel_for(x*y, KOKKOS_LAMBDA (const int idx) {
      const size_t kk = idx % x;
      const size_t jj = idx / x;

      if (0-1 < jj && jj < y && 0-1 < kk && kk < x) {
            const int index = kk + jj*x;
            p[index] = 0.0;
            r[index] = 0.0;
            u[index] = energy[index]*density[index];
        }
		});






		Kokkos::parallel_for(x*y, KOKKOS_LAMBDA (const int idx) {
      const size_t kk = idx % x;
      const size_t jj = idx / x;

      if (1-1 < jj && jj < y-1 && 1-1 < kk && kk < x-1) {
            const int index = kk + jj*x;
            w[index] = (coefficient == CONDUCTIVITY)
                ? density[index] : 1.0/density[index];
        }
		});






		Kokkos::parallel_for(x*y, KOKKOS_LAMBDA (const int idx) {
      const size_t kk = idx % x;
      const size_t jj = idx / x;

      if (halo_depth-1 < jj && jj < y-1 && halo_depth-1 < kk && kk < x-1) {
            const int index = kk + jj*x;
            kx[index] = rx*(w[index-1]+w[index]) /
                (2.0*w[index-1]*w[index]);
            ky[index] = ry*(w[index-x]+w[index]) /
                (2.0*w[index-x]*w[index]);
        }
		});



    double rro_temp = 0.0;




		Kokkos::parallel_reduce(x*y, KOKKOS_LAMBDA (const int idx, double& intermed) {
			const size_t kk = idx % x;
      const size_t jj = idx / x;

      if (halo_depth-1 < jj && jj < y-halo_depth && halo_depth-1 < kk && kk < x-halo_depth) {
            const int index = kk + jj*x;
            const double smvp = SMVP(u);
            w[index] = smvp;
            r[index] = u[index]-w[index];
            p[index] = r[index];
            intermed += r[index]*p[index];
        }
		},
		rro_temp);



    // Sum locally
    *rro += rro_temp;
}

// Calculates w
void cg_calc_w(
        const int x,
        const int y,
        const int halo_depth,
        double* pw,
        KView p,
        KView w,
        KView kx,
        KView ky)
{
    double pw_temp = 0.0;




		Kokkos::parallel_reduce(x*y, KOKKOS_LAMBDA (const int idx, double& intermed) {
			const size_t kk = idx % x;
      const size_t jj = idx / x;

      if (halo_depth-1 < jj && jj < y-halo_depth && halo_depth-1 < kk && kk < x-halo_depth) {
            const int index = kk + jj*x;
            const double smvp = SMVP(p);
            w[index] = smvp;
            intermed += w[index]*p[index];
        }
		},
		pw_temp);



    *pw += pw_temp;
}

// Calculates u and r
void cg_calc_ur(
        const int x,
        const int y,
        const int halo_depth,
        const double alpha,
        double* rrn,
        KView u,
        KView p,
        KView r,
        KView w)
{
    double rrn_temp = 0.0;




		Kokkos::parallel_reduce(x*y, KOKKOS_LAMBDA (const int idx, double& intermed) {
			const size_t kk = idx % x;
      const size_t jj = idx / x;

      if (halo_depth-1 < jj && jj < y-halo_depth && halo_depth-1 < kk && kk < x-halo_depth) {
            const int index = kk + jj*x;

            u[index] += alpha*p[index];
            r[index] -= alpha*w[index];
            intermed += r[index]*r[index];
        }
		},
		rrn_temp);



    *rrn += rrn_temp;
}

// Calculates p
void cg_calc_p(
        const int x,
        const int y,
        const int halo_depth,
        const double beta,
        KView p,
        KView r)
{



		Kokkos::parallel_for(x*y, KOKKOS_LAMBDA (const int idx) {
      const size_t kk = idx % x;
      const size_t jj = idx / x;

      if (halo_depth-1 < jj && jj < y-halo_depth && halo_depth-1 < kk && kk < x-halo_depth) {
            const int index = kk + jj*x;

            p[index] = beta*p[index] + r[index];
        }
		});


}
