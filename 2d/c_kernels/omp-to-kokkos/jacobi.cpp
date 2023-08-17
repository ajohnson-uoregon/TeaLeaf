#include <stdlib.h>
#include <math.h>
#include "../../shared.h"
#include "Kokkos_Core.hpp"
#include "shared.hpp"

/*
 *		JACOBI SOLVER KERNEL
 */

// Initialises the Jacobi solver
void jacobi_init(
        const int x,
        const int y,
        const int halo_depth,
        const int coefficient,
        double rx,
        double ry,
        KView density,
        KView energy,
        KView u0,
        KView u,
        KView kx,
        KView ky)
{
    if(coefficient < CONDUCTIVITY && coefficient < RECIP_CONDUCTIVITY)
    {
        die(__LINE__, __FILE__, "Coefficient %d is not valid.\n", coefficient);
    }




		Kokkos::parallel_for(x*y, KOKKOS_LAMBDA (const int idx) {
      const size_t kk = idx % x;
      const size_t jj = idx / x;

      if (1-1 < jj && jj < y-1 && 1-1 < kk && kk < x-1) {
            const int index = kk + jj*x;
            double temp = energy[index]*density[index];
            u0[index] = temp;
            u[index] = temp;
        }
		});






		Kokkos::parallel_for(x*y, KOKKOS_LAMBDA (const int idx) {
      const size_t kk = idx % x;
      const size_t jj = idx / x;

      if (halo_depth-1 < jj && jj < y-1 && halo_depth-1 < kk && kk < x-1) {
            const int index = kk + jj*x;
            double densityCentre = (coefficient == CONDUCTIVITY)
                ? density[index] : 1.0/density[index];
            double densityLeft = (coefficient == CONDUCTIVITY)
                ? density[index-1] : 1.0/density[index-1];
            double densityDown = (coefficient == CONDUCTIVITY)
                ? density[index-x] : 1.0/density[index-x];

            kx[index] = rx*(densityLeft+densityCentre)/(2.0*densityLeft*densityCentre);
            ky[index] = ry*(densityDown+densityCentre)/(2.0*densityDown*densityCentre);
        }
		});


}

// The main Jacobi solve step
void jacobi_iterate(
        const int x,
        const int y,
        const int halo_depth,
        double* error,
        KView kx,
        KView ky,
        KView u0,
        KView u,
        KView r)
{



		Kokkos::parallel_for(x*y, KOKKOS_LAMBDA (const int idx) {
      const size_t kk = idx % x;
      const size_t jj = idx / x;

      if (0-1 < jj && jj < y && 0-1 < kk && kk < x) {
            const int index = kk + jj*x;
            r[index] = u[index];
        }
		});



    double err=0.0;



		Kokkos::parallel_reduce(x*y, KOKKOS_LAMBDA (const int idx, double& intermed) {
			const size_t kk = idx % x;
      const size_t jj = idx / x;

      if (halo_depth-1 < jj && jj < y-halo_depth && halo_depth-1 < kk && kk < x-halo_depth) {
            const int index = kk + jj*x;
            u[index] = (u0[index]
                    + (kx[index+1]*r[index+1] + kx[index]*r[index-1])
                    + (ky[index+x]*r[index+x] + ky[index]*r[index-x]))
                / (1.0 + (kx[index]+kx[index+1])
                        + (ky[index]+ky[index+x]));

            intermed += fabs(u[index]-r[index]);
        }
		},
		err);



    *error = err;
}
