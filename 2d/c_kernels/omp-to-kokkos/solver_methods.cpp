#include <stdlib.h>
#include "../../shared.h"
#include "Kokkos_Core.hpp"

/*
 *		SHARED SOLVER METHODS
 */

// Copies the current u into u0
void copy_u(
        const int x,
        const int y,
        const int halo_depth,
        double* u0,
        double* u)
{



		Kokkos::parallel_for(x*y, KOKKOS_LAMBDA (const int idx) {
      const size_t kk = idx % x;
      const size_t jj = idx / x;

      if (halo_depth-1 < jj && jj < y-halo_depth && halo_depth-1 < kk && kk < x-halo_depth) {
            const int index = kk + jj*x;
            u0[index] = u[index];
        }
		});
	

}

// Calculates the current value of r
void calculate_residual(
        const int x,
        const int y,
        const int halo_depth,
        double* u,
        double* u0,
        double* r,
        double* kx,
        double* ky)
{



		Kokkos::parallel_for(x*y, KOKKOS_LAMBDA (const int idx) {
      const size_t kk = idx % x;
      const size_t jj = idx / x;

      if (halo_depth-1 < jj && jj < y-halo_depth && halo_depth-1 < kk && kk < x-halo_depth) {
            const int index = kk + jj*x;
            const double smvp = SMVP(u);
            r[index] = u0[index] - smvp;
        }
		});
	

}

// Calculates the 2 norm of a given buffer
void calculate_2norm(
        const int x,
        const int y,
        const int halo_depth,
        double* buffer,
        double* norm)
{
    double norm_temp = 0.0;




		Kokkos::parallel_reduce(x*y, KOKKOS_LAMBDA (const int idx, double& intermed) {
			const size_t kk = idx % x;
      const size_t jj = idx / x;

      if (halo_depth-1 < jj && jj < y-halo_depth && halo_depth-1 < kk && kk < x-halo_depth) {
            const int index = kk + jj*x;
            intermed += buffer[index]*buffer[index];
        }
		},
		norm_temp);
	


    *norm += norm_temp;
}

// Finalises the solution
void finalise(
        const int x,
        const int y,
        const int halo_depth,
        double* energy,
        double* density,
        double* u)
{



		Kokkos::parallel_for(x*y, KOKKOS_LAMBDA (const int idx) {
      const size_t kk = idx % x;
      const size_t jj = idx / x;

      if (halo_depth-1 < jj && jj < y-halo_depth && halo_depth-1 < kk && kk < x-halo_depth) {
            const int index = kk + jj*x;
            energy[index] = u[index]/density[index];
        }
		});
	

}
