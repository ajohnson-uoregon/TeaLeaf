#include "../../shared.h"
#include "Kokkos_Core.hpp"

/*
 *		CHEBYSHEV SOLVER KERNEL
 */

// Calculates the new value for u.
void cheby_calc_u(
        const int x,
        const int y,
        const int halo_depth,
        double* u,
        double* p)
{



		Kokkos::parallel_for(x*y, KOKKOS_LAMBDA (const int idx) {
      const size_t kk = idx % x;
      const size_t jj = idx / x;

      if (halo_depth-1 < jj && jj < y-halo_depth && halo_depth-1 < kk && kk < x-halo_depth) {
            const int index = kk + jj*x;
            u[index] += p[index];
        }
		});
	

}

// Initialises the Chebyshev solver
void cheby_init(
        const int x,
        const int y,
        const int halo_depth,
        const double theta,
        double* u,
        double* u0,
        double* p,
        double* r,
        double* w,
        double* kx,
        double* ky)
{



		Kokkos::parallel_for(x*y, KOKKOS_LAMBDA (const int idx) {
      const size_t kk = idx % x;
      const size_t jj = idx / x;

      if (halo_depth-1 < jj && jj < y-halo_depth && halo_depth-1 < kk && kk < x-halo_depth) {
            const int index = kk + jj*x;
            const double smvp = SMVP(u);
            w[index] = smvp;
            r[index] = u0[index]-w[index];
            p[index] = r[index] / theta;
        }
		});
	


    cheby_calc_u(x, y, halo_depth, u, p);
}

// The main chebyshev iteration
void cheby_iterate(
        const int x,
        const int y,
        const int halo_depth,
        double alpha,
        double beta,
        double* u,
        double* u0,
        double* p,
        double* r,
        double* w,
        double* kx,
        double* ky)
{



		Kokkos::parallel_for(x*y, KOKKOS_LAMBDA (const int idx) {
      const size_t kk = idx % x;
      const size_t jj = idx / x;

      if (halo_depth-1 < jj && jj < y-halo_depth && halo_depth-1 < kk && kk < x-halo_depth) {
            const int index = kk + jj*x;
            const double smvp = SMVP(u);
            w[index] = smvp;
            r[index] = u0[index]-w[index];
            p[index] = alpha*p[index] + beta*r[index];
        }
		});
	


    cheby_calc_u(x, y, halo_depth, u, p);
}
