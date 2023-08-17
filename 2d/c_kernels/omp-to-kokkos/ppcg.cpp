#include "../../shared.h"
#include "Kokkos_Core.hpp"
#include "shared.hpp"

/*
 *		PPCG SOLVER KERNEL
 */

// Initialises the PPCG solver
void ppcg_init(
        const int x,
        const int y,
        const int halo_depth,
        double theta,
        KView r,
        KView sd)
{



		Kokkos::parallel_for(x*y, KOKKOS_LAMBDA (const int idx) {
      const size_t kk = idx % x;
      const size_t jj = idx / x;

      if (halo_depth-1 < jj && jj < y-halo_depth && halo_depth-1 < kk && kk < x-halo_depth) {
            const int index = kk + jj*x;
            sd[index] = r[index] / theta;
        }
		});


}

// The PPCG inner iteration
void ppcg_inner_iteration(
        const int x,
        const int y,
        const int halo_depth,
        double alpha,
        double beta,
        KView u,
        KView r,
        KView kx,
        KView ky,
        KView sd)
{



		Kokkos::parallel_for(x*y, KOKKOS_LAMBDA (const int idx) {
      const size_t kk = idx % x;
      const size_t jj = idx / x;

      if (halo_depth-1 < jj && jj < y-halo_depth && halo_depth-1 < kk && kk < x-halo_depth) {
            const int index = kk + jj*x;
            const double smvp = SMVP(sd);
            r[index] -= smvp;
            u[index] += sd[index];
        }
		});






		Kokkos::parallel_for(x*y, KOKKOS_LAMBDA (const int idx) {
      const size_t kk = idx % x;
      const size_t jj = idx / x;

      if (halo_depth-1 < jj && jj < y-halo_depth && halo_depth-1 < kk && kk < x-halo_depth) {
            const int index = kk + jj*x;
            sd[index] = alpha*sd[index] + beta*r[index];
        }
		});


}
