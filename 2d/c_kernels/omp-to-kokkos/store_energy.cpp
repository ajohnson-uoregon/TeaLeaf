#include "../../shared.h"
#include "Kokkos_Core.hpp"

// Store original energy state
void store_energy(
        int x,
        int y,
        double* energy0,
        double* energy)
{
  Kokkos::parallel_for(x*y, KOKKOS_LAMBDA (const int index)
  {
  energy[index] = energy0[index];
  });
}
