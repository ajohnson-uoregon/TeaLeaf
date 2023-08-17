#include "../../shared.h"
#include "Kokkos_Core.hpp"
#include "shared.hpp"

// Store original energy state
void store_energy(
        int x,
        int y,
        KView energy0,
        KView energy)
{
  Kokkos::parallel_for(x*y, KOKKOS_LAMBDA (const int index)
  {
  energy[index] = energy0[index];
  });
}
