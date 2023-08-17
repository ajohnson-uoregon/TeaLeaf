#pragma once

#include <Kokkos_Core.hpp>

#ifdef CUDA
    #define DEVICE Kokkos::Cuda
#endif

#ifdef OPENMP
    #define DEVICE Kokkos::OpenMP
#endif

#ifndef DEVICE
    #define DEVICE Kokkos::OpenMP
#endif

typedef Kokkos::View<double*, DEVICE> KView;
