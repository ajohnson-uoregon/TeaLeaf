#include "../../settings.h"
#include "shared.hpp"

/*
 *      This is the main interface file for C based implementations.
 */

// Initialisation kernels
void set_chunk_data(
        Settings* settings, int x, int y, int left,
        int bottom, KView cell_x, KView cell_y,
		KView vertex_x, KView vertex_y, KView volume,
		KView x_area, KView y_area);

void set_chunk_state(
        int x, int y, KView vertex_x, KView vertex_y, KView cell_x,
        KView cell_y, KView density, KView energy0, KView u,
        const int num_states, State* state);

void kernel_initialise(
        Settings* settings, int x, int y, KView* density0,
        KView* density, KView* energy0, KView* energy, KView* u,
        KView* u0, KView* p, KView* r, KView* mi,
        KView* w, KView* kx, KView* ky, KView* sd,
        KView* volume, KView* x_area, KView* y_area, KView* cell_x,
        KView* cell_y, KView* cell_dx, KView* cell_dy, KView* vertex_dx,
        KView* vertex_dy, KView* vertex_x, KView* vertex_y,
        double** cg_alphas, double** cg_betas, double** cheby_alphas,
        double** cheby_betas);

void kernel_finalise(
        KView density0, KView density, KView energy0, KView energy,
        KView u, KView u0, KView p, KView r, KView mi,
        KView w, KView kx, KView ky, KView sd,
        KView volume, KView x_area, KView y_area, KView cell_x,
        KView cell_y, KView cell_dx, KView cell_dy, KView vertex_dx,
        KView vertex_dy, KView vertex_x, KView vertex_y,
        double* cg_alphas, double* cg_betas, double* cheby_alphas,
        double* cheby_betas);

// Solver-wide kernels
void local_halos(
        const int x, const int y, const int depth,
        const int halo_depth, const int* chunk_neighbours,
        const bool* fields_to_exchange, KView density, KView energy0,
        KView energy, KView u, KView p, KView sd);

void pack_or_unpack(
        const int x, const int y, const int depth,
        const int halo_depth, const int face, bool pack,
        KView field, double* buffer);

void store_energy(
        int x, int y, KView energy0, KView energy);

void field_summary(
        const int x, const int y, const int halo_depth,
        KView volume, KView density, KView energy0, KView u,
        double* volOut, double* massOut, double* ieOut, double* tempOut);

// CG solver kernels
void cg_init(
        const int x, const int y, const int halo_depth,
        const int coefficient, double rx, double ry, double* rro,
        KView density, KView energy, KView u, KView p,
        KView r, KView w, KView kx, KView ky);
void cg_calc_w(
        const int x, const int y, const int halo_depth, double* pw,
        KView p, KView w, KView kx, KView ky);

void cg_calc_ur(
        const int x, const int y, const int halo_depth,
        const double alpha, double* rrn, KView u, KView p,
        KView r, KView w);

void cg_calc_p(
        const int x, const int y, const int halo_depth,
        const double beta, KView p, KView r);

// Chebyshev solver kernels
void cheby_init(const int x, const int y,
        const int halo_depth, const double theta, KView u, KView u0,
        KView p, KView r, KView w, KView kx,
        KView ky);
void cheby_iterate(const int x, const int y,
        const int halo_depth, double alpha, double beta, KView u,
        KView u0, KView p, KView r, KView w,
        KView kx, KView ky);

// Jacobi solver kernels
void jacobi_init(const int x, const int y,
        const int halo_depth, const int coefficient, double rx, double ry,
        KView density, KView energy, KView u0, KView u,
        KView kx, KView ky);
void jacobi_iterate(const int x, const int y,
        const int halo_depth, double* error, KView kx, KView ky,
        KView u0, KView u, KView r);

// PPCG solver kernels
void ppcg_init(const int x, const int y, const int halo_depth,
        double theta, KView r, KView sd);
void ppcg_inner_iteration(const int x, const int y,
        const int halo_depth, double alpha, double beta, KView u,
        KView r, KView kx, KView ky,
        KView sd);

// Shared solver kernels
void copy_u(
        const int x, const int y, const int halo_depth,
        KView u0, KView u);

void calculate_residual(
        const int x, const int y, const int halo_depth,
        KView u, KView u0, KView r, KView kx,
        KView ky);

void calculate_2norm(
        const int x, const int y, const int halo_depth,
        KView buffer, double* norm);

void finalise(
        const int x, const int y, const int halo_depth,
        KView energy, KView density, KView u);
