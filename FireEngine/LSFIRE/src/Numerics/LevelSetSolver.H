#ifndef ELMFIRE_LEVEL_SET_SOLVER_H
#define ELMFIRE_LEVEL_SET_SOLVER_H

#include <AMReX_MultiFab.H>
#include "../DataStructures/GpuNarrowBand.H"

/**
 * @file LevelSetSolver.H
 * @brief Level set equation solver using RK2 time integration
 *
 * This module implements the level set evolution equation:
 *   dphi/dt + u·grad(phi) = 0
 *
 * where phi is the signed distance function and u is the fire velocity.
 */

/**
 * @brief Second-order Runge-Kutta time integration
 *
 * Advances the level set field phi by one timestep using the RK2 method:
 *
 * Step 1 (predictor):
 *   phi_star = phi_n - dt * (ux * dphidx + uy * dphidy)
 *
 * Step 2 (corrector):
 *   phi_new = 0.5 * (phi_n + phi_star - dt * (ux * dphidx_star + uy * dphidy_star))
 *
 * This is an EXACT replication of ELMFIRE Fortran RK2_INTEGRATE
 * (elmfire_level_set.f90:1956-1982)
 *
 * @param cells Narrow band tracker containing nodes to update
 * @param phi Level set field (modified in place)
 * @param dt Timestep size
 * @param istep RK2 step number (1 = predictor, 2 = corrector)
 *
 * @note Uses flux-limited gradients (dphidx_limited, dphidy_limited)
 * @note Clamps phi to [-100, 100] to prevent overflow
 * @note Checks for NaN and replaces with 1.0
 * @note Only updates cells in narrow band (not entire domain)
 */
void Rk2Integrate(GpuNarrowBand& cells,
                  amrex::MultiFab& phi,
                  amrex::Real dt,
                  int istep);

#endif
