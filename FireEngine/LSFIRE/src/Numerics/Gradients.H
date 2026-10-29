#ifndef ELMFIRE_GRADIENTS_H
#define ELMFIRE_GRADIENTS_H

#include <AMReX_MultiFab.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Math.H>
#include "../DataStructures/GpuNarrowBand.H"

/**
 * @file Gradients.H
 * @brief Gradient calculations and flux limiters for level set method
 *
 * This module implements gradient calculations for the level set field φ,
 * including normal vector computation and flux-limited gradients.
 */

/**
 * @brief Compute normal vectors from level set gradient
 *
 * Calculates the unit normal vector to the fire front for each node in the
 * narrow band. The normal vector points in the direction of fire spread
 * (from burned to unburned).
 *
 * This is an EXACT replication of ELMFIRE Fortran CALC_NORMAL_VECTORS
 * (elmfire_level_set.f90:1681-1719)
 *
 * Algorithm:
 *   1. Compute centered difference gradient: ∇φ = (∂φ/∂x, ∂φ/∂y)
 *   2. Calculate magnitude: |∇φ| = √(∂φ/∂x² + ∂φ/∂y²)
 *   3. Normalize: n = ∇φ / |∇φ|
 *
 * @param cells Narrow band tracker containing nodes to update
 * @param phi Level set field (signed distance function)
 * @param half_rcellsize Half of reciprocal cell size (0.5 / Δx)
 * @param istep RK2 step number (1 or 2)
 *
 * @note For istep=1, also stores φ_old for time integration
 * @note Gradients are computed using centered differences
 * @note Results stored in: node.normx, node.normy, node.dphidx, node.dphidy
 */
void CalcNormalVectors(GpuNarrowBand& cells,
                       const amrex::MultiFab& phi,
                       amrex::Real half_rcellsize,
                       int istep,
                       const amrex::MultiFab* dead = nullptr);

// Inline so it is visible inside GPU device lambdas in Gradients.cpp
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real HalfSuperbee(amrex::Real r)
{
    amrex::Real t1 = amrex::min(amrex::Real(0.5) * r, amrex::Real(1.0));
    amrex::Real t2 = amrex::min(r, amrex::Real(0.5));
    return amrex::max(amrex::Real(0.0), amrex::max(t1, t2));
}

/**
 * @brief Compute CFL condition and limit gradients for all nodes
 *
 * Combined function that:
 *   1. Computes CFL condition to determine stable timestep
 *   2. Applies flux limiters to all nodes in narrow band
 *
 * This is a simplified version of ELMFIRE's CFL_AND_FLUX_LIMITER
 * (elmfire_level_set.f90:1986-2102)
 *
 * @param cells Narrow band tracker
 * @param phi Level set field
 * @param rcellsize Reciprocal cell size (1 / Δx)
 * @param dt Timestep (modified to satisfy CFL if istep=1)
 * @param istep RK2 step number (1 or 2)
 * @param target_cfl Target CFL number (typically 0.5-0.9)
 *
 * @note CFL condition: dt ≤ CFL × Δx / max(|u|)
 * @note Only computes CFL on istep=1 (first RK2 stage)
 */
void CflAndFluxLimiter(GpuNarrowBand& cells,
                       const amrex::MultiFab& phi,
                       amrex::Real rcellsize,
                       amrex::Real& dt,
                       int istep,
                       amrex::Real target_cfl = 0.9,
                       const amrex::MultiFab* dead = nullptr);

#endif
