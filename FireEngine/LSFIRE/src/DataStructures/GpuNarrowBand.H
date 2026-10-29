#ifndef ELMFIRE_GPUNARROWBAND_H
#define ELMFIRE_GPUNARROWBAND_H

/**
 * GpuNarrowBand — GPU/CPU-compatible replacement for NarrowBandTracker.
 *
 * Stores narrow-band fire nodes in a flat std::vector<FireNodeData> (CPU side)
 * mirrored to amrex::Gpu::DeviceVector<FireNodeData> (GPU side) on demand.
 * This removes the doubly-linked FireNode* list, enabling ParallelFor kernels
 * to access all nodes via a contiguous device pointer.
 *
 * CPU operations (add, remove, tag, untag) work on m_nodes directly.
 * GPU compute kernels call GetDevicePtr() after CopyToDevice().
 *
 * Migration note: replaces NarrowBandTracker. Call sites change from
 *   while (node) { ... node = node->next; }
 * to
 *   for (int i = 0; i < nb.GetNumNodes(); i++) { auto& nd = nb[i]; ... }
 */

#include "FireNodeData.H"
#include <AMReX_MultiFab.H>
#include <AMReX_Gpu.H>
#include <vector>
#include <functional>

class GpuNarrowBand {
public:
    GpuNarrowBand();
    ~GpuNarrowBand() = default;

    GpuNarrowBand(const GpuNarrowBand&) = delete;
    GpuNarrowBand& operator=(const GpuNarrowBand&) = delete;

    // ----------------------------------------------------------------
    // Core node management (CPU side)
    // ----------------------------------------------------------------

    /** Add a new node; returns reference so caller can fill remaining fields. */
    FireNodeData& AddNode(int ix, int iy, amrex::Real time);

    /** Remove all nodes satisfying pred in a single compact pass. */
    void RemoveIf(std::function<bool(const FireNodeData&)> pred);

    /** Remove all nodes. */
    void Clear();

    // ----------------------------------------------------------------
    // Access
    // ----------------------------------------------------------------

    int            GetNumNodes() const { return static_cast<int>(m_nodes.size()); }
    bool           IsEmpty()     const { return m_nodes.empty(); }
    FireNodeData&  operator[](int i)       { return m_nodes[i]; }
    const FireNodeData& operator[](int i) const { return m_nodes[i]; }

    /** Raw CPU pointer — for loops that need pointer arithmetic. */
    FireNodeData*       data()       { return m_nodes.data(); }
    const FireNodeData* data() const { return m_nodes.data(); }

    // ----------------------------------------------------------------
    // GPU device mirror
    // ----------------------------------------------------------------

    /** Copy host → device before a ParallelFor kernel. */
    void CopyToDevice();

    /** Copy device → host after a ParallelFor kernel that wrote node fields. */
    void CopyFromDevice();

    /** Device pointer valid after CopyToDevice(). */
    FireNodeData* GetDevicePtr();

    // ----------------------------------------------------------------
    // ELMFIRE narrow-band operations (same semantics as NarrowBandTracker)
    // ----------------------------------------------------------------

    void TagBand(int i_center, int j_center, amrex::Real time,
                 int bandthickness, const amrex::MultiFab& phi);

    void UntagCells(amrex::Real current_time, amrex::Real max_distance);

    void UpdateBurnedCells(amrex::Real current_time,
                           const amrex::MultiFab& phi,
                           amrex::MultiFab& fuel_frac,
                           amrex::MultiFab& time_of_arrival);

    void FindZeroLevelSetCells(const amrex::MultiFab& phi,
                               amrex::Real threshold,
                               std::vector<std::pair<int,int>>& cell_list);

    // ----------------------------------------------------------------
    // Diagnostics
    // ----------------------------------------------------------------

    void PrintStats() const;

private:
    std::vector<FireNodeData>             m_nodes;   // CPU flat array
    amrex::Gpu::DeviceVector<FireNodeData> m_device; // GPU mirror
};

#endif // ELMFIRE_GPUNARROWBAND_H
