#include "GpuNarrowBand.H"
#include <AMReX_Print.H>
#include <algorithm>
#include <cmath>

using namespace amrex;

GpuNarrowBand::GpuNarrowBand()
{
    m_nodes.reserve(20000);
}

// ============================================================================
// Core node management
// ============================================================================

FireNodeData& GpuNarrowBand::AddNode(int ix, int iy, Real time)
{
    m_nodes.emplace_back(ix, iy, time);
    return m_nodes.back();
}

void GpuNarrowBand::RemoveIf(std::function<bool(const FireNodeData&)> pred)
{
    m_nodes.erase(std::remove_if(m_nodes.begin(), m_nodes.end(), pred),
                  m_nodes.end());
}

void GpuNarrowBand::Clear()
{
    m_nodes.clear();
    m_device.clear();
}

// ============================================================================
// GPU device mirror
// ============================================================================

void GpuNarrowBand::CopyToDevice()
{
#ifdef AMREX_USE_GPU
    m_device.resize(m_nodes.size());
    Gpu::copyAsync(Gpu::hostToDevice,
                   m_nodes.begin(), m_nodes.end(),
                   m_device.begin());
    Gpu::streamSynchronize();
#endif
}

void GpuNarrowBand::CopyFromDevice()
{
#ifdef AMREX_USE_GPU
    AMREX_ASSERT(m_device.size() == m_nodes.size());
    Gpu::copyAsync(Gpu::deviceToHost,
                   m_device.begin(), m_device.end(),
                   m_nodes.begin());
    Gpu::streamSynchronize();
#endif
}

FireNodeData* GpuNarrowBand::GetDevicePtr()
{
#ifdef AMREX_USE_GPU
    return m_device.data();
#else
    // CPU mode: device mirror is not used; return host pointer directly
    return m_nodes.data();
#endif
}

// ============================================================================
// ELMFIRE narrow-band operations
// ============================================================================

void GpuNarrowBand::TagBand(int i_center, int j_center, Real time,
                             int bandthickness, const MultiFab& phi)
{
    Real band_distance = bandthickness * 1.5;

    // Build a quick lookup set of existing (ix,iy) to avoid duplicates
    // (linear scan is fine here — TagBand is called rarely, at ignition)
    auto NodeExists = [this](int i, int j) -> bool {
        for (const auto& nd : m_nodes)
            if (nd.ix == i && nd.iy == j) return true;
        return false;
    };

    for (MFIter mfi(phi); mfi.isValid(); ++mfi) {
        const Box& box = mfi.validbox();
        const auto& phi_arr = phi.array(mfi);

        int i_start = std::max(box.smallEnd(0), i_center - bandthickness);
        int i_end   = std::min(box.bigEnd(0),   i_center + bandthickness);
        int j_start = std::max(box.smallEnd(1), j_center - bandthickness);
        int j_end   = std::min(box.bigEnd(1),   j_center + bandthickness);

        for (int j = j_start; j <= j_end; ++j) {
            for (int i = i_start; i <= i_end; ++i) {
                Real phi_val = phi_arr(i, j, 0);
                if (std::abs(phi_val) < band_distance && !NodeExists(i, j)) {
                    AddNode(i, j, time);
                }
            }
        }
    }
}

void GpuNarrowBand::UntagCells(Real current_time, Real max_distance)
{
    constexpr Real TIME_THRESHOLD   = 604800.0;  // 1 week (matches ELMFIRE Fortran)
    Real           dist_threshold   = max_distance * 0.4;

    RemoveIf([&](const FireNodeData& nd) {
        if (nd.phi > dist_threshold)                              return true;
        if (nd.phi < -dist_threshold)                             return true;
        if (nd.phi > 0.0 && nd.time_added > 0.0 &&
            (current_time - nd.time_added) > TIME_THRESHOLD)     return true;
        return false;
    });
}

void GpuNarrowBand::UpdateBurnedCells(Real current_time,
                                      const MultiFab& /*phi*/,
                                      MultiFab& /*fuel_frac*/,
                                      MultiFab& /*time_of_arrival*/)
{
    for (auto& nd : m_nodes) {
        if (nd.phi < 0.0 && !nd.burned) {
            nd.burned = true;
        }
    }
}

void GpuNarrowBand::FindZeroLevelSetCells(const MultiFab& phi,
                                           Real threshold,
                                           std::vector<std::pair<int,int>>& cell_list)
{
    cell_list.clear();
    for (MFIter mfi(phi); mfi.isValid(); ++mfi) {
        const Box& box = mfi.validbox();
        const auto& phi_arr = phi.array(mfi);
        for (int j = box.smallEnd(1); j <= box.bigEnd(1); ++j)
            for (int i = box.smallEnd(0); i <= box.bigEnd(0); ++i)
                if (std::abs(phi_arr(i, j, 0)) < threshold)
                    cell_list.push_back({i, j});
    }
}

// ============================================================================
// Diagnostics
// ============================================================================

void GpuNarrowBand::PrintStats() const
{
    int burned_count = 0;
    Real min_phi =  1.0e10;
    Real max_phi = -1.0e10;

    for (const auto& nd : m_nodes) {
        if (nd.burned) burned_count++;
        min_phi = std::min(min_phi, nd.phi);
        max_phi = std::max(max_phi, nd.phi);
    }

    Print() << "GpuNarrowBand: " << GetNumNodes() << " nodes"
            << "  burned=" << burned_count
            << "  phi=[" << min_phi << ", " << max_phi << "]\n";
}
