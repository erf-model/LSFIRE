#ifndef ELMFIRE_OUTPUT_WRITER_H
#define ELMFIRE_OUTPUT_WRITER_H

#include <AMReX_MultiFab.H>
#include <AMReX_Geometry.H>
#include <string>
#include "../DataStructures/GpuNarrowBand.H"

/**
 * @brief Write simulation data to files for visualization
 *
 * Supports VTK format for ParaView/VisIt visualization
 * and CSV format for Python/MATLAB post-processing
 */
class OutputWriter {
public:
    /**
     * @brief Write VTK structured grid file
     *
     * @param filename Output filename (without extension)
     * @param geom Domain geometry
     * @param phi Level set field
     * @param velocity_u U-component of velocity (optional)
     * @param velocity_v V-component of velocity (optional)
     * @param fuel_model Fuel model field (optional)
     * @param time Current simulation time
     */
    static void WriteVTK(const std::string& filename,
                        const amrex::Geometry& geom,
                        const amrex::MultiFab& phi,
                        const amrex::MultiFab* velocity_u = nullptr,
                        const amrex::MultiFab* velocity_v = nullptr,
                        const amrex::MultiFab* fuel_model = nullptr,
                        amrex::Real time = 0.0);

    /**
     * @brief Write narrow band tags to VTK
     *
     * @param filename Output filename
     * @param geom Domain geometry
     * @param narrow_band Narrow band tracker
     */
    static void WriteNarrowBandVTK(const std::string& filename,
                                   const amrex::Geometry& geom,
                                   const GpuNarrowBand& narrow_band);

    /**
     * @brief Write CSV file with scalar field
     *
     * @param filename Output filename
     * @param geom Domain geometry
     * @param data Scalar field to write
     * @param var_name Variable name
     */
    static void WriteCSV(const std::string& filename,
                        const amrex::Geometry& geom,
                        const amrex::MultiFab& data,
                        const std::string& var_name);
};

#endif
