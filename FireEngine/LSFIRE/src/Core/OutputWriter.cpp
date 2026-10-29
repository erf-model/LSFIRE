#include "OutputWriter.H"
#include <fstream>
#include <iomanip>

using namespace amrex;

void OutputWriter::WriteVTK(const std::string& filename,
                           const amrex::Geometry& geom,
                           const amrex::MultiFab& phi,
                           const amrex::MultiFab* velocity_u,
                           const amrex::MultiFab* velocity_v,
                           const amrex::MultiFab* fuel_model,
                           amrex::Real time)
{
    // Get domain info
    const auto* problo = geom.ProbLo();
    const auto* dx = geom.CellSize();
    const Box& domain = geom.Domain();

    int nx = domain.length(0);
    int ny = domain.length(1);
    int nz = 1;

    // MPI CRITICAL FIX: Only IO processor opens and writes to VTK file
    // Non-IO processors still participate in data gathering and MPI_Reduce
    std::ofstream vtkfile;
    if (ParallelDescriptor::IOProcessor()) {
        vtkfile.open(filename + ".vtk");
        if (!vtkfile.is_open()) {
            amrex::Print() << "Error: Could not open " << filename << ".vtk for writing\n";
            return;
        }

        // Write VTK header
        vtkfile << "# vtk DataFile Version 3.0\n";
        vtkfile << "ELMFIRE-CPP Fire Simulation Output (time = " << time << " s)\n";
        vtkfile << "ASCII\n";
        vtkfile << "DATASET STRUCTURED_POINTS\n";
        vtkfile << "DIMENSIONS " << (nx+1) << " " << (ny+1) << " " << (nz+1) << "\n";
        vtkfile << "ORIGIN " << problo[0] << " " << problo[1] << " 0.0\n";
        vtkfile << "SPACING " << dx[0] << " " << dx[1] << " 1.0\n";
        vtkfile << "CELL_DATA " << nx*ny*nz << "\n";
    }

    // Write phi (level set field)
    // MPI FIX: Initialize to 1.0 (unburned) so cells not owned by this rank show correctly
    std::vector<Real> phi_data(nx * ny, 1.0);  // Initialize to unburned value

    // Gather data from all boxes on this rank
    for (MFIter mfi(phi); mfi.isValid(); ++mfi) {
        const Box& box = mfi.validbox();
        const auto& phi_arr = phi.const_array(mfi);

        for (int j = box.smallEnd(1); j <= box.bigEnd(1); ++j) {
            for (int i = box.smallEnd(0); i <= box.bigEnd(0); ++i) {
                int idx = j * nx + i;
                phi_data[idx] = phi_arr(i, j, 0);
            }
        }
    }

    // MPI FIX: Gather data from all ranks to IO processor
    #ifdef AMREX_USE_MPI
    if (ParallelDescriptor::NProcs() > 1) {
        std::vector<Real> global_phi_data(nx * ny, 1.0);

        // Use MPI_Reduce to get MAX value across all ranks for each cell
        // This works because: owned cells have correct phi, unowned cells have 1.0
        // Taking MAX ensures we get the correct value (either phi<1 or phi=1)
        // Note: This assumes phi<=1 always, which is true for our level set
        MPI_Reduce(phi_data.data(), global_phi_data.data(), nx * ny,
                   MPI_DOUBLE, MPI_MIN, ParallelDescriptor::IOProcessorNumber(),
                   ParallelDescriptor::Communicator());

        if (ParallelDescriptor::IOProcessor()) {
            phi_data = global_phi_data;
        }
    }
    #endif

    // MPI CRITICAL FIX: Only IO processor writes phi data to file
    // All ranks have participated in MPI_Reduce, but only IO processor writes
    if (ParallelDescriptor::IOProcessor()) {
        vtkfile << "\nSCALARS phi float 1\n";
        vtkfile << "LOOKUP_TABLE default\n";
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                vtkfile << phi_data[j * nx + i] << "\n";
            }
        }
    }

    // Write velocity components if provided
    if (velocity_u && velocity_v) {
        // Gather velocity data (initialize to 0.0 for unburned regions)
        std::vector<Real> u_data(nx * ny, 0.0);
        std::vector<Real> v_data(nx * ny, 0.0);

        for (MFIter mfi(*velocity_u); mfi.isValid(); ++mfi) {
            const Box& box = mfi.validbox();
            const auto& u_arr = velocity_u->const_array(mfi);
            const auto& v_arr = velocity_v->const_array(mfi);

            for (int j = box.smallEnd(1); j <= box.bigEnd(1); ++j) {
                for (int i = box.smallEnd(0); i <= box.bigEnd(0); ++i) {
                    int idx = j * nx + i;
                    u_data[idx] = u_arr(i, j, 0);
                    v_data[idx] = v_arr(i, j, 0);
                }
            }
        }

        // MPI FIX: Gather velocity from all ranks
        #ifdef AMREX_USE_MPI
        if (ParallelDescriptor::NProcs() > 1) {
            std::vector<Real> global_u(nx * ny, 0.0);
            std::vector<Real> global_v(nx * ny, 0.0);

            // Sum across ranks (unowned cells are 0.0, owned cells have actual values)
            MPI_Reduce(u_data.data(), global_u.data(), nx * ny,
                       MPI_DOUBLE, MPI_SUM, ParallelDescriptor::IOProcessorNumber(),
                       ParallelDescriptor::Communicator());
            MPI_Reduce(v_data.data(), global_v.data(), nx * ny,
                       MPI_DOUBLE, MPI_SUM, ParallelDescriptor::IOProcessorNumber(),
                       ParallelDescriptor::Communicator());

            if (ParallelDescriptor::IOProcessor()) {
                u_data = global_u;
                v_data = global_v;
            }
        }
        #endif

        // MPI CRITICAL FIX: Only IO processor writes velocity to file
        if (ParallelDescriptor::IOProcessor()) {
            vtkfile << "\nVECTORS velocity float\n";
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    int idx = j * nx + i;
                    vtkfile << u_data[idx] << " " << v_data[idx] << " 0.0\n";
                }
            }
        }
    }

    // Write fuel model if provided
    if (fuel_model) {
        // Gather fuel data (initialize to 1 - default fuel model)
        std::vector<int> fuel_data(nx * ny, 1);

        for (MFIter mfi(*fuel_model); mfi.isValid(); ++mfi) {
            const Box& box = mfi.validbox();
            const auto& fuel_arr = fuel_model->const_array(mfi);

            for (int j = box.smallEnd(1); j <= box.bigEnd(1); ++j) {
                for (int i = box.smallEnd(0); i <= box.bigEnd(0); ++i) {
                    int idx = j * nx + i;
                    fuel_data[idx] = static_cast<int>(fuel_arr(i, j, 0));
                }
            }
        }

        // MPI FIX: Gather fuel model from all ranks
        #ifdef AMREX_USE_MPI
        if (ParallelDescriptor::NProcs() > 1) {
            std::vector<int> global_fuel(nx * ny, 1);

            // Max across ranks (assumes fuel_model >= 1, unowned cells = 1)
            MPI_Reduce(fuel_data.data(), global_fuel.data(), nx * ny,
                       MPI_INT, MPI_MAX, ParallelDescriptor::IOProcessorNumber(),
                       ParallelDescriptor::Communicator());

            if (ParallelDescriptor::IOProcessor()) {
                fuel_data = global_fuel;
            }
        }
        #endif

        // MPI CRITICAL FIX: Only IO processor writes fuel model to file
        if (ParallelDescriptor::IOProcessor()) {
            vtkfile << "\nSCALARS fuel_model int 1\n";
            vtkfile << "LOOKUP_TABLE default\n";
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    vtkfile << fuel_data[j * nx + i] << "\n";
                }
            }
        }
    }

    // MPI CRITICAL FIX: Only IO processor closes file and prints message
    if (ParallelDescriptor::IOProcessor()) {
        vtkfile.close();
        amrex::Print() << "Wrote VTK file: " << filename << ".vtk\n";
    }
    // Non-IO processors never opened the file, so nothing to close
}

void OutputWriter::WriteNarrowBandVTK(const std::string& filename,
                                      const amrex::Geometry& geom,
                                      const GpuNarrowBand& narrow_band)
{
    const auto* problo = geom.ProbLo();
    const auto* dx = geom.CellSize();
    const Box& domain = geom.Domain();

    int nx = domain.length(0);
    int ny = domain.length(1);

    std::vector<std::vector<int>> tags(nx, std::vector<int>(ny, 0));

    for (int i = 0; i < narrow_band.GetNumNodes(); i++) {
        const FireNodeData& nd = narrow_band[i];
        if (nd.ix >= 0 && nd.ix < nx && nd.iy >= 0 && nd.iy < ny) {
            tags[nd.ix][nd.iy] = 1;
        }
    }

    // Open VTK file
    std::ofstream vtkfile(filename + "_narrowband.vtk");
    if (!vtkfile.is_open()) {
        amrex::Print() << "Error: Could not open " << filename << "_narrowband.vtk for writing\n";
        return;
    }

    // Write VTK header
    vtkfile << "# vtk DataFile Version 3.0\n";
    vtkfile << "ELMFIRE-CPP Narrow Band Tags\n";
    vtkfile << "ASCII\n";
    vtkfile << "DATASET STRUCTURED_POINTS\n";
    vtkfile << "DIMENSIONS " << (nx+1) << " " << (ny+1) << " 2\n";
    vtkfile << "ORIGIN " << problo[0] << " " << problo[1] << " 0.0\n";
    vtkfile << "SPACING " << dx[0] << " " << dx[1] << " 1.0\n";
    vtkfile << "CELL_DATA " << nx*ny << "\n";

    // Write tags
    vtkfile << "\nSCALARS narrowband_tag int 1\n";
    vtkfile << "LOOKUP_TABLE default\n";

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            vtkfile << tags[i][j] << "\n";
        }
    }

    vtkfile.close();
    amrex::Print() << "Wrote narrow band VTK file: " << filename << "_narrowband.vtk\n";
}

void OutputWriter::WriteCSV(const std::string& filename,
                           const amrex::Geometry& geom,
                           const amrex::MultiFab& data,
                           const std::string& var_name)
{
    // Get domain info
    const auto* problo = geom.ProbLo();
    const auto* dx = geom.CellSize();

    // Open CSV file
    std::ofstream csvfile(filename + ".csv");
    if (!csvfile.is_open()) {
        amrex::Print() << "Error: Could not open " << filename << ".csv for writing\n";
        return;
    }

    // Write header
    csvfile << "x,y," << var_name << "\n";
    csvfile << std::fixed << std::setprecision(6);

    // Write data
    for (MFIter mfi(data); mfi.isValid(); ++mfi) {
        const Box& box = mfi.validbox();
        const auto& arr = data.array(mfi);

        for (int j = box.smallEnd(1); j <= box.bigEnd(1); ++j) {
            for (int i = box.smallEnd(0); i <= box.bigEnd(0); ++i) {
                Real x = problo[0] + (i + 0.5) * dx[0];
                Real y = problo[1] + (j + 0.5) * dx[1];
                csvfile << x << "," << y << "," << arr(i, j, 0) << "\n";
            }
        }
    }

    csvfile.close();
    amrex::Print() << "Wrote CSV file: " << filename << ".csv\n";
}
