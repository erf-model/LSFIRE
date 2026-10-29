#ifndef PHI_CORRECTION_STATS_H
#define PHI_CORRECTION_STATS_H

#include <AMReX_REAL.H>

namespace ELMFIRE {
struct PhiCorrectionStats {
    int changed_cells = 0;
    int newly_burned_cells = 0;
    int edge_newly_burned_cells = 0;
    int newly_tagged_nodes = 0;
    int newly_burned_nodes = 0;
    int existing_nodes_activated = 0;
    amrex::Real total_phi_decrease = 0.0;
    amrex::Real max_phi_decrease = 0.0;
    amrex::Real new_node_fuel_frac_sum = 0.0;
};
}
#endif
