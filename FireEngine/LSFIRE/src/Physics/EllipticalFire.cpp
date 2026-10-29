// EllipticalFireVelocity template implementation lives in EllipticalFire.H.
// Explicit instantiations so both FireNode (CPU) and FireNodeData (GPU/CPU)
// link correctly without duplicating the physics code.

#include "EllipticalFire.H"
#include "../Core/FireNode.H"
#include "../DataStructures/FireNodeData.H"

template void EllipticalFireVelocity<FireNode>(FireNode&, const FuelModelTable&, amrex::Real);
template void EllipticalFireVelocity<FireNodeData>(FireNodeData&, const FuelModelTable&, amrex::Real);
