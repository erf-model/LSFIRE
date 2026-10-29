// SurfaceSpreadRate template implementation lives in Rothermel.H.
// This translation unit forces instantiation for FireNode (CPU) and
// FireNodeData (GPU/CPU) so both types link correctly.

#include "Rothermel.H"
#include "../Core/FireNode.H"
#include "../DataStructures/FireNodeData.H"

// Explicit instantiations — ensures the linker finds both overloads
// even in builds that only reference one type.
template void SurfaceSpreadRate<FireNode>(FireNode&, const FuelModelTable&);
template void SurfaceSpreadRate<FireNodeData>(FireNodeData&, const FuelModelTable&);
