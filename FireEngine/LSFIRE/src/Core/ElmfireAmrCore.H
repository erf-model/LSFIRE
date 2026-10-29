#ifndef ELMFIRE_AMR_CORE_H
#define ELMFIRE_AMR_CORE_H

#include "FireHierarchy.H"

// Source-compatible adapter for existing standalone/metgrid clients.
// New integrations can derive from FireHierarchy directly.
class ElmfireAmrCore : public FireHierarchy {
public:
    using FireHierarchy::FireHierarchy;
};

#endif // ELMFIRE_AMR_CORE_H
