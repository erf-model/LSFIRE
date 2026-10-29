#include "Spotting.H"

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

#include <algorithm>
#include <cmath>

namespace Spotting {

using amrex::Real;

namespace {
Generation ParseGeneration(const std::string& s)
{
    if (s == "RANDOM")   return Generation::Random;
    if (s == "PER-AREA") return Generation::PerArea;
    if (s == "PER-MW")   return Generation::PerMW;
    amrex::Abort("fire.spotting.generation_model must be RANDOM, PER-AREA or PER-MW");
    return Generation::PerArea;
}
Distance ParseDistance(const std::string& s)
{
    if (s == "UNIFORM")   return Distance::Uniform;
    if (s == "LOGNORMAL") return Distance::Lognormal;
    if (s == "EMPIRICAL") return Distance::Empirical;
    amrex::Abort("fire.spotting.spotting_distance_model must be UNIFORM, LOGNORMAL or EMPIRICAL");
    return Distance::Empirical;
}
Ignition ParseIgnition(const std::string& s)
{
    if (s == "DIRECT") return Ignition::Direct;
    if (s == "SIMPLE") return Ignition::Simple;
    amrex::Abort("fire.spotting.ignition_model must be DIRECT or SIMPLE (PHYSICAL is not in stage S1)");
    return Ignition::Direct;
}
const char* Name(Generation g) { return g == Generation::Random ? "RANDOM" : g == Generation::PerArea ? "PER-AREA" : "PER-MW"; }
const char* Name(Distance d)   { return d == Distance::Uniform ? "UNIFORM" : d == Distance::Lognormal ? "LOGNORMAL" : "EMPIRICAL"; }
const char* Name(Ignition i)   { return i == Ignition::Direct ? "DIRECT" : "SIMPLE"; }
} // namespace

void Params::Read()
{
    amrex::ParmParse pp("fire.spotting");
    pp.query("enable", enable);
    if (!enable) return;

    std::string s;
    if (pp.query("generation_model", s))        generation = ParseGeneration(s);
    if (pp.query("spotting_distance_model", s)) distance   = ParseDistance(s);
    if (pp.query("ignition_model", s))          ignition   = ParseIgnition(s);
    std::string acc = "LAGRANGIAN";
    pp.query("accumulation_model", acc);
    if (acc != "LAGRANGIAN") amrex::Abort("fire.spotting.accumulation_model: only LAGRANGIAN in stage S1");
    pp.query("per_step_draws", per_step_draws);

    pp.query("tau_embergen", tau_embergen);
    pp.query("critical_spotting_fireline_intensity", critical_flin);
    pp.query("surface_fire_spotting_percent", surface_spotting_pct);
    pp.query("nembers_min", nembers_min);
    pp.query("nembers_max", nembers_max);
    pp.query("ember_gr", ember_gr);
    pp.query("ember_gr_per_mw_vege", ember_gr_per_mw_vege);
    pp.query("ember_sampling_factor", ember_sampling_factor);

    pp.query("min_spotting_distance", min_distance);
    pp.query("max_spotting_distance", max_distance);
    pp.query("mean_spotting_dist", mean_spotting_dist);
    pp.query("normalized_spotting_dist_variance", normalized_dist_variance);
    pp.query("spot_flin_exp", spot_flin_exp);
    pp.query("spot_ws_exp", spot_ws_exp);
    pp.query("p_eps", p_eps);
    pp.query("use_customized_pdf", use_customized_pdf);
    pp.query("mu_downwind", mu_downwind);
    pp.query("sigma_downwind", sigma_downwind);
    pp.query("use_crosswind_distribution", use_crosswind);
    pp.query("mu_crosswind", mu_crosswind);
    pp.query("sigma_crosswind", sigma_crosswind);
    pp.query("tstop", tstop);

    pp.query("pign", pign);
    pp.query("source_fuel_ign_mult", source_fuel_ign_mult);
    pp.query("local_ignition_time", local_ignition_time);
    pp.query("cell_ignition_delay", cell_ignition_delay);

    pp.queryarr("ignitable_fuels", ignitable_fuels);
    pp.query("dump_embers", dump_embers);
    long long sd = static_cast<long long>(seed);
    pp.query("seed", sd);
    seed = static_cast<std::uint64_t>(sd);

    if (ember_sampling_factor <= 0.0) amrex::Abort("fire.spotting.ember_sampling_factor must be > 0");
    if (!(p_eps > 0.0 && p_eps < 1.0)) amrex::Abort("fire.spotting.p_eps must be in (0, 1)");
    if (distance == Distance::Uniform && max_distance < min_distance)
        amrex::Abort("fire.spotting: max_spotting_distance < min_spotting_distance");
    if (distance == Distance::Lognormal && mean_spotting_dist <= 0.0)
        amrex::Abort("fire.spotting: LOGNORMAL needs mean_spotting_dist > 0");
}

void Params::Print() const
{
    if (!enable) return;
    amrex::Print() << "Spotting (S1, ELMFIRE modular model, fire level 0):\n"
                   << "  generation " << Name(generation) << ", distance " << Name(distance)
                   << ", accumulation LAGRANGIAN, ignition " << Name(ignition)
                   << (per_step_draws ? ", per-step draws (ELMFIRE)" : ", draws once per cell") << "\n"
                   << "  tau_embergen " << tau_embergen << " s, critical FLIN " << critical_flin
                   << " kW/m, surface spotting " << surface_spotting_pct << " %, PIGN " << pign
                   << " %, sampling factor " << ember_sampling_factor << ", seed " << seed << "\n";
    if (!ignitable_fuels.empty()) {
        amrex::Print() << "  extra spot-ignitable fuels:";
        for (int f : ignitable_fuels) amrex::Print() << " " << f;
        amrex::Print() << "\n";
    }
}

// ----------------------------------------------------------------------------

Real ErfInv(Real x)
{
    x = std::max(Real(-0.999999), std::min(Real(0.999999), x));
    const Real a = 0.147;
    const Real pi = 3.14159265358979;
    const Real ln = std::log(Real(1.0) - x * x);
    const Real t1 = Real(2.0) / (pi * a) + ln / Real(2.0);
    const Real t2 = std::sqrt(t1 * t1 - ln / a);
    return std::copysign(std::sqrt(t2 - t1), x);
}

Real LognormCdf(Real x, Real mu, Real sigma)
{
    return Real(0.5) * (Real(1.0) + std::erf((std::log(std::max(x, Real(1.0e-6))) - mu) / std::sqrt(Real(2.0)) / sigma));
}

void EmpiricalPdfParameters(Real ws, Real fi, Real& mu, Real& sigma, Real& mu_cw, Real& sigma_cw)
{
    constexpr Real rho_inf = 1.1, c_pg = 1.0, t_inf = 300.0, g = 9.81;
    const Real u_wind = Real(0.447) * std::max(Real(1.0e-3), std::abs(ws)) / Real(0.87);  // 10-m wind [m/s]
    const Real I  = std::max(fi, Real(1.0e-6)) / Real(1000.0);                           // [MW/m]
    const Real lc = std::pow(I * Real(1000.0) / (rho_inf * c_pg * t_inf * std::sqrt(g)), Real(0.67));
    const Real fr = u_wind / std::sqrt(g * lc);
    if (fr <= Real(1.0)) {
        mu    = Real(1.47) * std::pow(I, Real(0.54)) / std::max(std::pow(u_wind, Real(0.55)), Real(1.0e-5)) + Real(1.14);
        sigma = Real(0.86) * std::pow(u_wind, Real(0.44)) / std::max(std::pow(I, Real(0.21)), Real(1.0e-5)) + Real(0.19);
    } else {
        mu    = Real(1.32) * std::pow(I, Real(0.26)) * std::pow(u_wind, Real(0.11)) - Real(0.02);
        sigma = Real(4.95) / std::max(std::pow(I, Real(0.01)), Real(1.0e-5)) / std::max(std::pow(u_wind, Real(0.02)), Real(1.0e-5)) - Real(3.48);
    }
    mu = std::min(mu, Real(5.0));
    mu_cw = 0.0;
    sigma_cw = Real(0.92) * lc;
}

void DistanceParameters(const Params& p, Real ws20, Real flin, Real& mu, Real& sigma, Real& mu_cw, Real& sigma_cw)
{
    if (p.distance == Distance::Empirical && !p.use_customized_pdf) {
        EmpiricalPdfParameters(ws20, flin, mu, sigma, mu_cw, sigma_cw);
    } else if (p.distance == Distance::Empirical) {
        mu = p.mu_downwind; sigma = p.sigma_downwind; mu_cw = p.mu_crosswind; sigma_cw = p.sigma_crosswind;
    } else {
        // UNIFORM does not use mu/sigma; LOGNORMAL from the mean spotting distance (ELMFIRE SPOTTING)
        const Real msd = std::max(p.mean_spotting_dist * std::pow(std::max(flin, Real(0.0)), p.spot_flin_exp)
                                  * std::pow(std::max(ws20, Real(0.0)), p.spot_ws_exp), Real(1.0));
        mu    = std::log(msd * msd / std::sqrt(msd * p.normalized_dist_variance + msd * msd));
        sigma = std::sqrt(std::log(Real(1.0) + msd * p.normalized_dist_variance / (msd * msd)));
        mu_cw = p.mu_crosswind; sigma_cw = p.sigma_crosswind;
    }
    sigma = std::max(sigma, Real(1.0e-5));  // keeps the EMPIRICAL CDF defined
    sigma_cw = std::max(sigma_cw, Real(1.0e-5));
}

Real SampleDistance(const Params& p, Real mu, Real sigma, Real cellsize, double u)
{
    const Real r = static_cast<Real>(u);
    switch (p.distance) {
    case Distance::Uniform:
        return p.min_distance + r * (p.max_distance - p.min_distance);
    case Distance::Lognormal:
        return std::exp(std::sqrt(Real(2.0)) * sigma * ErfInv(Real(2.0) * r - Real(1.0)) + mu);
    case Distance::Empirical: {
        const Real q     = std::sqrt(Real(2.0)) * ErfInv(Real(2.0) * (Real(1.0) - p.p_eps) - Real(1.0));
        const Real x_max = std::nearbyint(std::exp(mu + sigma * q) / cellsize) * cellsize;
        const Real lo = LognormCdf(Real(0.0), mu, sigma);
        const Real hi = LognormCdf(x_max + Real(0.5) * cellsize, mu, sigma);
        const Real rr = r * (hi - lo) + lo;
        const Real d  = std::exp(std::sqrt(Real(2.0)) * sigma * ErfInv(Real(2.0) * rr - Real(1.0)) + mu);
        return std::nearbyint(d / cellsize) * cellsize;
    }
    }
    return 0.0;
}

Real MaxDistance(const Params& p, Real ws20, Real flin, Real cellsize)
{
    Real mu, sigma, mu_cw, sigma_cw;
    DistanceParameters(p, ws20, flin, mu, sigma, mu_cw, sigma_cw);
    if (p.distance == Distance::Uniform) return p.max_distance;
    const Real q = std::sqrt(Real(2.0)) * ErfInv(Real(2.0) * (Real(1.0) - p.p_eps) - Real(1.0));
    return (std::nearbyint(std::exp(mu + sigma * q) / cellsize) + Real(1.0)) * cellsize;
}

Real EmberRate(const Params& p, Real flin, Real cellsize, Real dt_emit)
{
    switch (p.generation) {
    case Generation::Random:
        return Real(0.0);   // handled by the caller (drawn count)
    case Generation::PerArea:
        return p.ember_gr * cellsize * cellsize * dt_emit;
    case Generation::PerMW:
        // ELMFIRE EMBER_TO_EMIT_PER_CELL: FLIN [kW/m] * dx [m] / 1000 = MW, times embers per MW-s
        return flin * cellsize * p.ember_gr_per_mw_vege / Real(1000.0) * dt_emit;
    }
    return 0.0;
}

} // namespace Spotting
