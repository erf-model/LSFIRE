"""Heat and water budget of inputs_lsfire_heat_feedback (lagged, exponential profile, Exner on).
Usage: python3 -I check_heat_feedback_budget.py <run dir>   (needs erf.fire.fire_debug = true, fire.verbosity = 2)"""
import sys, re, numpy as np
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from amrex_plt import read_var
d = sys.argv[1].rstrip('/') + '/'
log = open(d + 'run.log').read()
deck = open(d + 'inputs_lsfire_heat_feedback').read()
Cp, R, p0, Lv, alfg = 1004.5, 287.0, 1.0e5, 2.5e6, 45.0
dt = float(re.search(r'erf.fixed_dt\s*=\s*([\d.]+)', deck).group(1))
ext = [float(x) for x in re.search(r'prob_extent\s*=\s*([\d. ]+)', deck).group(1).split()]
nc = [int(x) for x in re.search(r'n_cell\s*=\s*([\d ]+)', deck).group(1).split()]
dz = ext[2] / nc[2]; vol = np.prod([e / n for e, n in zip(ext, nc)])
L = 1 - np.exp(-((nc[2] - 1) * dz + dz) / alfg)   # z_top from the lowest cell centre

# per step: engine release (heat J, water kg); ERF energy_in/out (first RK stage of each step)
steps = re.split(r'Coarse STEP \d+ starts', log)[1:]
Q, W, ein, eout = [], [], [], []
for s in steps:
    m = re.search(r'heat released .*? = ([\d.e+-]+) J, water vapour = ([\d.e+-]+) kg', s)
    Q.append(float(m.group(1))); W.append(float(m.group(2)))
    m = re.search(r'energy_in=([\d.e+-]+) W\s+energy_out=([\d.e+-]+) W', s)
    ein.append(float(m.group(1))); eout.append(float(m.group(2)))
Q, W, ein, eout = map(np.array, (Q, W, ein, eout))
last = sorted(re.findall(r'plt(\d{5})', ' '.join(__import__('os').listdir(d))))[-1]
r0 = read_var(d + 'plt00000', 'rhotheta'); r1 = read_var(d + 'plt' + last, 'rhotheta')
q0 = read_var(d + 'plt00000', 'rhoQ1');    q1 = read_var(d + 'plt' + last, 'rhoQ1')
pi = lambda rt: (R * rt / p0) ** (R / (Cp - R))       # dry: Pi = (R rho theta / p0)^(R/cv)
print(f'{len(steps)} steps, dt {dt} s, L = 1 - exp(-z_top/alfg) = {L:.12f}')
# H1 interface: what ERF receives at step n+1 = what LSFIRE released over step n
h1 = ein[1:] * dt / Q[:-1] - 1
print(f'H1 ERF energy_in(n+1)*dt / LSFIRE heat(n) - 1:    max |.| = {np.abs(h1).max():.2e}')
# H2 ERF: Cp_d * d(int rho theta) = sum energy_out * dt  (dycore conserves int rho theta)
dI = (r1 - r0).sum() * vol
h2 = Cp * dI / (eout.sum() * dt) - 1
print(f'H2 Cp_d*d(int rho*theta) / sum(energy_out*dt) - 1:    {h2:.2e}')
# H3 Exner: energy_out/energy_in = L / Pi (uniform state at the first injection)
ratio = eout[1] / ein[1]; pi0 = pi(r0.mean())
print(f'H3 step 2: out/in = {ratio:.10f}, L/Pi(initial) = {L/pi0:.10f} (Pi = {pi0:.6f}), rel {ratio/(L/pi0)-1:.2e}')
print(f'   over the run out/in ranges {np.min(eout[1:]/ein[1:]):.6f} .. {np.max(eout[1:]/ein[1:]):.6f}')
# Energy with Pi: Cp_d * int Pi d(rho theta) ~ L * sum Q (Pi at mid state; approximate, Pi changes)
pim = pi(0.5 * (r0 + r1))
e_pi = Cp * (pim * (r1 - r0)).sum() * vol
print(f'E  Cp_d*int Pi_mid d(rho*theta) / (L*sum Q[:-1]) - 1:  {e_pi / (L * Q[:-1].sum()) - 1:.2e}   (no Pi: {Cp*dI/(L*Q[:-1].sum())-1:.2e})')
# Water: d(int rho qv) = L * sum water released (all updates but the last: one-step lag)
dWq = (q1 - q0).sum() * vol
print(f'W  d(int rho*qv) = {dWq:.10e} kg, L*sum water[:-1] = {L*W[:-1].sum():.10e} kg, rel {dWq/(L*W[:-1].sum())-1:.2e}')
print(f'   totals over the run: heat {Q.sum():.6e} J, water {W.sum():.6e} kg, latent/sensible {W.sum()*Lv/Q.sum():.4f}')
