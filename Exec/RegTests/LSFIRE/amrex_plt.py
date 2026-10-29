"""Minimal reader for single-level AMReX plotfiles (native FAB, double precision).

Usage:
    from amrex_plt import read_var
    arr = read_var("fire_plt0003600", "fire_phi")   # numpy array indexed [i, j, k]
Command line:
    python3 amrex_plt.py compare PLT_A PLT_B VAR [threshold] [level]
"""
import os
import re
import sys

import numpy as np


def _header(plt):
    with open(os.path.join(plt, "Header")) as f:
        lines = [l.strip() for l in f]
    nvar = int(lines[1])
    names = lines[2:2 + nvar]
    return names


def read_var(plt, var, lev=0, with_origin=False):
    names = _header(plt)
    comp = names.index(var)
    lev = os.path.join(plt, "Level_%d" % lev)
    hs = sorted(f for f in os.listdir(lev) if f.endswith("_H"))
    cell_h = "Cell_H" if "Cell_H" in hs else hs[0]   # ERF terrain plotfiles also hold nodal z (Nu_nd_H)
    prefix = cell_h[:-2]
    with open(os.path.join(lev, cell_h)) as f:
        txt = f.read()
    fabs = re.findall(r"FabOnDisk:\s+(\S+)\s+(\d+)", txt)
    pieces, lo_all, hi_all = [], [], []
    for fname, off in fabs:
        with open(os.path.join(lev, fname), "rb") as f:
            f.seek(int(off))
            hdr = b""
            while not hdr.endswith(b"\n"):
                hdr += f.read(1)
            h = hdr.decode()
            m = re.search(r"\(\((-?\d+),(-?\d+),(-?\d+)\) \((-?\d+),(-?\d+),(-?\d+)\) \((\d+),(\d+),(\d+)\)\) (\d+)", h)
            lo = np.array(m.group(1, 2, 3), int)
            hi = np.array(m.group(4, 5, 6), int)
            ncomp = int(m.group(10))
            shape = hi - lo + 1
            n = int(np.prod(shape))
            f.seek(n * 8 * comp, 1)
            data = np.fromfile(f, dtype="<f8", count=n).reshape(shape[::-1]).T
        pieces.append((lo, data))
        lo_all.append(lo)
        hi_all.append(hi)
    glo = np.min(lo_all, axis=0)
    ghi = np.max(hi_all, axis=0)
    out = np.full(ghi - glo + 1, np.nan)
    for lo, d in pieces:
        s = lo - glo
        out[s[0]:s[0] + d.shape[0], s[1]:s[1] + d.shape[1], s[2]:s[2] + d.shape[2]] = d
    return (out, glo) if with_origin else out


def compare(a, b, var, thr=0.0, lev=0):
    x, y = read_var(a, var, lev), read_var(b, var, lev)
    assert x.shape == y.shape, (x.shape, y.shape)
    burned_x, burned_y = x < thr, y < thr
    inter = np.sum(burned_x & burned_y)
    union = np.sum(burned_x | burned_y)
    print(f"{var}: shape {x.shape}, identical={np.array_equal(x, y)}, max|diff|={np.nanmax(np.abs(x - y)):.3e}")
    print(f"burned cells (phi<{thr}): A={burned_x.sum()} B={burned_y.sum()} IoU={inter / max(union, 1):.6f}")


if __name__ == "__main__":
    if sys.argv[1] == "compare":
        compare(sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5]) if len(sys.argv) > 5 else 0.0,
                int(sys.argv[6]) if len(sys.argv) > 6 else 0)
