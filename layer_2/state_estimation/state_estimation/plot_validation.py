#!/usr/bin/env python3
"""Offline analysis of the state-estimation pipeline vs ground truth.

Reads the CSVs written by validation_node.py and produces:

  * Quantitative metrics (position/velocity RMSE, leg-odom velocity RMSE split
    by support phase, contact duty cycle).
  * A tracking figure (position error + per-axis velocity: GT vs EKF vs leg-odom).
  * A contact-diagnosis figure (ankle efforts + detected contact bands, and
    leg-odom velocity error colored by support phase).

Ground-truth velocity is obtained by finite-differencing the GT *position*.
That is world-frame and independent of however the simulator fills the
Odometry twist (body vs world), which removes a common source of confusion when
"validating" leg odometry. leg_odom and the EKF both report world-frame
velocity, so all three are directly comparable.

Usage (no ROS needed):
  python3 -m state_estimation.plot_validation /tmp/se_validation/<run>
  python3 -m state_estimation.plot_validation <run> --threshold 8 --no-show
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np


# ----------------------------------------------------------------- CSV loading
def _load(path: str) -> dict[str, np.ndarray]:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return {}
        rows = [[float(x) for x in r] for r in reader if r]
    if not rows:
        return {k: np.array([]) for k in header}
    arr = np.array(rows, dtype=float)
    return {k: arr[:, i] for i, k in enumerate(header)}


def _sorted_dedup(t: np.ndarray, *cols: np.ndarray):
    """Sort by time and drop duplicate timestamps (keep first)."""
    order = np.argsort(t)
    t = t[order]
    cols = [c[order] for c in cols]
    keep = np.concatenate(([True], np.diff(t) > 0))
    return (t[keep], *[c[keep] for c in cols])


def _interp(target_t, src_t, src_v):
    """Linear interpolation guarded against empty / single-sample sources."""
    if len(src_t) == 0:
        return np.full_like(target_t, np.nan, dtype=float)
    if len(src_t) == 1:
        return np.full_like(target_t, src_v[0], dtype=float)
    return np.interp(target_t, src_t, src_v)


def _rmse(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if not np.any(m):
        return float('nan')
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))


# ------------------------------------------------------------------- core load
def load_run(run_dir: str, threshold: float | None = None):
    gt = _load(os.path.join(run_dir, 'gt.csv'))
    ekf = _load(os.path.join(run_dir, 'ekf.csv'))
    leg = _load(os.path.join(run_dir, 'legodom.csv'))
    joints = _load(os.path.join(run_dir, 'joints.csv'))
    forces = _load(os.path.join(run_dir, 'forces.csv'))

    if not gt or len(gt.get('t', [])) < 2:
        raise SystemExit(f"No usable ground-truth data in {run_dir}/gt.csv")

    meta = {}
    meta_path = os.path.join(run_dir, 'meta.csv')
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            for row in csv.reader(f):
                if len(row) == 2:
                    meta[row[0]] = float(row[1])

    # Zero the clock to the first GT sample for readable axes.
    t0 = gt['t'][0]
    for d in (gt, ekf, leg, joints, forces):
        if d and 't' in d and len(d['t']):
            d['t'] = d['t'] - t0

    # Contact source: prefer foot ground-reaction force (the real signal);
    # fall back to ankle effort for older recordings without forces.csv.
    if forces and len(forces.get('t', [])):
        contact = dict(source='force', units='N',
                       t=forces['t'],
                       left=np.abs(forces['fz_left']),
                       right=np.abs(forces['fz_right']),
                       thr=threshold if threshold is not None
                           else meta.get('force_contact_threshold', 30.0))
    elif joints and len(joints.get('t', [])):
        contact = dict(source='effort', units='N·m',
                       t=joints['t'],
                       left=np.abs(joints['left_effort']),
                       right=np.abs(joints['right_effort']),
                       thr=threshold if threshold is not None
                           else meta.get('contact_effort_threshold', 5.0))
    else:
        contact = None

    # GT velocity by finite difference of GT position (world frame).
    gtt, gx, gy, gz = _sorted_dedup(gt['t'], gt['x'], gt['y'], gt['z'])
    gt_vx = np.gradient(gx, gtt)
    gt_vy = np.gradient(gy, gtt)
    gt_vz = np.gradient(gz, gtt)

    return dict(gt=gt, ekf=ekf, leg=leg, joints=joints, forces=forces,
                contact=contact, gtt=gtt, gx=gx, gy=gy, gz=gz,
                gt_vx=gt_vx, gt_vy=gt_vy, gt_vz=gt_vz)


# --------------------------------------------------------------------- metrics
def compute_metrics(D) -> str:
    gt, ekf, leg, joints = D['gt'], D['ekf'], D['leg'], D['joints']
    gtt = D['gtt']
    lines = ["", "=" * 64, "STATE-ESTIMATION VALIDATION vs GROUND TRUTH", "=" * 64]

    # ---- EKF position error ----
    if ekf and len(ekf.get('t', [])):
        et = ekf['t']
        ex = _interp(et, gtt, D['gx']); ey = _interp(et, gtt, D['gy']); ez = _interp(et, gtt, D['gz'])
        dx, dy, dz = ekf['x'] - ex, ekf['y'] - ey, ekf['z'] - ez
        norm = np.sqrt(dx**2 + dy**2 + dz**2)
        lines += ["", "[EKF position error vs GT]",
                  f"  RMSE   x={_rmse(ekf['x'], ex):.3f}  y={_rmse(ekf['y'], ey):.3f}  "
                  f"z={_rmse(ekf['z'], ez):.3f}  (m)",
                  f"  |err|  mean={np.nanmean(norm):.3f}  median={np.nanmedian(norm):.3f}  "
                  f"max={np.nanmax(norm):.3f}  (m)"]
        # EKF velocity error
        evx = _interp(et, gtt, D['gt_vx']); evy = _interp(et, gtt, D['gt_vy']); evz = _interp(et, gtt, D['gt_vz'])
        lines += ["", "[EKF velocity error vs GT (finite-diff)]",
                  f"  RMSE   vx={_rmse(ekf['vx'], evx):.3f}  vy={_rmse(ekf['vy'], evy):.3f}  "
                  f"vz={_rmse(ekf['vz'], evz):.3f}  (m/s)"]

    # ---- Leg-odom velocity validation (the direct test of culprit #2) ----
    contact = D['contact']
    if leg and len(leg.get('t', [])):
        lt = leg['t']
        lvx_gt = _interp(lt, gtt, D['gt_vx'])
        lvy_gt = _interp(lt, gtt, D['gt_vy'])
        lvz_gt = _interp(lt, gtt, D['gt_vz'])
        # Support phase at each leg-odom sample, from the contact source.
        if contact is not None:
            le = _interp(lt, contact['t'], contact['left'])
            re = _interp(lt, contact['t'], contact['right'])
            lc = le > contact['thr']
            rc = re > contact['thr']
            double = lc & rc
            single = (lc ^ rc)
        else:
            double = np.ones(len(lt), bool); single = np.zeros(len(lt), bool)

        def rmse3(mask):
            if not np.any(mask):
                return (float('nan'),) * 3
            return (_rmse(leg['vx'][mask], lvx_gt[mask]),
                    _rmse(leg['vy'][mask], lvy_gt[mask]),
                    _rmse(leg['vz'][mask], lvz_gt[mask]))

        allm = np.ones(len(lt), bool)
        lines += ["", "[Leg-odometry velocity error vs GT]  (world frame)",
                  "  phase        vx     vy     vz   (m/s RMSE)   n",
                  "  ---------- ------ ------ ------ ----------- -----"]
        for label, mask in (('all', allm), ('double-supp', double), ('single-supp', single)):
            r = rmse3(mask)
            lines.append(f"  {label:<10} {r[0]:6.3f} {r[1]:6.3f} {r[2]:6.3f}"
                         f"             {int(np.sum(mask))}")
        # Speed-magnitude error: how often leg-odom disagrees badly with GT.
        err = np.sqrt((leg['vx']-lvx_gt)**2 + (leg['vy']-lvy_gt)**2 + (leg['vz']-lvz_gt)**2)
        bad = np.mean(err > 0.5) * 100
        lines += [f"  samples with |v_err| > 0.5 m/s: {bad:.1f}%"]

    # ---- Contact duty cycle (culprit #1 context) ----
    if contact is not None:
        thr = contact['thr']
        lc = contact['left'] > thr
        rc = contact['right'] > thr
        lines += ["", f"[Contact detection from {contact['source']} "
                  f"@ threshold={thr:g} {contact['units']}]",
                  f"  left  contact: {100*np.mean(lc):.1f}% of time",
                  f"  right contact: {100*np.mean(rc):.1f}% of time",
                  f"  double support: {100*np.mean(lc & rc):.1f}%   "
                  f"flight (neither): {100*np.mean(~lc & ~rc):.1f}%"]
        if leg and len(leg.get('t', [])):
            lines += [f"  leg-odom publish rate: {len(leg['t'])} msgs over "
                      f"{gtt[-1]-gtt[0]:.1f}s"]

    lines += ["=" * 64, ""]
    return "\n".join(lines)


# ----------------------------------------------------------------------- plots
def make_figures(D, show=True, save_dir=None):
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    gt, ekf, leg, joints = D['gt'], D['ekf'], D['leg'], D['joints']
    gtt = D['gtt']

    # ===== Figure 1: tracking =====
    fig1, ax = plt.subplots(4, 1, sharex=True, figsize=(12, 10))
    fig1.suptitle('State estimation tracking vs ground truth', fontweight='bold')

    # Position error norm
    if ekf and len(ekf.get('t', [])):
        et = ekf['t']
        ex = _interp(et, gtt, D['gx']); ey = _interp(et, gtt, D['gy']); ez = _interp(et, gtt, D['gz'])
        norm = np.sqrt((ekf['x']-ex)**2 + (ekf['y']-ey)**2 + (ekf['z']-ez)**2)
        ax[0].plot(et, norm, color='crimson', lw=1.2)
        ax[0].set_ylabel('|pos err| (m)')
        ax[0].grid(alpha=0.3)
        ax[0].set_title('EKF position error magnitude')

    # Per-axis velocity: GT vs EKF vs leg-odom
    axis_specs = [('vx', D['gt_vx'], 1), ('vy', D['gt_vy'], 2), ('vz', D['gt_vz'], 3)]
    for name, gtv, row in axis_specs:
        a = ax[row]
        a.plot(gtt, gtv, color='k', lw=1.4, label='GT (finite-diff)')
        if ekf and len(ekf.get('t', [])):
            a.plot(ekf['t'], ekf[name], color='tab:blue', lw=1.0, alpha=0.8, label='EKF')
        if leg and len(leg.get('t', [])):
            a.scatter(leg['t'], leg[name], s=8, color='tab:orange',
                      alpha=0.6, label='leg-odom')
        a.set_ylabel(f'{name} (m/s)')
        a.grid(alpha=0.3)
        if row == 1:
            a.legend(loc='upper right', ncol=3, fontsize=8)
    ax[-1].set_xlabel('time (s)')
    fig1.tight_layout()

    # ===== Figure 2: contact diagnosis =====
    fig2, bx = plt.subplots(2, 1, sharex=True, figsize=(12, 7))
    fig2.suptitle('Contact detection diagnosis', fontweight='bold')
    contact = D['contact']
    thr = contact['thr'] if contact is not None else 0.0
    if contact is not None:
        ct = contact['t']; le = contact['left']; re = contact['right']
        units = contact['units']
        bx[0].plot(ct, le, color='tab:green', lw=0.9, label=f'|left foot {contact["source"]}|')
        bx[0].plot(ct, re, color='tab:purple', lw=0.9, label=f'|right foot {contact["source"]}|')
        bx[0].axhline(thr, color='red', ls='--', lw=1.0, label=f'threshold={thr:g} {units}')
        bx[0].set_ylabel(f'|{contact["source"]}| ({units})')
        bx[0].grid(alpha=0.3)
        bx[0].legend(loc='upper right', fontsize=8)
        # Shade detected contact (any foot) as background.
        lc = le > thr; rc = re > thr
        _shade(bx[0], ct, lc | rc, color='green', alpha=0.06)

    # Leg-odom velocity error colored by support phase
    if leg and len(leg.get('t', [])) and len(gtt):
        lt = leg['t']
        gx = _interp(lt, gtt, D['gt_vx']); gy = _interp(lt, gtt, D['gt_vy']); gz = _interp(lt, gtt, D['gt_vz'])
        err = np.sqrt((leg['vx']-gx)**2 + (leg['vy']-gy)**2 + (leg['vz']-gz)**2)
        if contact is not None:
            lcl = _interp(lt, contact['t'], contact['left']) > thr
            rcl = _interp(lt, contact['t'], contact['right']) > thr
            double = lcl & rcl
            bx[1].scatter(lt[double], err[double], s=10, color='tab:blue',
                          label='double support', alpha=0.7)
            bx[1].scatter(lt[~double], err[~double], s=10, color='tab:red',
                          label='single support', alpha=0.7)
            bx[1].legend(loc='upper right', fontsize=8)
        else:
            bx[1].scatter(lt, err, s=10, color='tab:red', alpha=0.7)
        bx[1].axhline(0.5, color='gray', ls=':', lw=1.0)
        bx[1].set_ylabel('|leg-odom v err| (m/s)')
        bx[1].set_xlabel('time (s)')
        bx[1].grid(alpha=0.3)
        bx[1].set_title('Leg-odom velocity error vs GT '
                        '(red spikes in single-support hint at swing contamination)')
    fig2.tight_layout()

    if save_dir:
        f1 = os.path.join(save_dir, 'tracking.png')
        f2 = os.path.join(save_dir, 'contact.png')
        fig1.savefig(f1, dpi=120); fig2.savefig(f2, dpi=120)
        print(f"[plot] saved {f1}\n[plot] saved {f2}")
    if show:
        plt.show()


def _shade(ax, t, mask, color, alpha):
    """Shade contiguous True regions of mask along time axis t."""
    if not np.any(mask):
        return
    idx = np.where(mask.astype(int))[0]
    splits = np.where(np.diff(idx) > 1)[0]
    groups = np.split(idx, splits + 1)
    for g in groups:
        ax.axvspan(t[g[0]], t[g[-1]], color=color, alpha=alpha, lw=0)


# ------------------------------------------------------------------------ main
def plot_run(run_dir: str, threshold: float | None = None, show: bool = True):
    D = load_run(run_dir, threshold)
    print(compute_metrics(D))
    make_figures(D, show=show, save_dir=run_dir)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run_dir', help='Directory written by validation_node.py')
    ap.add_argument('--threshold', type=float, default=None,
                    help='Override contact effort threshold for re-analysis (N·m)')
    ap.add_argument('--no-show', action='store_true', help='Save PNGs without opening a window')
    a = ap.parse_args()
    plot_run(a.run_dir, threshold=a.threshold, show=not a.no_show)


if __name__ == '__main__':
    main()
