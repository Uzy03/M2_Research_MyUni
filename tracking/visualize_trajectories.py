#!/usr/bin/env python3
"""Visualize player trajectories from tracking clips."""

import argparse
import json
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

FIELD_W = 105.0  # meters
FIELD_H = 68.0


def draw_field(ax):
    ax.set_facecolor('#3a7d44')
    ax.add_patch(patches.Rectangle((0, 0), FIELD_W, FIELD_H,
                                   fill=False, edgecolor='white', linewidth=2))
    ax.plot([FIELD_W / 2, FIELD_W / 2], [0, FIELD_H], color='white', linewidth=1.5)
    ax.add_patch(plt.Circle((FIELD_W / 2, FIELD_H / 2), 9.15,
                            fill=False, color='white', linewidth=1.5))
    ax.plot(FIELD_W / 2, FIELD_H / 2, 'o', color='white', markersize=3)
    # penalty areas
    for x0 in (0, FIELD_W - 16.5):
        ax.add_patch(patches.Rectangle((x0, (FIELD_H - 40.32) / 2), 16.5, 40.32,
                                       fill=False, edgecolor='white', linewidth=1.5))
    for x0 in (0, FIELD_W - 5.5):
        ax.add_patch(patches.Rectangle((x0, (FIELD_H - 18.32) / 2), 5.5, 18.32,
                                       fill=False, edgecolor='white', linewidth=1.5))
    ax.set_xlim(-3, FIELD_W + 3)
    ax.set_ylim(-3, FIELD_H + 3)
    ax.set_aspect('equal')
    ax.axis('off')


def plot_trajectories(ax, npy, mask):
    """npy: (T, N, F), mask: (T, N) True=missing. x,y are [0,1] normalized."""
    T, N, _ = npy.shape
    colors = plt.cm.tab20(np.linspace(0, 1, N))

    for n in range(N):
        valid = ~mask[:, n]
        if valid.sum() < 2:
            continue

        x = npy[:, n, 0] * FIELD_W
        y = npy[:, n, 1] * FIELD_H

        t_idx = np.where(valid)[0]
        # split into continuous segments (gap > 1 frame)
        breaks = np.where(np.diff(t_idx) > 1)[0] + 1
        for seg in np.split(t_idx, breaks):
            if len(seg) < 2:
                continue
            # color gradient: dark at start, bright at end
            for k in range(len(seg) - 1):
                alpha = 0.4 + 0.6 * k / max(len(seg) - 2, 1)
                ax.plot(x[seg[k:k+2]], y[seg[k:k+2]],
                        color=colors[n], linewidth=1.5, alpha=alpha, solid_capstyle='round')

        ax.plot(x[t_idx[0]], y[t_idx[0]], 'o', color=colors[n], markersize=5, zorder=5)
        ax.plot(x[t_idx[-1]], y[t_idx[-1]], '*', color=colors[n], markersize=7, zorder=5)


def main():
    parser = argparse.ArgumentParser(description='Visualize player trajectories from clips.json')
    parser.add_argument('--json_path',
                        default='soccerdata_clips/fps1_sec30_onball_step5s/clips.json')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of clips to visualize')
    parser.add_argument('--out_dir', default='results/trajectory_vis')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.json_path) as f:
        clips = json.load(f)

    clips_sorted = sorted(clips, key=lambda e: e['npy_path'])
    samples = clips_sorted[:args.num_samples]
    base_dir = Path(args.json_path).parent

    for i, entry in enumerate(samples):
        npy  = np.load(base_dir / entry['npy_path'])   # (T, N, F)
        mask = np.load(base_dir / entry['mask_path'])  # (T, N)

        fig, ax = plt.subplots(figsize=(13, 9))
        fig.patch.set_facecolor('#1e3a24')
        draw_field(ax)
        plot_trajectories(ax, npy, mask)

        clip_id = entry.get('clip_id', entry.get('seq_id', str(i)))
        action  = entry.get('action_label', '')
        title   = f'clip: {clip_id}   T={npy.shape[0]} frames   N={npy.shape[1]} players'
        if action:
            title += f'\naction: {action}'
        ax.set_title(title, color='white', fontsize=10, pad=10)

        out_path = out_dir / f'{i:03d}_{clip_id}.png'
        fig.savefig(out_path, dpi=120, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f'[{i+1}/{len(samples)}] {out_path}')

    print(f'\nDone. {len(samples)} figures saved to {out_dir}/')


if __name__ == '__main__':
    main()
