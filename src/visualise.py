#!/usr/bin/env python3
import pathlib
import json
import os
import sys

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# Try to import canonical OUTPUT_DIR from your definitions; fallback if not available.
try:
    from src.definitions import OUTPUT_DIR
except Exception:  # pragma: no cover
    def find_project_root(current_path: pathlib.Path = None) -> pathlib.Path:
        if current_path is None:
            current_path = pathlib.Path(__file__).resolve().parent
        indicators = ['src', '.git', 'requirements.txt']
        if any((current_path / i).exists() for i in indicators):
            return current_path
        if current_path.parent == current_path:
            raise RuntimeError("Could not find project root.")
        return find_project_root(current_path.parent)
    PROJECT_ROOT = find_project_root(pathlib.Path(__file__).resolve())
    OUTPUT_DIR = PROJECT_ROOT / 'data' / 'output'
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_figure_output_path(aggregate_name: str, filename: str) -> pathlib.Path:
    fig_dir = OUTPUT_DIR / "figures" / aggregate_name
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir / filename


def load_and_flatten_results(json_paths, log_name):
    rows = []
    for p in json_paths:
        with open(p, 'r') as f:
            raw = json.load(f)
        for r in raw:
            base = {
                'anom_alg': r['parameters']['anom_alg'],
                'z': r['parameters']['z'],
                'time_window': r['parameters'].get('time_window'),
                'mode': r['parameters'].get('mode'),
                'ngram_size': r['parameters'].get('ngram_size'),
                'explicit': r['parameters'].get('explicit'),
                'source_file': os.path.basename(p),
                'log_name': log_name,
            }
            base['ratio_of_remaining_directly_follows'] = r.get('ratio_of_remaining_directly_follows')
            base['fitness'] = r.get('fitness')
            if 'reidentification_protection' in r:
                base['reidentification_protection'] = r.get('reidentification_protection')
            elif 'reidentification_risk' in r:
                base['reidentification_protection'] = r.get('reidentification_risk')
            else:
                base['reidentification_protection'] = None

            stats = r.get('anonymized_log_stats', {}) or {}
            base['event_removal_rate'] = stats.get('event_removal_rate')
            base['trace_removal_rate'] = stats.get('trace_removal_rate')
            rows.append(base)
    return pd.DataFrame(rows)


def _wrap_label(text, max_len=16):
    if len(text) <= max_len:
        return text
    parts = text.split(' ')
    if len(parts) == 1:
        return text
    for i in range(1, len(parts)):
        left = ' '.join(parts[:i])
        right = ' '.join(parts[i:])
        if abs(len(left) - len(right)) <= max_len / 2:
            return f"{left}\n{right}"
    mid = len(parts) // 2
    return f"{' '.join(parts[:mid])}\n{' '.join(parts[mid:])}"


def plot_z_vs_baseline_and_save(df, aggregate_name, save_png=False):
    metrics = [
        'event_removal_rate',
        'trace_removal_rate',
        'ratio_of_remaining_directly_follows',
        'fitness',
        'reidentification_protection'
    ]
    display_names = {
        'event_removal_rate': 'Ratio of remaining events',
        'trace_removal_rate': 'Ratio of remaining traces',
        'ratio_of_remaining_directly_follows': 'Ratio of remaining directly follows',
        'fitness': 'Fitness',
        'reidentification_protection': 'Reidentification protection'
    }

    df = df[df['anom_alg'].isin(['z-anonymity', 'baseline'])].copy()

    ngram_sizes = sorted(df['ngram_size'].dropna().unique())
    if not ngram_sizes:
        raise ValueError(f"No ngram_size values found in the dataframe for '{aggregate_name}'.")

    # preserve insertion order of logs by not sorting
    log_names = list(dict.fromkeys(df['log_name'].tolist()))
    n_metrics = len(metrics)
    n_cols = len(ngram_sizes)
    fig, axes = plt.subplots(
        n_metrics,
        n_cols,
        figsize=(4 * n_cols, 3 * n_metrics),
        sharey='row',
    )
    if n_metrics == 1:
        axes = np.expand_dims(axes, 0)
    if n_cols == 1:
        axes = np.expand_dims(axes, 1)

    # color per log
    palette = sns.color_palette(n_colors=len(log_names))
    log_color = {log: palette[i] for i, log in enumerate(log_names)}

    for i, metric in enumerate(metrics):
        for j, ngram in enumerate(ngram_sizes):
            ax = axes[i][j]
            subset_ngram = df[df['ngram_size'] == ngram]

            for log in log_names:
                subset_log = subset_ngram[subset_ngram['log_name'] == log]
                if subset_log.empty:
                    continue

                color = log_color[log]

                # implicit z-anonymity: solid full-intensity
                subset_z = subset_log[
                    (subset_log['anom_alg'] == 'z-anonymity') &
                    ((subset_log['explicit'] == False) | (subset_log['explicit'].isna()))
                ].sort_values('z')
                if not subset_z.empty:
                    ax.plot(
                        subset_z['z'],
                        subset_z[metric],
                        linestyle='-',
                        alpha=1.0,
                        marker=None,
                        color=color,
                    )

                # baseline implicit only: solid, faded
                subset_baseline = subset_log[
                    (subset_log['anom_alg'] == 'baseline') &
                    ((subset_log['explicit'] == False) | (subset_log['explicit'].isna()))
                ].sort_values('z')
                if not subset_baseline.empty:
                    ax.plot(
                        subset_baseline['z'],
                        subset_baseline[metric],
                        linestyle='-',
                        alpha=0.3,
                        marker=None,
                        color=color,
                    )

                # explicit z-anonymity: dashed
                subset_z_explicit = subset_log[
                    (subset_log['anom_alg'] == 'z-anonymity') &
                    (subset_log['explicit'] == True)
                ].sort_values('z')
                if not subset_z_explicit.empty:
                    ax.plot(
                        subset_z_explicit['z'],
                        subset_z_explicit[metric],
                        linestyle='--',
                        alpha=1.0,
                        marker=None,
                        color=color,
                    )

            ax.set_ylim(-0.02, 1.02)

            if i == 0:
                ax.set_title(f"ngram_size={ngram}", fontsize=10)
            if j == 0:
                label = display_names.get(metric, metric)
                wrapped = _wrap_label(label, max_len=22)
                ax.set_ylabel(wrapped, fontsize=13)
            else:
                ax.set_ylabel('')
            ax.set_xlabel('z', fontsize=10)
            ax.grid(True, linestyle=':', linewidth=0.5)

    # --- single compact legend: one row per log (z-anonymity, explicit z-anonymity, baseline) ---
    fig.subplots_adjust(top=0.92)  # keep legend close to plots

    legend_handles = []
    legend_labels = []
    for log in log_names:
        color = log_color[log]
        # add in exact order per log
        handle_z = Line2D([0], [0], color=color, linestyle='-', linewidth=2, alpha=1.0)
        handle_explicit_z = Line2D([0], [0], color=color, linestyle='--', linewidth=2, alpha=1.0)
        handle_baseline = Line2D([0], [0], color=color, linestyle='-', linewidth=2, alpha=0.3)
        legend_handles.extend([handle_z, handle_explicit_z, handle_baseline])
        legend_labels.extend([
            f"{log}: z-anonymity",
            f"{log}: explicit z-anonymity",
            f"{log}: baseline",
        ])

    fig.legend(
        legend_handles,
        legend_labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.97),
        ncol=3,
        fontsize=12,
        frameon=True,
        handlelength=2.5,
        handletextpad=0.5,
        labelspacing=0.3,
        columnspacing=0.8,
        fancybox=False,
        edgecolor='black',
        borderpad=0.3,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.90])

    base_name = f"{'_'.join(log_names)}_comparison"
    pdf_path = get_figure_output_path(base_name, f"{base_name}.pdf")
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight', pad_inches=0.2, format='pdf')
    print(f"[INFO] Saved comparison PDF to: {pdf_path}")
    if save_png:
        png_path = get_figure_output_path(base_name, f"{base_name}.png")
        fig.savefig(png_path, dpi=300, bbox_inches='tight', pad_inches=0.2, format='png')
        print(f"[INFO] Saved comparison PNG to: {png_path}")
    return fig


def visualize_multiple_logs(log_names, result_dir=None, save_png=False):
    all_dfs = []
    for log_name in log_names:
        if result_dir:
            json_dir = pathlib.Path(result_dir) / log_name
        else:
            json_dir = OUTPUT_DIR / log_name
        print(f"[DEBUG] Looking for result JSONs in: {json_dir}")
        json_paths = sorted([str(p) for p in json_dir.glob("results*.json")])
        if not json_paths:
            existing = list(json_dir.iterdir()) if json_dir.exists() else []
            if existing:
                print(f"[DEBUG] Files present in {json_dir}: {[p.name for p in existing][:30]}")
            else:
                print(f"[DEBUG] Directory missing or empty: {json_dir}")
            raise FileNotFoundError(f"No result JSONs found for log '{log_name}' at {json_dir}")
        df_log = load_and_flatten_results(json_paths, log_name)
        all_dfs.append(df_log)

    combined = pd.concat(all_dfs, ignore_index=True)
    fig = plot_z_vs_baseline_and_save(combined, aggregate_name="_".join(log_names), save_png=save_png)
    plt.show()
    return combined, fig


# === configuration, hardcoded ===
LOG_NAMES = ["env_permit", "Sepsis", "BPIC_2012_O", "BPIC20_PTC"]  # e.g., ["Sepsis", "env_permit"]
RESULT_DIR = None  # override if needed
SAVE_PNG = False  # set True to get PNG as well

if __name__ == "__main__":
    df, fig = visualize_multiple_logs(LOG_NAMES, result_dir=RESULT_DIR, save_png=SAVE_PNG)
