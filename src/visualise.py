#!/usr/bin/env python3
import pathlib
import json
import os
import sys

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

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


def get_figure_output_path(log_name: str, filename: str) -> pathlib.Path:
    fig_dir = OUTPUT_DIR / "figures" / log_name
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir / filename


def load_and_flatten_results(json_paths):
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
    df = pd.DataFrame(rows)
    return df


def load_and_flatten_results(json_paths):
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
            }
            base['ratio_of_remaining_directly_follows'] = r.get('ratio_of_remaining_directly_follows')
            base['fitness'] = r.get('fitness')
            # reidentification protection variants
            if 'reidentification_protection' in r:
                base['reidentification_protection'] = r.get('reidentification_protection')
            elif 'reidentification_risk' in r:
                base['reidentification_protection'] = r.get('reidentification_risk')
            else:
                base['reidentification_protection'] = None

            base['reidentification_protection_A_star'] = r.get('reidentification_protection_A_star')

            stats = r.get('anonymized_log_stats', {}) or {}
            base['event_removal_rate'] = stats.get('event_removal_rate')
            base['trace_removal_rate'] = stats.get('trace_removal_rate')
            rows.append(base)
    return pd.DataFrame(rows)


def plot_z_vs_baseline_and_save(df, log_name, save_png=False):
    metrics = [
        'event_removal_rate',
        'trace_removal_rate',
        'ratio_of_remaining_directly_follows',
        'fitness',
        'reidentification_protection',
        'reidentification_protection_A_star',
    ]
    display_names = {
        'event_removal_rate': 'Ratio of remaining events',
        'trace_removal_rate': 'Ratio of remaining traces',
        'ratio_of_remaining_directly_follows': 'Ratio of remaining directly follows',
        'fitness': 'Fitness',
        'reidentification_protection': 'Reidentification protection',
        'reidentification_protection_A_star': 'Reidentification protection A*',
    }

    df = df[df['anom_alg'].isin(['z-anonymity', 'baseline'])].copy()

    ngram_sizes = sorted(df['ngram_size'].dropna().unique())
    if not ngram_sizes:
        raise ValueError(f"No ngram_size values found in the dataframe for log '{log_name}'.")

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

    palette = sns.color_palette(n_colors=n_metrics)
    metric_color = {m: palette[i] for i, m in enumerate(metrics)}

    for i, metric in enumerate(metrics):
        for j, ngram in enumerate(ngram_sizes):
            ax = axes[i][j]
            subset_ngram = df[df['ngram_size'] == ngram]

            # implicit z-anonymity: solid full-intensity
            subset_z = subset_ngram[
                (subset_ngram['anom_alg'] == 'z-anonymity') &
                ((subset_ngram['explicit'] == False) | (subset_ngram['explicit'].isna()))
            ]
            if not subset_z.empty:
                sns.lineplot(
                    data=subset_z.sort_values('z'),
                    x='z',
                    y=metric,
                    ax=ax,
                    label="z-anonymity" if (i == 0 and j == 0) else None,
                    linestyle='-',
                    color=metric_color[metric],
                    alpha=1.0,
                    marker=None,
                    err_style=None,
                )

            # baseline implicit only: solid, faded
            subset_baseline = subset_ngram[
                (subset_ngram['anom_alg'] == 'baseline') &
                ((subset_ngram['explicit'] == False) | (subset_ngram['explicit'].isna()))
            ]
            if not subset_baseline.empty:
                sns.lineplot(
                    data=subset_baseline.sort_values('z'),
                    x='z',
                    y=metric,
                    ax=ax,
                    label="baseline" if (i == 0 and j == 0) else None,
                    linestyle='-',
                    color=metric_color[metric],
                    alpha=0.3,
                    marker=None,
                    err_style=None,
                )

            # explicit z-anonymity: dashed
            subset_z_explicit = subset_ngram[
                (subset_ngram['anom_alg'] == 'z-anonymity') &
                (subset_ngram['explicit'] == True)
            ]
            if not subset_z_explicit.empty:
                sns.lineplot(
                    data=subset_z_explicit.sort_values('z'),
                    x='z',
                    y=metric,
                    ax=ax,
                    label="z-anonymity explicit" if (i == 0 and j == 0) else None,
                    linestyle='--',
                    color=metric_color[metric],
                    alpha=1.0,
                    marker=None,
                    err_style=None,
                )

            # vertical padding so 0 and 1 lines are distinct
            ax.set_ylim(-0.02, 1.02)

            if i == 0:
                ax.set_title(f"ngram_size={ngram}")
            if j == 0:
                ax.set_ylabel(display_names.get(metric, metric))
            else:
                ax.set_ylabel('')
            ax.set_xlabel('z')

            if i == 0 and j == 0:
                ax.legend(fontsize='small')
            else:
                if ax.get_legend():
                    ax.get_legend().remove()
            ax.grid(True, linestyle=':', linewidth=0.5)

    plt.suptitle(f"Comparison for log '{log_name}'", y=1.02)
    plt.tight_layout()

    base_name = f"{log_name}_comparison"
    pdf_path = get_figure_output_path(log_name, f"{base_name}.pdf")
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight', format='pdf')
    print(f"[INFO] Saved comparison PDF to: {pdf_path}")
    if save_png:
        png_path = get_figure_output_path(log_name, f"{base_name}.png")
        fig.savefig(png_path, dpi=300, bbox_inches='tight', format='png')
        print(f"[INFO] Saved comparison PNG to: {png_path}")
    return fig



def visualize_all_results_for_log(log_name, result_dir=None, save_png=False):
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
    df = load_and_flatten_results(json_paths)
    fig = plot_z_vs_baseline_and_save(df, log_name=log_name, save_png=save_png)
    plt.show()
    return df, fig


# === configuration, hardcoded ===
LOG_NAME = "BPIC12"  # change to whichever log directory you want to visualize
RESULT_DIR = None  # e.g., "/Users/henrikkirchmann/z_anonymity_pm/data/output" if overriding
SAVE_PNG = False  # set True if you want PNG alongside PDF

# run visualization
if __name__ == "__main__":
    df, fig = visualize_all_results_for_log(LOG_NAME, result_dir=RESULT_DIR, save_png=SAVE_PNG)
