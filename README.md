# Filtering at the Edge: Exploring the Privacy-Utility Trade-Off

This repository contains the implementation and experimental code for the paper **"Filtering at the Edge: Exploring the Privacy-Utility Trade-Off"** by Maximilian Weisenseel, Fabian Sandkuhl, Henrik Kirchmann, Matthias Weidlich and Florian Tschorsch, submitted to COMINDS2025: 4th Workshop on Collaboration Mining for Distributed Systems at ICPM 2025.

## Overview

This work presents an approach for **distributed filtering of event data directly at event sources**.

The project implements **z-anonymity** and **explicit z-anonymity** techniques that filter event streams based on different behavioral predicates, ensuring that only behavioral information observed by at least `z` distinct cases within a time window is released. This approach realizes fundamental privacy design strategies of *minimization* and *separation* by design.


## Project Structure

```
z_anonymity_pm/
├── data/
│   ├── input/                    # Event log datasets (.xes.gz files)
│   │   ├── env_permit.xes.gz          # Environmental Permit dataset
│   │   ├── Sepsis.xes.gz              # Sepsis Cases dataset  
│   │   ├── BPI_Challenge_2012_O.xes.gz # BPI Challenge 2012 (Offers)
│   │   └── BPIC20_PrepaidTravelCost.xes.gz # BPI Challenge 2020
│   └── output/                   # Experimental results and generated figures
│       ├── env_permit/           # Results for Environmental Permit dataset
│       ├── Sepsis/               # Results for Sepsis dataset
│       ├── BPIC_2012_O/          # Results for BPI Challenge 2012 dataset
│       ├── BPIC20_PTC/           # Results for BPI Challenge 2020 dataset
│       └── figures/              # Generated comparison plots and visualizations
├── src/
│   ├── anonymization/            # Anonymization algorithm implementations
│   │   ├── z_anonymity.py            # Standard z-anonymity (ngram_z_anonymity.py with n =1)
│   │   ├── ngram_z_anonymity.py      # N-gram based z-anonymity 
│   │   └── baseline.py               # Baseline anonymization
│   ├── evaluation/               # Privacy and utility metrics
│   │   ├── metrics.py                # Core evaluation metrics
│   │   ├── reidentification_risk.py  # Privacy risk assessment
│   │   ├── follows_relations.py      # Utility preservation metrics
│   │   └── event_log_stats.py        # Statistical analysis
│   ├── utils/                    # Helper utilities
│   │   └── log_utils.py              # Event log processing utilities
│   ├── test_z_anonymity.py       # Main experimental script
│   ├── visualise.py              # Results visualization and plotting
│   └── definitions.py            # Project configuration and paths
├── requirements.txt              # Python dependencies
└── README.md                    # This file
```

## Setup

Make sure you are using **Python 3.11** to run the scripts.

Install all required packages using the [`requirements.txt`](requirements.txt) file.


### Dataset Preparation

1. Download the datasets in XES format from the provided sources
2. Compress them using gzip (`.xes.gz` format)
3. Place them in the `data/input/` directory with the exact names shown above


## Reproducing Paper Experiments

### Running the Complete Experiment Suite

To reproduce all results from the paper:

```bash
python src/test_z_anonymity.py
```

This script will:
1. Process all four datasets in the `data/input/` directory
2. Run experiments with z-values from 1 to 30
3. Test n-gram sizes of 1, 2, and 3
4. Compare explicit vs. non-explicit modes
5. Generate results in JSON format stored in `data/output/{dataset_name}/`

### Experimental Configuration

**Paper Methodology**: The experiments simulate distributed event streams by treating each distinct `org:resource` (or `org:group` for Sepsis) as an originating stream, modeling the event logs as collections of distributed, concurrent data streams.

The experiments use the following parameters:
- **Z-values**: 1 to 30 (sweeping anonymity parameter)
- **Time windows**: 259,200 seconds (72 hours/3 days)
- **Behavioral Predicates**: Activity sequences of length 1, 2, and 3
- **Algorithms**: Standard z-anonymity, explicit z-anonymity, and centralized baseline
- **Parallelization**: Uses 80% of available CPU cores for efficiency
  - **Global configuration**: Modify `cores_to_use = int(total_cores * 0.8)` in [`src/test_z_anonymity.py`](src/test_z_anonymity.py) (line 163)
    - Change multiplier: `0.5` for 50% of cores, `1.0` for all cores, etc.


### Running the Evaluation

To execute the complete evaluation pipeline that reproduces the paper results:

#### 1. Run Main Experiments
```bash
# Run all experiments (this may take several hours)
python src/test_z_anonymity.py
```

#### 2. Generate Visualizations  
```bash
# Create comparison plots and figures
python src/visualise.py
```

#### 3. Custom Evaluation
```python
# For individual dataset evaluation
from src.test_z_anonymity import test_different_z_values_with_pool

# Example: Run evaluation for specific parameters
test_different_z_values_with_pool(
    log_path="data/input/Sepsis.xes.gz",
    time_windows=[259200],  # 72 hours
    z_values=[1, 5, 10, 15, 20, 25, 30],
    mode='ngram',
    ngram_size=2,
    explicit=True,
    source_attribute='org:group',  # 'org:resource' for other datasets
    log_name="Sepsis",
    seed=42
)
```

The evaluation workflow:
1. **Data Loading**: Loads event logs from `data/input/`
2. **Source Simulation**: Partitions logs by `org:resource`/`org:group` 
3. **Filtering**: Applies z-anonymity algorithms to each source independently
4. **Reassembly**: Combines filtered streams preserving trace order
5. **Metric Calculation**: Computes privacy and utility metrics
6. **Results Storage**: Saves detailed results as JSON files in `data/output/`

### From Event Logs to Distributed Streams

The evaluation transforms centralized event logs into distributed streaming scenarios:
1. **Source Partitioning**: Each `org:resource`/`org:group` becomes a separate event source
2. **Stream Filtering**: Each source applies (explicit) z-anonymity independently  
3. **Stream Reassembly**: Filtered events are recombined preserving original trace ordering
4. **Evaluation**: Anonymized logs are compared against originals using privacy and utility metrics

### Running Individual Dataset Experiments

For testing a specific dataset:

```python
from src.test_z_anonymity import test_different_z_values_with_pool

# Example for Sepsis dataset
test_different_z_values_with_pool(
    log_path="data/input/Sepsis.xes.gz",
    time_windows=[259200],
    z_values=list(range(1, 31)),
    mode='ngram',
    ngram_size=3,
    explicit=False,
    source_attribute='org:group',  # Note: Sepsis uses 'org:group'
    log_name="Sepsis",
    cores_to_use=4,
    seed=42
)

# For other datasets, use 'org:resource' as source_attribute
```

## Generating Visualizations

To create the comparison plots and visualizations shown in the paper:

```bash
python src/visualise.py
```

This will:
1. Load experimental results from all four datasets
2. Generate comprehensive comparison plots
3. Save figures to `data/output/figures/`
4. Display interactive plots for analysis

### Customizing Visualizations

You can modify the visualization configuration in `src/visualise.py`:

```python
# Current configuration includes all four datasets
LOG_NAMES = ["env_permit", "Sepsis", "BPIC_2012_O", "BPIC20_PTC"]
SAVE_PNG = True  # Enable PNG export in addition to PDF
```

## Evaluation Metrics

Following the paper's methodology, the implementation evaluates both privacy and utility using specific metrics:

### Privacy Metrics

#### Re-identification Protection (A* Projection)
- **Implementation**: [`src/evaluation/reidentification_risk.py`](src/evaluation/reidentification_risk.py)
  - Function: `calculate_reidentification_risk()` (lines 132-396)
  - Specific A* projection: `detailed_counts_projection_A()` (lines 70-129)
- **Methodology**: Samples k = ⌈0.1 × max_trace_length⌉ activity-timestamp pairs with day-granular timestamps
- **Assessment**: Measures fraction of traces that remain uniquely identifiable 
- **Protection Score**: Protection = 1 - (fraction of unique traces) 
- **Usage**: Called in [`src/test_z_anonymity.py`](src/test_z_anonymity.py) (lines 112-120)
- **Interpretation**: Higher values indicate stronger protection against re-identification attacks

### Utility Metrics

#### Data Preservation Metrics
- **Implementation**: [`src/evaluation/event_log_stats.py`](src/evaluation/event_log_stats.py)
  - Function: `get_event_log_stats()` (lines 1-14)
- **Ratio of Remaining Events (RRE)**: `event_removal_rate` - Fraction of original events retained after filtering
- **Ratio of Remaining Traces (RRT)**: `trace_removal_rate` - Fraction of original traces retained after filtering
- **Usage**: Called in [`src/test_z_anonymity.py`](src/test_z_anonymity.py) (line 95)

#### Behavioral Preservation Metrics  
- **Implementation**: [`src/evaluation/metrics.py`](src/evaluation/metrics.py)
  - **Directly-Follows Relations**: `get_ratio_of_remaining_directly_follows()` (lines 190-203)
    - Helper function: `get_directly_follows_relations()` (lines 174-187)
  - **Fitness Score**: `compute_fitness()` (lines 206-229)
- **Preservation of Directly-Follows Relations (RDF)**: Fraction of original directly-follows relations preserved
- **Fitness**: Conformance score of anonymized log against process model discovered from original log using inductive miner and token-based replay
- **Usage**: Called in [`src/test_z_anonymity.py`](src/test_z_anonymity.py) (lines 96-97)

#### Key Findings from Paper
- **Non-linear Dependencies**: Privacy and utility show non-linear relationships, creating opportunities for favorable trade-offs
- **Dataset Variability**: Different datasets exhibit varying sensitivity to anonymization parameters
- **Explicit vs. Standard**: Explicit z-anonymity generally preserves more utility while providing comparable privacy protection

## Results Structure

Experimental results are organized as follows:

```
data/output/
├── {dataset_name}/
│   ├── results_ngram_ngram1.json         # 1-gram, non-explicit
│   ├── results_ngram_ngram1_explicit.json # 1-gram, explicit
│   ├── results_ngram_ngram2.json         # 2-gram, non-explicit
│   ├── results_ngram_ngram2_explicit.json # 2-gram, explicit
│   ├── results_ngram_ngram3.json         # 3-gram, non-explicit
│   └── results_ngram_ngram3_explicit.json # 3-gram, explicit
└── figures/
    └── {combined_dataset_comparison}/     # Multi-dataset comparison plots
```

Each JSON file contains detailed metrics for all tested z-values and algorithms.









