import time
import json
from typing import Literal
from src.utils.log_utils import load_event_log
from src.definitions import get_output_path
from src.anonymization.z_anonymity import apply_z_anonymity
from src.anonymization.ngram_z_anonymity import apply_ngram_z_anonymity
from src.anonymization.baseline import apply_baseline
from src.evaluation.metrics import get_ratio_of_remaining_directly_follows, compute_fitness
from src.evaluation.follows_relations import compare_eventual_follows_relations
from src.evaluation.event_log_stats import get_event_log_stats
from src.evaluation.follows_relations import compare_eventual_follows_relations


from src.evaluation.reidentification_risk import calculate_reidentification_risk
import pm4py
import numpy as np
from pathlib import Path
#from src.evaluation.reidentification_risk import calculate_reidentification_risk
import os

# Constants
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
#EVENT_LOG_PATH = os.path.join(SCRIPT_DIR, 'data')
EVENT_LOG_PATH = Path(os.path.join(SCRIPT_DIR, '../data/input1/')).resolve()


def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj

def anonymize_log(original_log, z, time_window, mode='single', ngram_size=1, explicit=False, source_attribute=None):
    """Apply z-anonymity to a log."""
    if mode == 'single':
        return apply_z_anonymity(original_log, z, time_window, explicit, source_attribute)
    else:  # mode == 'ngram'
        return apply_ngram_z_anonymity(original_log, z, time_window, ngram_size, explicit, source_attribute)

def test_different_z_values(
    log_path: str,
    time_windows: list,
    z_values: list,
    mode: Literal['single', 'ngram'] = 'single',
    ngram_size: int = 3,
    explicit: bool = False,
    source_attribute: str = None,
    log_name: str = None,
):
    """
    Test different combinations of z-values and time windows on the event log.
    
    Args:
        log_path: Path to the event log file
        time_windows: List of time windows to test (in seconds)
        z_values: List of z values to test
        mode: Whether to use single-activity ('single') or n-gram ('ngram') z-anonymity
        ngram_size: Size of n-grams to use when mode='ngram'
        explicit: Whether to use explicit z-anonymity mode
    """
    # Load the original log
    print(f"Loading log from {log_path}...")
    original_log = load_event_log(log_path)
    print(f"Log loaded: {len(original_log)} traces\n")


    
    # Determine output filename based on mode and explicit flag
    mode_suffix = f"_{mode}"
    if mode == 'ngram':
        mode_suffix += f"_ngram{ngram_size}"
    if explicit:
        mode_suffix += "_explicit"
    
    results = []
    for z in z_values:
        for window in time_windows:
            print(f"Testing z={z}, time_window={window} seconds, mode={mode}, explicit={explicit}")
            
            # Apply z-anonymity
            anonymized_log, number_of_edited_traces = anonymize_log(original_log, z, window, mode, ngram_size, explicit, source_attribute)
            
            # Get anonymized log stats
            anonymized_log_stats = get_event_log_stats(original_log, anonymized_log, number_of_edited_traces)

            # Evaluate utility
            #utility_metrics = compare_eventual_follows_relations(original_log, anonymized_log)
            ratio_of_remaining_directly_follows = get_ratio_of_remaining_directly_follows(original_log, anonymized_log)
            fitness = compute_fitness(original_log, anonymized_log)

            # Evaluate reidentification risk (mean), handle empty log
            if len(anonymized_log) == 0:
                reidentification_protection = None
                reidentification_protection_A_star = None
            else:
                if mode == 'ngram':
                    projection = 'N'
                else: projection = 'E'
                reid_risk = calculate_reidentification_risk(anonymized_log, projection=projection, ngram_size=ngram_size)
                reidentification_protection = 1- reid_risk['risk_metrics']['mean']
                reid_risk = calculate_reidentification_risk(anonymized_log, projection='A*', ngram_size=ngram_size)
                reidentification_protection_A_star = 1- reid_risk['risk_metrics']['mean']
            # Store results
            result = {
                'parameters': {
                    'anom_alg': 'z-anonymity',
                    'z': z,
                    'time_window': window,
                    'mode': mode,
                    'ngram_size': ngram_size if mode == 'ngram' else None,
                    'explicit': explicit
                },
                'anonymized_log_stats': anonymized_log_stats,
                'ratio_of_remaining_directly_follows': ratio_of_remaining_directly_follows,
                'fitness': fitness,
                'reidentification_protection': reidentification_protection,
                'reidentification_protection_A_star': reidentification_protection_A_star
            }
            results.append(result)
            
            print()  # Add empty line between results

    # Baseline, fitler offiline and central
    print(f"Baseline offline and central filtering, mode={mode}, ngram_size={ngram_size}")

    #''''
    for z in z_values:
        anonymized_log, number_of_edited_traces = apply_baseline(original_log, z, window, mode, ngram_size, explicit)

        # Get anonymization stats
        anonymized_log_stats = get_event_log_stats(original_log, anonymized_log, number_of_edited_traces)

        # Evaluate utility
        #utility_metrics = compare_eventual_follows_relations(original_log, anonymized_log)
        ratio_of_remaining_directly_follows = get_ratio_of_remaining_directly_follows(original_log, anonymized_log)
        fitness = compute_fitness(original_log, anonymized_log)

        # Evaluate reidentification risk (mean), handle empty log
        if len(anonymized_log) == 0:
            reidentification_protection = None
            reidentification_protection_A_star = None
        else:
            reid_risk = calculate_reidentification_risk(anonymized_log, ngram_size=ngram_size)
            reidentification_protection = 1- reid_risk['risk_metrics']['mean']
            reid_risk = calculate_reidentification_risk(anonymized_log, projection='A*', ngram_size=ngram_size)
            reidentification_protection_A_star = 1 - reid_risk['risk_metrics']['mean']

        # Store results
        result = {
            'parameters': {
                'anom_alg': 'baseline',
                'z': z,
                'time_window': None,
                'mode': mode,
                'ngram_size': ngram_size if mode == 'ngram' else None,
                'explicit': explicit
            },
            'anonymized_log_stats': anonymized_log_stats,
            'ratio_of_remaining_directly_follows': ratio_of_remaining_directly_follows,
            'fitness': fitness,
            'reidentification_protection': reidentification_protection,
            'reidentification_protection_A_star': reidentification_protection_A_star

        }
        results.append(result)

    # Save results with mode-specific filename
    output_filename = f'results{mode_suffix}.json'
    with open(get_output_path(output_filename, log_name), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == '__main__':
    # Test regular z-anonymity
    #print("Testing regular z-anonymity...")
    #test_different_z_values(str(SEPSIS_LOG_PATH), TIME_WINDOWS, Z_VALUES, mode='single', explicit=False)
    
    raw_logs = [f for f in os.listdir(EVENT_LOG_PATH) if f.endswith(".xes.gz")]
    for raw_log in raw_logs:
        log_name = os.path.splitext(os.path.splitext(raw_log)[0])[0]
        if log_name == 'Sepsis':
            source_attribute = 'org:group'
        else:
            source_attribute = 'org:resource'
        full_path = os.path.join(EVENT_LOG_PATH, raw_log)

        TIME_WINDOWS = [259200]  # 24h, 72h in seconds
        Z_VALUES = [1, 3, 5, 8, 10, 12, 15, 18, 20, 25, 30]  # 5 evenly spaced z-values
        Z_VALUES = list(range(1, 31))
        for mode in ['ngram']:
            for ngram_size in [1, 2, 3, 5, 7]:
                for explicit in [False, True]:
                    print(
                        f"\nTesting z-anonymity for {log_name} | "
                        f"mode={mode} | ngram_size={ngram_size} | explicit={explicit}"
                    )
                    test_different_z_values(full_path, TIME_WINDOWS, Z_VALUES, mode=mode, ngram_size=ngram_size,
                                            explicit=explicit, source_attribute=source_attribute, log_name=log_name)





    #test_different_z_values(str(SEPSIS_LOG_PATH), TIME_WINDOWS, Z_VALUES, mode='single', explicit=False)