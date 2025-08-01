from pm4py.objects.log.obj import EventLog
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import random
import numpy as np
from typing import List, Any, Tuple
from math import ceil


def _is_subset(small: Counter, big: Counter) -> bool:
    for k, v in small.items():
        if big.get(k, 0) < v:
            return False
    return True


def _sample_indices(trace_length: int, number_points: float, rng: random.Random) -> List[int]:
    if trace_length == 0:
        return []
    if number_points > 1:
        k = int(number_points)
    else:
        k = max(1, int(trace_length * number_points))
    if trace_length <= k:
        return list(range(trace_length))
    return rng.sample(range(trace_length), k)


def _extract_ngrams(sequence: List[Any], n: int) -> List[Tuple[Any, ...]]:
    if len(sequence) < n:
        return [tuple(sequence)]
    return [tuple(sequence[i:i + n]) for i in range(len(sequence) - n + 1)]


def _sample_ngrams(ngrams: List[Tuple[Any, ...]], number_points: float, rng: random.Random) -> List[Tuple[Any, ...]]:
    total = len(ngrams)
    if total == 0:
        return []
    if total == 1:
        return ngrams[:]
    if number_points > 1:
        k = int(number_points)
        if total <= k:
            return ngrams[:]
        return rng.sample(ngrams, k)
    else:
        k = max(1, int(ceil(total * number_points)))
        if total <= k:
            return ngrams[:]
        return rng.sample(ngrams, k)


def calculate_reidentification_risk(
    log: EventLog,
    projection: str = 'N',  # 'N' = n-gram, 'E' = activities only, 'A' = activities + timestamps
    number_points: float = 5,  # >=1 absolute, <1 fraction (also governs number of n-grams in N)
    repetitions: int = 10,
    seed: int | None = None,
    n_jobs: int = 1,
    ngram_size: int = 7  # used only when projection == 'N'
) -> dict:
    """
    Compute unicity (re-identification risk) under projections:
      - 'E': activities only,
      - 'A': activities + aligned timestamps,
      - 'N': n-gram uniqueness (sampled n-grams per trace).
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    base_rng = random.Random(seed if seed is not None else random.randrange(2**30))

    # Extract per-trace sequences
    traces_act: List[List[Any]] = []
    traces_time: List[List[Any]] = []
    for trace in log:
        activities = [event.get('concept:name') for event in trace]
        timestamps = [event.get('time:timestamp') for event in trace]
        traces_act.append(activities)
        traces_time.append(timestamps)

    n_traces = len(traces_act)
    if n_traces == 0:
        key = f'{projection}_{ngram_size}_{number_points}_points' if projection == 'N' else f'{projection}_{number_points}_points'
        return {
            'parameters': {
                'projection': projection,
                'number_points': number_points if projection in ('A', 'E') else None,
                'repetitions': repetitions,
                'seed': seed,
                'n_jobs': n_jobs,
                'ngram_size': ngram_size if projection == 'N' else None
            },
            'risk_metrics': {
                key: {
                    'mean': 0.0,
                    'std': 0.0,
                    'min': 0.0,
                    'max': 0.0,
                    'individual_runs': [0.0] * (repetitions if projection != 'N' else 1)
                }
            }
        }

    # Precompute full counters for activity and timestamp (for A/E logic)
    full_counter_activity = [Counter([a for a in acts if a is not None]) for acts in traces_act]
    if projection == 'A':
        full_counter_timestamp = [Counter([t for t in times if t is not None]) for times in traces_time]
    else:
        full_counter_timestamp = None

    def _single_rep_AE(rep_idx: int) -> float:
        rng = random.Random(base_rng.randint(0, 2**30 - 1) + rep_idx)

        sampled_patterns = []
        for i in range(n_traces):
            acts = traces_act[i]
            ts = traces_time[i]
            trace_len = len(acts)

            idxs = _sample_indices(trace_len, number_points, rng)

            sampled_act = Counter([acts[j] for j in idxs if j < len(acts) and acts[j] is not None])
            pattern = {'activity': sampled_act}

            if projection == 'A':
                sampled_ts = Counter([ts[j] for j in idxs if j < len(ts) and ts[j] is not None])
                pattern['timestamp'] = sampled_ts

            sampled_patterns.append(pattern)

        uniques = 0
        memo = {}
        for i, pat in enumerate(sampled_patterns):
            key = (
                tuple(sorted(pat['activity'].items())),
                tuple(sorted(pat['timestamp'].items())) if projection == 'A' else None
            )
            if key in memo:
                uniques += memo[key]
                continue

            match_count = 0
            for j in range(n_traces):
                if not _is_subset(pat['activity'], full_counter_activity[j]):
                    continue
                if projection == 'A':
                    if not _is_subset(pat['timestamp'], full_counter_timestamp[j]):
                        continue
                match_count += 1
                if match_count > 1:
                    break
            is_unique = 1 if match_count == 1 else 0
            memo[key] = is_unique
            uniques += is_unique

        return uniques / n_traces

    def _single_rep_N(rep_idx: int) -> float:
        rng = random.Random(base_rng.randint(0, 2**30 - 1) + rep_idx)

        # Build full n-grams per trace (filtered of None)
        trace_ngrams = []
        for acts in traces_act:
            cleaned = [a for a in acts if a is not None]
            ngrams = _extract_ngrams(cleaned, ngram_size)
            trace_ngrams.append(ngrams)

        # Build inverted index of all n-grams to traces containing them
        gram_to_traces = defaultdict(set)
        for idx, grams in enumerate(trace_ngrams):
            for g in grams:
                gram_to_traces[g].add(idx)

        # Sample per-trace subset of its n-grams according to number_points
        sampled_trace_ngrams = []
        for i, grams in enumerate(trace_ngrams):
            sampled = _sample_ngrams(grams, number_points, rng)
            sampled_trace_ngrams.append(sampled)

        # Determine uniqueness: any sampled gram exclusive to trace
        uniques = 0
        for i, sampled_grams in enumerate(sampled_trace_ngrams):
            found_unique = False
            for g in sampled_grams:
                owners = gram_to_traces.get(g, set())
                if len(owners) == 1 and i in owners:
                    found_unique = True
                    break
            if found_unique:
                uniques += 1
        return uniques / n_traces

    # Dispatch runs
    if projection == 'N':
        # repetitions apply because sampling of which n-grams is randomized
        if n_jobs == 1:
            results = [_single_rep_N(i) for i in range(repetitions)]
        else:
            with ThreadPoolExecutor(max_workers=n_jobs) as ex:
                results = list(ex.map(_single_rep_N, range(repetitions)))
        metric_key = f'N_{ngram_size}_{number_points}_points'
    else:
        if n_jobs == 1:
            results = [_single_rep_AE(i) for i in range(repetitions)]
        else:
            with ThreadPoolExecutor(max_workers=n_jobs) as ex:
                results = list(ex.map(_single_rep_AE, range(repetitions)))
        metric_key = f'{projection}_{number_points}_points'

    arr = np.array(results)
    return {
        'parameters': {
            'projection': projection,
            'number_points': number_points if projection in ('A', 'E') else None,
            'repetitions': repetitions,
            'seed': seed,
            'n_jobs': n_jobs if projection in ('A', 'E', 'N') else None,
            'ngram_size': ngram_size if projection == 'N' else None
        },
        'risk_metrics': {

            'mean': float(arr.mean()),
            'std': float(arr.std(ddof=0)) if len(arr) > 1 else 0.0,
            'min': float(arr.min()),
            'max': float(arr.max()),
            'individual_runs': [float(x) for x in results],

        }
    }
