import copy
from collections import defaultdict
from typing import Tuple

from pm4py.objects.log.obj import EventLog


def extract_ngrams_from_trace(trace_activity_list, n):
    """
    Return list of tuples: (ngram_string, start_pos, end_pos_inclusive)
    """
    if len(trace_activity_list) < n:
        return []
    ngrams = []
    for i in range(len(trace_activity_list) - n + 1):
        ngram = ">".join(trace_activity_list[i : i + n])
        # positions [i, i+n-1]
        ngrams.append((ngram, i, i + n - 1))
    return ngrams


def apply_baseline(
    original_log: EventLog,
    z: int,
    window,  # ignored
    mode: str,
    ngram_size: int,
    explicit: bool,
) -> Tuple[EventLog, int]:
    """
    Baseline anonymization without time window or source separation.

    mode: "single" or "ngram"
    """
    anonymized_log = copy.deepcopy(original_log)

    # Track which trace indices get edited
    edited_traces = set()

    if mode == "single":
        # Build activity -> set of case ids
        activity_to_cases = defaultdict(set)
        for trace in original_log:
            case_id = trace.attributes["concept:name"]
            for ev in trace:
                activity = ev["concept:name"]
                activity_to_cases[activity].add(case_id)

        # Determine qualifying activities (appearing in >= z distinct cases)
        qualifying_activities = {act for act, cases in activity_to_cases.items() if len(cases) >= z}

        """
        # If explicit: we need all cases that contain any qualifying activity, and release all their events
        if explicit:
            # Build set of case_ids to fully keep (for activities that qualify)
            cases_to_keep_full = set()
            for trace in original_log:
                case_id = trace.attributes["concept:name"]
                for ev in trace:
                    if ev["concept:name"] in qualifying_activities:
                        cases_to_keep_full.add(case_id)
                        break  # this case is in, no need to check further

            # Filter anonymized_log: keep entire trace if its case is in cases_to_keep_full; otherwise drop all events
            traces_to_delete = []
            for ti, trace in enumerate(anonymized_log):
                case_id = trace.attributes["concept:name"]
                if case_id in cases_to_keep_full:
                    # keep entire trace
                    continue
                else:
                    # remove all events -> mark for deletion
                    traces_to_delete.append(ti)
                    edited_traces.add(ti)
            for ti in reversed(traces_to_delete):
                del anonymized_log._list[ti]

        """
        # Non-explicit: keep only events whose activity is qualifying
        traces_to_delete = []
        for ti, trace in enumerate(anonymized_log):
            case_id = trace.attributes["concept:name"]
            kept = []
            for ev in trace:
                if ev["concept:name"] in qualifying_activities:
                    kept.append(ev)
                else:
                    edited_traces.add(ti)
            trace._list = kept
            if not kept:
                traces_to_delete.append(ti)
        for ti in reversed(traces_to_delete):
            del anonymized_log._list[ti]

    elif mode == "ngram":
        if ngram_size <= 0:
            raise ValueError("ngram_size must be >=1 for ngram mode")

        # Step 1: collect for each trace its n-grams and which cases have which ngrams
        ngram_to_cases = defaultdict(set)
        trace_ngrams = []  # per trace list of (ngram, start, end)
        for trace in original_log:
            case_id = trace.attributes["concept:name"]
            activities = [ev["concept:name"] for ev in trace]
            ngrams = extract_ngrams_from_trace(activities, ngram_size)
            trace_ngrams.append(ngrams)
            for ngram, start, end in ngrams:
                ngram_to_cases[ngram].add(case_id)

        # Qualifying ngrams: appearing in >= z distinct cases
        qualifying_ngrams = {ng for ng, cases in ngram_to_cases.items() if len(cases) >= z}

        # Build release set: (case_id, position) tuples to keep
        events_to_release = set()

        #if explicit:
        # For each trace: if it contains any qualifying ngram, release all events composing those ngrams
        for trace_idx, trace in enumerate(original_log):
            case_id = trace.attributes["concept:name"]
            ngrams = trace_ngrams[trace_idx]
            for ngram, start, end in ngrams:
                if ngram in qualifying_ngrams:
                    for pos in range(start, end + 1):
                        events_to_release.add((case_id, pos))
                        ''' 
        else:
            # Non-explicit: release only the exact occurrences of qualifying ngrams in each trace (their constituent events)
            for trace_idx, trace in enumerate(original_log):
                case_id = trace.attributes["concept:name"]
                ngrams = trace_ngrams[trace_idx]
                for ngram, start, end in ngrams:
                    if ngram in qualifying_ngrams:
                        for pos in range(start, end + 1):
                            events_to_release.add((case_id, pos))
        '''

        # Apply filtering based on events_to_release
        traces_to_delete = set()
        for trace_idx, trace in enumerate(anonymized_log):
            case_id = trace.attributes["concept:name"]
            kept = []
            for pos, ev in enumerate(trace):
                if (case_id, pos) in events_to_release:
                    kept.append(ev)
                else:
                    edited_traces.add(trace_idx)
            trace._list = kept
            if not kept:
                traces_to_delete.add(trace_idx)
        for ti in sorted(traces_to_delete, reverse=True):
            del anonymized_log._list[ti]
    else:
        raise ValueError(f"Unknown mode '{mode}'; expected 'single' or 'ngram'")

    return anonymized_log, len(edited_traces)
