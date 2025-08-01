import copy
from datetime import timedelta
from typing import List
from collections import defaultdict
from typing import Dict, List

from pm4py.objects.log.obj import EventLog


def extract_ngrams(trace: List[str], n: int) -> List[str]:
    """
    Extract n-grams from a sequence of activities.
    Only returns complete n-grams (exactly n events).
    
    Args:
        trace: List of activity names
        n: Size of n-grams
        
    Returns:
        List of n-grams (each n-gram is a string of activities joined by '>')
    """
    if len(trace) < n:
        return []  # Return empty list if trace is shorter than n

    ngrams = []
    for i in range(len(trace) - n + 1):
        ngram = trace[i:i + n]
        ngrams.append('>'.join(ngram))
    return ngrams


def apply_ngram_z_anonymity(log: EventLog, z_value: int, time_window: int, n: int, explicit: bool, source_attribute: str) -> EventLog:
    """
    Apply  z-anonymity to n-grams of activities in the event log.

    An n-gram is considered sensitive if it appears for fewer than z different cases
    within the time window. Only complete n-grams (exactly n events) are considered.
    The timestamp of the n-gram is taken from its last event.

    In explicit mode, when z different cases have the same n-gram in a time window,
    we release ALL events from ALL n-grams that helped reach the threshold,
    not just the z-th n-gram.

    Events that are part of n-grams that will be released (appear in z or more cases)
    are protected from deletion, even if they are also part of sensitive n-grams.
    
    Args:
        log: Original event log
        z_value: Minimum number of different cases required to release n-grams
        time_window: Time window in seconds
        n: Size of n-grams to protect
        explicit: If true, explicit z-anonymity will be applied.
        
    Returns:
        Anonymized event log
    """
    # Create a deep copy of the log to modify
    anonymized_log = copy.deepcopy(log)

    # Collect all events with their n-grams
    all_events = []
    for trace in log:
        case_id = trace.attributes['concept:name']
        # Extract activities for n-gram generation
        activities = [event['concept:name'] for event in trace]
        timestamps = [event['time:timestamp'] for event in trace]

        # Only process traces that are long enough for at least one complete n-gram
        if len(activities) >= n:
            # Generate n-grams for each event position
            for i, event in enumerate(trace):
                # Only process positions that can start a complete n-gram
                if i <= len(activities) - n:
                    ngram = '>'.join(activities[i:i + n])
                    # Use timestamp of last event in n-gram
                    ngram_timestamp = timestamps[i + n - 1]

                    all_events.append((
                        ngram_timestamp,  # Using last event's timestamp
                        case_id,
                        ngram,
                        event,
                        i  # Store position in trace for later deletion
                    ))

    # Sort events by timestamp
    all_events.sort(key=lambda x: x[0])

    event_source_dict: defaultdict[str, List] = defaultdict(list)

    for ngram_timestamp, case_id, ngram, event, i in all_events:
        src = event[source_attribute]
        event_source_dict[src].append((ngram_timestamp, case_id, ngram, event, i))


    # Track events to release (only those that helped reach the threshold)
    events_to_release = set()  # Set of (case_id, position) tuples to release

    # 3) Slide over indexed_events in temporal order
    for source_index_events in event_source_dict.values():
        #for curr_pos, (curr_idx, curr_case, curr_time, curr_act, curr_ev) in enumerate(source_index_events):
        # Process events in temporal order
        for curr_idx, (curr_time, curr_case, curr_ngram, curr_event, curr_pos) in enumerate(source_index_events):
            window_start = curr_time - timedelta(seconds=time_window)

            # Find all cases that have this n-gram in the time window (backward look only)
            cases_with_ngram = set()
            ngram_events = []  # Track all events from n-grams that have this n-gram in the window

            # Look backward in time window only
            for rev_i, (prev_time, prev_case, prev_ngram, prev_event, prev_pos) in enumerate(
                    reversed(source_index_events[:curr_idx])):
                # rev_i == 0 corresponds to the event just *before* curr_pos,
                # rev_i == 1 is the one before that, etc.
                if prev_time < window_start:
                    break
                if prev_ngram == curr_ngram:
                    cases_with_ngram.add(prev_case)
                    # Add all events from this n-gram
                    for offset in range(n):
                        ngram_events.append((prev_case, prev_pos + offset))

            # Add current case and its n-gram events
            cases_with_ngram.add(curr_case)
            for offset in range(n):
                ngram_events.append((curr_case, curr_pos + offset))

            # If we have z or more different cases with this n-gram, release ALL events from ALL n-grams that helped reach the threshold
            if len(cases_with_ngram) >= z_value:
                if not explicit:
                    ngram_events = []
                    for offset in range(n):
                        ngram_events.append((curr_case, curr_pos + offset))
                events_to_release.update(ngram_events)


    edited_traces = set()
    # Delete events that are NOT in the release set
    traces_to_delete = set()
    for trace_idx, trace in enumerate(anonymized_log):
        case_id = trace.attributes['concept:name']
        events_to_keep = []

        for pos, event in enumerate(trace):
            if (case_id, pos) in events_to_release:
                events_to_keep.append(event)  # Keep events that are in release set
            else:
                edited_traces.add(trace_idx)


        trace._list = events_to_keep

        if not events_to_keep:
            traces_to_delete.add(trace_idx)

    # Remove empty traces (in reverse order to maintain correct indices)
    for trace_idx in sorted(traces_to_delete, reverse=True):
        del anonymized_log._list[trace_idx]

    return anonymized_log, len(edited_traces)
