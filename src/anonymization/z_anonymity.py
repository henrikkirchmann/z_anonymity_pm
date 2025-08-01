import copy
from collections import defaultdict
from datetime import timedelta
from typing import Dict, List

from pm4py.objects.log.obj import EventLog


def apply_z_anonymity(log: EventLog, z_value: int, time_window: int, explicit: bool, source_attribute: str) -> EventLog:
    """
    Apply explicit z-anonymity to the event log.
    In explicit mode, when z different cases have the same activity in a time window,
    we release ALL events from ALL cases that helped reach the threshold,
    not just the z-th event.
    
    Args:
        log: Original event log
        z_value: Minimum number of different cases required to release activities
        time_window: Time window in seconds to look for similar activities
        explicit: If true, explicit z-anonymity will be applied.

    Returns:
        Anonymized event log
    """
    # Create a deep copy of the log to modify
    anonymized_log = copy.deepcopy(log)

    # 1) Build a list of all events with a unique index
    all_events = []
    for trace in log:
        case = trace.attributes['concept:name']
        for event in trace:
            all_events.append((case,
                               event['time:timestamp'],
                               event['concept:name'],
                               event))
    # sort by timestamp
    all_events.sort(key=lambda x: x[1])

    # Now re-enumerate to get unique idx
    indexed_events = [
        (idx, case, ts, act, ev)
        for idx, (case, ts, act, ev) in enumerate(all_events)
    ]

    event_source_dict: defaultdict[str, List] = defaultdict(list)

    for idx, case, ts, act, ev in indexed_events:
        src = ev[source_attribute]
        event_source_dict[src].append((idx, case, ts, act, ev))

    # 2) Prepare release set of idx values
    events_to_release = set()

    # 3) Slide over indexed_events in temporal order
    for source_index_events in event_source_dict.values():
        for curr_pos, (curr_idx, curr_case, curr_time, curr_act, curr_ev) in enumerate(source_index_events):
            window_start = curr_time - timedelta(seconds=time_window)

            # backward look only
            cases_with_act = set()
            window_idxs = []
            # scan earlier events within window
            for rev_i, (idx, prev_case, prev_time, prev_act, prev_ev) in enumerate(
                    reversed(source_index_events[:curr_pos])):
                # rev_i == 0 corresponds to the event just *before* curr_pos,
                # rev_i == 1 is the one before that, etc.

                # if we’ve slid past the window, stop entirely
                if prev_time < window_start:
                    break

                if prev_act == curr_act:
                    cases_with_act.add(prev_case)
                    window_idxs.append(idx)

            # include current
            cases_with_act.add(curr_case)
            window_idxs.append(curr_idx)

            if len(cases_with_act) >= z_value:
                if explicit:
                    # mark all window events by idx
                    events_to_release.update(window_idxs)
                else:
                    events_to_release.add(curr_idx)

    # 4) Rebuild anonymized_log by filtering on idx membership
    # We'll map each original event object to its idx
    ev_to_idx = {ev: idx for idx, *_rest in indexed_events for ev in [indexed_events[idx][4]]}

    edited_traces = set()
    traces_to_delete = []
    for ti, trace in enumerate(anonymized_log):
        kept = []
        for ev in trace:
            idx = ev_to_idx.get(ev)
            if idx in events_to_release:
                kept.append(ev)
            else:
                edited_traces.add(ti)

        trace._list = kept
        if not kept:
            traces_to_delete.append(ti)

    # remove empty traces in reverse order
    for ti in reversed(traces_to_delete):
        del anonymized_log._list[ti]

    return anonymized_log, len(edited_traces)


def get_anonymization_stats(original_log: EventLog, anonymized_log: EventLog, time_window: int) -> Dict:
    """
    Calculate statistics about the anonymization process.
    
    Args:
        original_log: Original event log
        anonymized_log: Anonymized event log
        time_window: Time window used for z-anonymity
        
    Returns:
        Dictionary containing various statistics about the anonymization
    """
    # Count events and traces
    original_events = sum(len(trace) for trace in original_log)
    original_traces = len(original_log)
    anonymized_events = sum(len(trace) for trace in anonymized_log)
    anonymized_traces = len(anonymized_log)

    # Count unique attributes
    original_attributes = set()
    anonymized_attributes = set()

    for trace in original_log:
        for event in trace:
            original_attributes.add(event['concept:name'])

    for trace in anonymized_log:
        for event in trace:
            anonymized_attributes.add(event['concept:name'])

    stats = {
        'original_unique_attributes': len(original_attributes),
        'anonymized_unique_attributes': len(anonymized_attributes),
        'total_events': original_events,
        'remaining_events': anonymized_events,
        'removed_events': original_events - anonymized_events,
        'total_traces': original_traces,
        'remaining_traces': anonymized_traces,
        'removed_traces': original_traces - anonymized_traces,
        'event_removal_rate': (original_events - anonymized_events) / original_events if original_events > 0 else 0,
        'trace_removal_rate': (original_traces - anonymized_traces) / original_traces if original_traces > 0 else 0
    }

    return stats
