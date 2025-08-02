import copy
from datetime import timedelta
from collections import defaultdict
from typing import List, Set, Tuple

from pm4py.objects.log.obj import EventLog


def extract_ngrams(trace: List[str], n: int) -> List[str]:
    """
    (Utility, retained for backwards-compatibility; not used in core logic below.)
    Extract n-grams from a sequence of activities.
    Only returns complete n-grams (exactly n events).
    """
    if len(trace) < n:
        return []
    return ['>'.join(trace[i:i + n]) for i in range(len(trace) - n + 1)]


def apply_ngram_z_anonymity(log: EventLog,
                            z_value: int,
                            time_window: int,
                            n: int,
                            explicit: bool,
                            source_attribute: str) -> Tuple[EventLog, int]:
    """
    Apply z-anonymity to n-grams of activities in the event log,
    where n-grams are built per source per case (i.e., all events in an n-gram
    share the same source_attribute value and case).

    Returns:
        (anonymized_log, number_of_edited_traces)
    """
    anonymized_log = copy.deepcopy(log)

    # Collect all source-local n-grams across the log.
    # Each record: (ngram_timestamp, case_id, ngram, source, start_pos, list_of_positions)
    all_source_ngrams = []

    for trace in log:
        case_id = trace.attributes['concept:name']
        length = len(trace)

        # Precompute source of each event to allow checking uniformity quickly
        sources = [event[source_attribute] for event in trace]
        activities = [event['concept:name'] for event in trace]
        timestamps = [event['time:timestamp'] for event in trace]

        for i in range(0, length - n + 1):
            # Check that the n events all have the same source
            window_sources = sources[i:i + n]
            if len(set(window_sources)) != 1:
                continue  # skip n-gram spanning multiple sources

            src = window_sources[0]
            ngram_list = activities[i:i + n]
            ngram = '>'.join(ngram_list)
            ngram_timestamp = timestamps[i + n - 1]  # timestamp of last event in n-gram
            positions = list(range(i, i + n))  # actual positions in trace

            all_source_ngrams.append((
                ngram_timestamp,
                case_id,
                ngram,
                src,
                i,           # start position
                tuple(positions)  # tuple of positions making up this n-gram
            ))

    # Group by source, then sort each source's n-grams chronologically
    source_to_ngrams: defaultdict[str, List] = defaultdict(list)
    for record in all_source_ngrams:
        ngram_timestamp, case_id, ngram, src, start_pos, positions = record
        source_to_ngrams[src].append(record)

    for src in source_to_ngrams:
        source_to_ngrams[src].sort(key=lambda x: x[0])  # sort by timestamp

    # Track events that are safe (to release / keep)
    events_to_release: Set[Tuple[str, int]] = set()

    # Slide window per source
    for src, ngram_records in source_to_ngrams.items():
        # For each current n-gram, look backwards within time window for same n-gram
        for curr_idx, (curr_time, curr_case, curr_ngram, _, curr_start_pos, curr_positions) in enumerate(ngram_records):
            window_start = curr_time - timedelta(seconds=time_window)

            cases_with_ngram = set()
            contributing_positions: Set[Tuple[str, int]] = set()

            # include current
            cases_with_ngram.add(curr_case)
            for pos in curr_positions:
                contributing_positions.add((curr_case, pos))

            # Look backward
            for prev_time, prev_case, prev_ngram, _, prev_start_pos, prev_positions in reversed(ngram_records[:curr_idx]):
                if prev_time < window_start:
                    break
                if prev_ngram == curr_ngram:
                    cases_with_ngram.add(prev_case)
                    for pos in prev_positions:
                        contributing_positions.add((prev_case, pos))

            if len(cases_with_ngram) >= z_value:
                if explicit:
                    # release all events from all contributing n-grams
                    events_to_release.update(contributing_positions)
                else:
                    # only release the current n-gram's events
                    for pos in curr_positions:
                        events_to_release.add((curr_case, pos))

    edited_traces = set()
    traces_to_delete = set()

    # Apply deletions: keep only events that are marked to release
    for trace_idx, trace in enumerate(anonymized_log):
        case_id = trace.attributes['concept:name']
        kept_events = []
        for pos, event in enumerate(trace):
            if (case_id, pos) in events_to_release:
                kept_events.append(event)
            else:
                edited_traces.add(trace_idx)

        trace._list = kept_events
        if not kept_events:
            traces_to_delete.add(trace_idx)

    # Remove empty traces in reverse order to preserve indices
    for trace_idx in sorted(traces_to_delete, reverse=True):
        del anonymized_log._list[trace_idx]

    return anonymized_log, len(edited_traces)
