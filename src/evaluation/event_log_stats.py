def get_event_log_stats(original_log, anonymized_log, number_of_edited_traces):
    """Get statistics about the anonymization process."""
    total_events = sum(len(trace) for trace in original_log)
    remaining_events = sum(len(trace) for trace in anonymized_log)
    event_removal_rate = (remaining_events) / total_events if total_events > 0 else 0
    trace_removal_rate = len(anonymized_log) / len(original_log) if len(original_log) > 0 else 0
    trace_edited_rate =  (len(original_log) - number_of_edited_traces) / len(original_log) if len(original_log) > 0 else 0
    return {
        'original_number_of_events': total_events,
        'anom_number_of_remaining_events': remaining_events,
        'event_removal_rate': event_removal_rate, #1 = no events deleted, 0 = all events deleted
        'trace_removal_rate': trace_removal_rate, #1 = no traces deleted, 0 = all traces deleted
        'trace_affected_rate': trace_edited_rate,  #1 = no traces edited,0 = for all events at least one events are deleted
    }

""" 
def get_ngram_anonymization_stats(original_log: EventLog, anonymized_log: EventLog, time_window: int, n: int) -> dict:
    
    Calculate statistics about the n-gram anonymization.
    Only considers complete n-grams (exactly n events).

    Args:
        original_log: Original event log
        anonymized_log: Anonymized event log
        time_window: Time window used for anonymization (in seconds)
        n: Size of n-grams that were protected

    Returns:
        Dictionary with statistics
    
    # Count original events and traces
    total_events = sum(len(trace) for trace in original_log)
    total_traces = len(original_log)

    # Count remaining events and traces
    remaining_events = sum(len(trace) for trace in anonymized_log)
    remaining_traces = len(anonymized_log)

    # Calculate n-gram statistics (only complete n-grams)
    original_ngrams = set()
    for trace in original_log:
        activities = [event['concept:name'] for event in trace]
        trace_ngrams = extract_ngrams(activities, n)  # Now only returns complete n-grams
        original_ngrams.update(trace_ngrams)

    remaining_ngrams = set()
    for trace in anonymized_log:
        activities = [event['concept:name'] for event in trace]
        trace_ngrams = extract_ngrams(activities, n)  # Now only returns complete n-grams
        remaining_ngrams.update(trace_ngrams)

    return {
        'total_events': total_events,
        'remaining_events': remaining_events,
        'event_removal_rate': (total_events - remaining_events) / total_events,
        'total_traces': total_traces,
        'remaining_traces': remaining_traces,
        'trace_removal_rate': (total_traces - remaining_traces) / total_traces,
        'total_ngrams': len(original_ngrams),
        'remaining_ngrams': len(remaining_ngrams),
        'ngram_removal_rate': (len(original_ngrams) - len(remaining_ngrams)) / len(
            original_ngrams) if original_ngrams else 0,
        'ngram_size': n,
        'time_window_seconds': time_window
    }
"""