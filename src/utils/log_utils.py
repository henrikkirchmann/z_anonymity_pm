import pm4py
from pm4py.objects.log.obj import EventLog
from typing import Dict, Set, Tuple
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from src.definitions import get_input_path

def load_event_log(file_path: str | Path) -> EventLog:
    """
    Load an event log from a file.
    
    Args:
        file_path: Path to the event log file (string or Path object)
        
    Returns:
        Loaded event log
    """
    # Convert to Path object if string
    path = Path(file_path) if isinstance(file_path, str) else file_path
    
    # Load the log using pm4py
    log = pm4py.read_xes(str(path))
    
    # Ensure the log is properly formatted
    if not isinstance(log, EventLog):
        log = pm4py.convert_to_event_log(log)
    
    return log

def extract_directly_follows_relations(log: EventLog, time_window: int) -> Dict[Tuple[str, str], Set[str]]:
    """
    Extract directly-follows relations from the event log with their case IDs.
    
    Args:
        log: Event log
        time_window: Maximum time difference (in seconds) between events to be considered as directly-follows
        
    Returns:
        Dictionary mapping (activity1, activity2) to set of case IDs
    """
    dfr = {}  # (act1, act2) -> set of case IDs
    
    for trace in log:
        case_id = trace.attributes['concept:name']
        events = [(event['time:timestamp'], event['concept:name']) for event in trace]
        events.sort(key=lambda x: x[0])
        
        for i in range(len(events) - 1):
            curr_event = events[i]
            next_event = events[i + 1]
            
            # Check if events are within time window
            time_diff = (next_event[0] - curr_event[0]).total_seconds()
            if time_diff <= time_window:
                relation = (curr_event[1], next_event[1])
                if relation not in dfr:
                    dfr[relation] = set()
                dfr[relation].add(case_id)
                
    return dfr

def get_relation_types(log: EventLog) -> Tuple[Set[Tuple[str, str]], Set[Tuple[str, str]], Set[Tuple[str, str]]]:
    """
    Identify always-follows, sometimes-follows, and never-follows relations.
    
    Args:
        log: Event log
        
    Returns:
        Tuple of (always_follows, sometimes_follows, never_follows) relations
    """
    # Extract all activities
    activities = set()
    for trace in log:
        for event in trace:
            activities.add(event['concept:name'])
    
    # Get all possible activity pairs
    all_pairs = set((a1, a2) for a1 in activities for a2 in activities if a1 != a2)
    
    # Get actual relations from log
    case_relations = {}
    for trace in log:
        case_id = trace.attributes['concept:name']
        events = [event['concept:name'] for event in trace]
        case_relations[case_id] = set()
        
        for i in range(len(events) - 1):
            case_relations[case_id].add((events[i], events[i + 1]))
    
    # Classify relations
    always_follows = set()
    sometimes_follows = set()
    never_follows = set()
    
    for pair in all_pairs:
        # Count in how many cases this relation appears
        cases_with_relation = sum(1 for relations in case_relations.values() if pair in relations)
        cases_with_first = sum(1 for trace in log if any(event['concept:name'] == pair[0] for event in trace))
        
        if cases_with_relation == cases_with_first and cases_with_relation > 0:
            always_follows.add(pair)
        elif cases_with_relation > 0:
            sometimes_follows.add(pair)
        else:
            never_follows.add(pair)
            
    return always_follows, sometimes_follows, never_follows 