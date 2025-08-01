from collections import defaultdict
from typing import Dict, Set, Tuple, Optional, Any
from pm4py.objects.log.obj import EventLog


def get_alphabet(log: EventLog) -> Set[str]:
    """
    Extract the set of unique activity names from the event log.
    """
    alphabet = set()
    for trace in log:
        for event in trace:
            # Unified access: try dict-like first, fallback to event itself
            activity = None
            if hasattr(event, "get"):
                activity = event.get("concept:name")
            elif isinstance(event, dict):
                activity = event.get("concept:name")
            else:
                activity = event  # assume event is already a name
            if activity:
                alphabet.add(activity)
    return alphabet


def compute_eventual_follows_relations(
    log: EventLog, alphabet: Set[str]
) -> Dict[str, Dict[str, int]]:
    """
    Computes eventual follows relations with correct semantics:
      0 = never follows
      1 = sometimes follows
      2 = always follows

    The logic is: for each pair (a, b), consider all traces where a appears.
    If in all such traces b appears after the last a -> always (2).
    If in none -> never (0). Otherwise -> sometimes (1).
    """
    # Counters: for each (a,b), count in how many traces a appears and in how many of those b follows
    trace_counts = {
        a: {b: {"a_occurs": 0, "b_after": 0} for b in alphabet if b != a}
        for a in alphabet
    }

    for trace in log:
        trace_activities = [evt.get("concept:name") if hasattr(evt, "get") else evt for evt in trace]
        for a in alphabet:
            if a not in trace_activities:
                continue
            # last occurrence of a
            last_a_idx = len(trace_activities) - 1 - trace_activities[::-1].index(a)
            for b in alphabet:
                if a == b:
                    continue
                trace_counts[a][b]["a_occurs"] += 1
                # check if b appears after last a
                if any(trace_activities[i] == b for i in range(last_a_idx + 1, len(trace_activities))):
                    trace_counts[a][b]["b_after"] += 1

    result = {a: {} for a in alphabet}
    for a in alphabet:
        for b in alphabet:
            if a == b:
                continue
            a_occurs = trace_counts[a][b]["a_occurs"]
            b_after = trace_counts[a][b]["b_after"]
            if a_occurs == 0:
                # Semantically: if a never appears, we choose "never follows" for consistency
                relation = 0
            elif b_after == 0:
                relation = 0
            elif b_after == a_occurs:
                relation = 2
            else:
                relation = 1
            result[a][b] = relation
    return result


def calculate_f1(tp: int, fp: int, fn: int) -> float:
    """
    Standard F1: 2*TP / (2*TP + FP + FN)
    """
    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom > 0 else 0.0


def compare_eventual_follows_relations(
    original_log: EventLog, generated_log: EventLog
) -> Dict[str, Any]:
    """
    Compare eventual follows relations between original and generated logs.

    Returns a dictionary with:
      - per-class counts: support (original), TP, FP, FN
      - per-class precision/recall/F1
      - micro F1 / precision / recall
      - macro F1 / precision / recall
      - overall (micro) F1

    If generated_log is empty, returns a dict where all comparison metrics are None
    (except original support counts which are still computed).
    """
    original_alphabet = get_alphabet(original_log)
    original_rel = compute_eventual_follows_relations(original_log, original_alphabet)

    # Count original support per class
    support = {"never": 0, "sometimes": 0, "always": 0}
    for a in original_alphabet:
        for b in original_alphabet:
            if a == b:
                continue
            rel = original_rel[a][b]
            if rel == 0:
                support["never"] += 1
            elif rel == 1:
                support["sometimes"] += 1
            elif rel == 2:
                support["always"] += 1

    # If generated log is empty, per request return None for matching metrics
    if len(generated_log) == 0:
        return {
            "support": support,
            "per_class": {
                "never": {"tp": None, "fp": None, "fn": None, "precision": None, "recall": None, "f1": None},
                "sometimes": {"tp": None, "fp": None, "fn": None, "precision": None, "recall": None, "f1": None},
                "always": {"tp": None, "fp": None, "fn": None, "precision": None, "recall": None, "f1": None},
            },
            "micro": {"precision": None, "recall": None, "f1": None},
            "macro": {"precision": None, "recall": None, "f1": None},
        }

    generated_alphabet = get_alphabet(generated_log)
    generated_rel = compute_eventual_follows_relations(generated_log, generated_alphabet)

    # Initialize confusion components
    class_map = {0: "never", 1: "sometimes", 2: "always"}
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    # We treat missing activities in generated as relation=0 (never follows)
    for a in original_alphabet:
        for b in original_alphabet:
            if a == b:
                continue
            orig = original_rel[a][b]
            gen = (
                generated_rel[a][b]
                if (a in generated_alphabet and b in generated_alphabet)
                else 0
            )

            if orig == gen:
                tp[class_map[orig]] += 1
            else:
                # orig is true class, gen is predicted
                fn[class_map[orig]] += 1  # missed the true class
                fp[class_map[gen]] += 1  # wrongly predicted this class

    # Compute per-class precision/recall/f1
    per_class_metrics = {}
    for cls in ["never", "sometimes", "always"]:
        cls_tp = tp[cls]
        cls_fp = fp[cls]
        cls_fn = fn[cls]
        precision = cls_tp / (cls_tp + cls_fp) if (cls_tp + cls_fp) > 0 else 0.0
        recall = cls_tp / (cls_tp + cls_fn) if (cls_tp + cls_fn) > 0 else 0.0
        f1 = calculate_f1(cls_tp, cls_fp, cls_fn)
        per_class_metrics[cls] = {
            "tp": cls_tp,
            "fp": cls_fp,
            "fn": cls_fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    # Micro metrics: aggregate all
    total_tp = sum(tp.values())
    total_fp = sum(fp.values())
    total_fn = sum(fn.values())
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = calculate_f1(total_tp, total_fp, total_fn)

    # Macro metrics: average of per-class metrics
    precisions = [per_class_metrics[c]["precision"] for c in ["never", "sometimes", "always"]]
    recalls = [per_class_metrics[c]["recall"] for c in ["never", "sometimes", "always"]]
    f1s = [per_class_metrics[c]["f1"] for c in ["never", "sometimes", "always"]]
    macro_precision = sum(precisions) / 3
    macro_recall = sum(recalls) / 3
    macro_f1 = sum(f1s) / 3

    return {
        "support": support,
        "per_class": per_class_metrics,
        "micro": {"precision": micro_precision, "recall": micro_recall, "f1": micro_f1},
        "macro": {"precision": macro_precision, "recall": macro_recall, "f1": macro_f1},
    }
