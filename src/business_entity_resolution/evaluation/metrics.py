"""
THE single source of truth for the competition metric. Every experiment, notebook,
and model comparison must import and call these functions - never reimplement inline,
and never substitute sklearn's fbeta_score / plain F1 / pairwise AUC as the basis for
a decision. See docs/architecture.md §14 and CLAUDE.md §3.
"""
from __future__ import annotations


def entity_f_beta(true_set: set, pred_set: set, beta: float = 0.5) -> float:
    """F_beta for a single Source-1 entity.

    Matches the official formula exactly (beta=0.5 => beta^2=0.25):
        F_0.5 = (1.25 * P * R) / (0.25 * P + R)

    Special cases, matching the problem statement:
      - true empty, pred empty  -> 1.0  (correct singleton)
      - true non-empty, pred empty -> 0.0 (missed everything)
      - true empty, pred non-empty -> 0.0 (false merge on a true singleton)
    """
    if not true_set and not pred_set:
        return 1.0
    if not pred_set:
        return 0.0

    tp = len(true_set & pred_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(true_set) if true_set else 0.0

    if precision == 0.0 and recall == 0.0:
        return 0.0

    beta2 = beta ** 2
    denom = beta2 * precision + recall
    if denom == 0:
        return 0.0
    return (1 + beta2) * precision * recall / denom


def macro_f_beta(all_true: dict, all_pred: dict, beta: float = 0.5) -> float:
    """all_true / all_pred: {source1_entity_id: set(matched_entity_ids)}.
    Macro-average over every S1 entity in all_true (all_pred.get(sid, set()) so a
    missing prediction is scored as an empty prediction, not skipped)."""
    if not all_true:
        raise ValueError("all_true must contain at least one Source-1 entity")
    scores = [entity_f_beta(all_true[sid], all_pred.get(sid, set()), beta) for sid in all_true]
    return sum(scores) / len(scores)


def precision_recall_f_beta_breakdown(all_true: dict, all_pred: dict, beta: float = 0.5) -> dict:
    """Extra diagnostics alongside the headline macro F_beta - singleton accuracy,
    false-merge rate, and average predicted matches per entity (architecture.md §11/§12)."""
    scores = []
    singleton_correct, singleton_total = 0, 0
    false_merge_count = 0
    pred_sizes = []

    for sid, true_set in all_true.items():
        pred_set = all_pred.get(sid, set())
        scores.append(entity_f_beta(true_set, pred_set, beta))
        pred_sizes.append(len(pred_set))
        if not true_set:
            singleton_total += 1
            if not pred_set:
                singleton_correct += 1
            else:
                false_merge_count += 1

    return {
        "macro_f_beta": sum(scores) / len(scores) if scores else 0.0,
        "singleton_total": singleton_total,
        "singleton_accuracy": (singleton_correct / singleton_total) if singleton_total else None,
        "false_merge_count": false_merge_count,
        "false_merge_rate": (false_merge_count / singleton_total) if singleton_total else None,
        "avg_predicted_matches_per_entity": sum(pred_sizes) / len(pred_sizes) if pred_sizes else 0.0,
    }


def parse_match_list(cell: str) -> set:
    """Parses a matching_results.tsv / ground_truth.tsv `matched_entity_ids` cell
    ("" or None -> empty set; "S2-0001,S3-0004" -> {"S2-0001", "S3-0004"})."""
    if cell is None or (isinstance(cell, float)) or str(cell).strip() == "":
        return set()
    return set(x.strip() for x in str(cell).split(",") if x.strip())
