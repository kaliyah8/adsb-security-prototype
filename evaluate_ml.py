from __future__ import annotations

from ml_evaluation import run_evaluation


def main() -> None:
    result = run_evaluation(runs_per_scenario=5, base_seed=100, save=True)
    metrics = result["metrics"]
    seq_metrics = metrics["sequence_metrics"]
    row_metrics = metrics["row_metrics"]

    print("=== ML EVALUATION COMPLETE ===")
    print(f"Runs per scenario: {metrics['runs_per_scenario']}")
    print(f"Sequence precision: {seq_metrics.get('precision', float('nan')):.4f}")
    print(f"Sequence recall: {seq_metrics.get('recall', float('nan')):.4f}")
    print(f"Sequence F1: {seq_metrics.get('f1', float('nan')):.4f}")
    print(f"Sequence ROC-AUC: {seq_metrics.get('roc_auc', float('nan')):.4f}")
    print(f"Sequence PR-AUC: {seq_metrics.get('pr_auc', float('nan')):.4f}")
    print(f"Sequence false-positive rate: {seq_metrics.get('false_positive_rate', float('nan')):.4f}")
    print(f"Row recall: {row_metrics.get('recall', float('nan')):.4f}")
    print("Artifacts written to artifacts/evaluation_metrics.json and companion CSV files.")


if __name__ == "__main__":
    main()
