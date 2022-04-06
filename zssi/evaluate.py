import pathlib

import jsonlines as jsonl
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay,
                             RocCurveDisplay, classification_report,
                             roc_auc_score)


def evaluate(similarities_file: str) -> None:
    similarities_file = pathlib.Path(similarities_file)
    y_true = []
    y_score = []
    with jsonl.open(similarities_file, "r") as f:
        for doc in f:
            doc_true_fields = list(filter(lambda x: x.endswith("_true")), doc.keys()).sort()
            doc_score_fields = list(filter(lambda x: x.endswith("_sim_max")), doc.keys()).sort()
            doc_true = list(map(lambda field: doc[field], doc_true_fields))
            doc_score = list(map(lambda field: doc[field], doc_score_fields))
            y_true.extend(doc_true)
            y_score.extend(doc_score)

    print(f"\nEvaluation for: {similarities_file}")
    print(f"roc_auc_score: {roc_auc_score(y_true, y_score, average='macro')}")
    RocCurveDisplay.from_predictions(y_true, y_score)
    PrecisionRecallDisplay.from_predictions(y_true, y_score)
    CalibrationDisplay.from_predictions(y_true, y_score)

    threshold = find_optimal_threshold(y_true, y_score)
    y_pred = list(map(lambda x: 0 if x <= threshold else 1, y_score))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    classification_report(y_true, y_pred)


def find_optimal_threshold(y_true, y_score):
    pass
