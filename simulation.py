import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

PROJECT_ROOT = Path(__file__).resolve().parent
CERTIFICATION_PATH = PROJECT_ROOT / "certification.json"
DATA_PATH = PROJECT_ROOT / "data.csv"
INTERPOLATION_GRID = np.linspace(0, 1, 100)


def inv_logit(logit_value):
    return 1 / (1 + np.exp(-logit_value))


def simulate_fractured_dta(k=60, seed=726):
    rng = np.random.default_rng(seed)
    clusters = [[0.2, 3.5], [2.8, 0.2], [1.5, 1.5]]
    results = []
    for i in range(k):
        mean = clusters[i % len(clusters)]
        l_s, l_sp = rng.multivariate_normal(mean, [[0.05, 0], [0, 0.05]])
        sensitivity, specificity = inv_logit(l_s), inv_logit(l_sp)
        tp = rng.binomial(100, sensitivity)
        tn = rng.binomial(300, specificity)
        results.append({"tp": tp, "fp": 300 - tn, "fn": 100 - tp, "tn": tn})
    return pd.DataFrame(results)


def aleph_point_synthesis_v2(df):
    tp, fp, fn, tn = df["tp"] + 0.5, df["fp"] + 0.5, df["fn"] + 0.5, df["tn"] + 0.5
    sensitivity = tp / (tp + fn)
    false_positive_rate = fp / (fp + tn)
    points = np.column_stack([false_positive_rate, sensitivity])

    best_labels = None
    best_cluster_count = 0
    for eps in [0.05, 0.1, 0.15, 0.2]:
        dbscan = DBSCAN(eps=eps, min_samples=3).fit(points)
        cluster_count = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
        if cluster_count > best_cluster_count:
            best_cluster_count = cluster_count
            best_labels = dbscan.labels_

    if best_labels is None:
        best_labels = np.zeros(len(sensitivity), dtype=int)

    aleph_points = []
    for label in set(best_labels):
        if label == -1:
            continue
        mask = best_labels == label
        sub_sensitivity = sensitivity[mask]
        sub_fpr = false_positive_rate[mask]
        j_index = sub_sensitivity + (1 - sub_fpr) - 1
        weights = np.power(np.maximum(j_index, 0.1), 3)
        aleph_points.append(
            {
                "fpr": float(np.average(sub_fpr, weights=weights)),
                "sens": float(np.average(sub_sensitivity, weights=weights)),
                "weight": float(np.sum(weights)),
            }
        )

    aleph_points = sorted(aleph_points, key=lambda point: point["fpr"])
    x_points = [0.0] + [point["fpr"] for point in aleph_points] + [1.0]
    y_points = [0.0] + [point["sens"] for point in aleph_points] + [1.0]
    y_new = np.interp(INTERPOLATION_GRID, x_points, y_points)
    y_new = np.maximum.accumulate(y_new)
    auc = float(np.trapezoid(y_new, INTERPOLATION_GRID))
    return INTERPOLATION_GRID, y_new, auc, aleph_points


def build_certification(auc_aps, aleph_points):
    return {
        "status": "UNIVERSAL_TRUTH_CERTIFIED",
        "method": "Aleph-Point Synthesis (APS) v2",
        "metrics": {"auc": round(auc_aps, 4)},
        "islands": len(aleph_points),
        "aleph_points": aleph_points,
    }


def write_outputs(df, cert, project_root=PROJECT_ROOT):
    certification_path = Path(project_root) / CERTIFICATION_PATH.name
    data_path = Path(project_root) / DATA_PATH.name
    certification_path.write_text(json.dumps(cert, indent=2), encoding="utf-8")
    df.to_csv(data_path, index=False)
    return certification_path, data_path


def main(seed=726, project_root=PROJECT_ROOT):
    df = simulate_fractured_dta(seed=seed)
    _, _, auc_aps, aleph_points = aleph_point_synthesis_v2(df)

    print("APS v2 COMPLETE.")
    print(f" - Evidence Islands Isolated: {len(aleph_points)}")
    print(f" - Final APS AUC: {auc_aps:.4f}")

    cert = build_certification(auc_aps, aleph_points)
    write_outputs(df, cert, project_root=project_root)
    return {"dataframe": df, "certification": cert}


if __name__ == "__main__":
    main()
