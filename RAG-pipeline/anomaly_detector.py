import os
import glob
import pandas as pd

DATA_DIR = "data/raw"

# Metrics most indicative of real RAN problems when they deviate significantly
KEY_METRICS = ["dl_bler", "ul_bler", "dl_snr", "rsrp"]


def load_sample_csvs(folder_name, data_dir=DATA_DIR, max_files=30):
    """
    Loads a SAMPLE of CSVs, not all 13k+/31k+ files -- there's no need to
    scan the entire corpus for a demo, and it would take far too long.
    """
    base_path = os.path.join(data_dir, folder_name)
    csv_paths = glob.glob(os.path.join(base_path, "**", "*.csv"), recursive=True)[:max_files]

    frames = []
    for path in csv_paths:
        df = pd.read_csv(path)
        df["source_file"] = os.path.relpath(path, data_dir)
        frames.append(df)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def detect_anomalies(df, z_threshold=3.0):
    """
    Flags rows where a key metric deviates more than z_threshold standard
    deviations from that metric's mean across the loaded sample. Simple
    statistical detection -- not a trained model, but fully explainable:
    every flag comes with the exact metric, value, and z-score that
    triggered it.
    """
    anomalies = []

    for metric in KEY_METRICS:
        if metric not in df.columns:
            continue
        mean = df[metric].mean()
        std = df[metric].std()
        if std == 0 or pd.isna(std):
            continue

        z_scores = (df[metric] - mean) / std
        mask = z_scores.abs() > z_threshold
        flagged = df[mask].copy()
        flagged["anomaly_metric"] = metric
        flagged["z_score"] = z_scores[mask]
        anomalies.append(flagged)

    if not anomalies:
        return pd.DataFrame()

    return pd.concat(anomalies, ignore_index=True)


def describe_anomaly(row):
    """Plain-language description of one flagged row -- this is what gets
    fed into the RAG system (see rca.py) for root cause explanation."""
    metric = row["anomaly_metric"]
    return (
        f"Anomaly detected in {metric} (value={row[metric]:.2f}, "
        f"z-score={row['z_score']:.2f}) in {row['source_file']} "
        f"at time={row.get('time', 'unknown')}."
    )


if __name__ == "__main__":
    print("Loading a sample of slice_mixed telemetry...")
    df = load_sample_csvs("slice_mixed", max_files=30)
    print(f"Loaded {len(df)} rows from {df['source_file'].nunique()} files")

    anomalies = detect_anomalies(df)
    print(f"\nFound {len(anomalies)} anomalous readings (|z| > 3.0)\n")

    for _, row in anomalies.head(5).iterrows():
        print("-", describe_anomaly(row))


METRIC_NAMES = {
    "dl_bler": "downlink block error rate (DL BLER)",
    "ul_bler": "uplink block error rate (UL BLER)",
    "dl_snr": "downlink signal-to-noise ratio (SNR)",
    "rsrp": "reference signal received power (RSRP)",
}


def anomaly_to_query(row):
    """
    Converts a flagged anomaly into a natural-language question suitable for
    semantic retrieval. The raw technical description (file paths, z-scores,
    experiment IDs) is useful for logging/display, but it's mostly noise for
    embedding-based search -- none of that vocabulary means anything in the
    3GPP knowledge base, so it was drowning out the one meaningful term.
    """
    metric = row["anomaly_metric"]
    readable_metric = METRIC_NAMES.get(metric, metric)
    value = row[metric]

    if "bler" in metric and value > 50:
        severity = "extremely high"
    elif metric == "rsrp" and value < -100:
        severity = "extremely low"
    else:
        severity = "abnormal"

    return f"What causes a {severity} {readable_metric} in a 5G RAN cell, and what does it indicate?"