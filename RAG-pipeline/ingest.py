import json
import os
import glob
import pandas as pd

DATA_DIR = "data/raw"


# ---------- TeleQnA ----------

def load_teleqna(data_dir=DATA_DIR, filename="TeleQnA.txt"):
    """
    Loads TeleQnA (despite the .txt extension, it's JSON-formatted) and
    turns each question into a single text block ready for chunking/embedding.
    This is your actual RAG knowledge base.
    """
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Couldn't find {path}")

    print(f"Loading TeleQnA from: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    entries = []
    for qid, item in data.items():
        question = item.get("question", "")
        answer = item.get("answer", "")
        explanation = item.get("explanation", "")
        category = item.get("category", "")

        # number of options varies per question
        options = [f"{k}: {v}" for k, v in item.items() if k.startswith("option")]
        options_text = "\n".join(options)

        text = (
            f"Question: {question}\n"
            f"{options_text}\n"
            f"Correct Answer: {answer}\n"
            f"Explanation: {explanation}"
        )

        entries.append({
            "id": qid,
            "text": text,
            "source": f"TeleQnA/{category}" if category else "TeleQnA",
        })

    print(f"Loaded {len(entries)} TeleQnA entries")
    return entries


# ---------- O-RAN telemetry (slice_mixed / slice_traffic) ----------

def summarize_oran_csv(folder_name, data_dir=DATA_DIR):
    """
    Walks slice_mixed/slice_traffic and reports what's there. This telemetry
    (rsrp, dl_brate, dl_bler, etc.) is for the anomaly-detection module later,
    loaded directly with pandas -- not turned into RAG text chunks, since one
    row of telemetry isn't a meaningful retrievable "fact".
    """
    base_path = os.path.join(data_dir, folder_name)
    if not os.path.isdir(base_path):
        print(f"Folder not found, skipping: {base_path}")
        return None

    csv_paths = glob.glob(os.path.join(base_path, "**", "*.csv"), recursive=True)
    print(f"\n{folder_name}: found {len(csv_paths)} CSV files")

    if not csv_paths:
        return None

    sample_df = pd.read_csv(csv_paths[0])
    print(f"Example file: {csv_paths[0]}")
    print(f"Columns: {list(sample_df.columns)}")
    print(f"Rows in that file: {len(sample_df)}")

    return csv_paths


# ---------- Zenodo network/fault data ----------

# Source columns are in Chinese -- renamed here to English for readability
# in code and in any demo output.
NETWORK_ELEMENT_COLUMNS = {
    "网元名称": "ne_name",
    "网元别名": "ne_alias",
    "网元型号": "ne_model",
    "网元类型": "ne_type",
    "设备IP": "device_ip",
    "所属站点": "site",
    "设备厂家": "vendor",
    "影响业务": "affected_service",
    "容量": "capacity",
    "网络层级": "network_level",
}

PERFORMANCE_METRIC_COLUMNS = {
    "设备名称": "device_name",
    "指标名称": "metric_name",
    "指标值": "metric_value",
    "时间戳": "timestamp",
}


def load_zenodo_fault_data(data_dir=DATA_DIR, folder_name="7003755"):
    """
    Loads the Zenodo network-element and performance-metric Excel files.
    Like the O-RAN CSVs, this is structured data for direct analysis
    (RCA module) rather than raw text for the vector store -- partly
    because embedding Chinese text with an English embedding model
    would produce meaningless results.
    """
    base_path = os.path.join(data_dir, folder_name)
    if not os.path.isdir(base_path):
        print(f"Folder not found, skipping: {base_path}")
        return None, None

    ne_path = os.path.join(base_path, "network element information.xlsx")
    perf_path = os.path.join(base_path, "performance metrics.xlsx")

    ne_df, perf_df = None, None

    if os.path.exists(ne_path):
        ne_df = pd.read_excel(ne_path)
        ne_df = ne_df.rename(columns=NETWORK_ELEMENT_COLUMNS)
        print(f"\nNetwork element info: {len(ne_df)} rows, columns: {list(ne_df.columns)}")
    else:
        print(f"Missing: {ne_path}")

    if os.path.exists(perf_path):
        perf_df = pd.read_excel(perf_path)
        perf_df = perf_df.rename(columns=PERFORMANCE_METRIC_COLUMNS)
        print(f"Performance metrics: {len(perf_df)} rows, columns: {list(perf_df.columns)}")
    else:
        print(f"Missing: {perf_path}")

    return ne_df, perf_df


# ---------- Run everything ----------

def load_all_data():
    teleqna = load_teleqna()
    summarize_oran_csv("slice_mixed")
    summarize_oran_csv("slice_traffic")
    load_zenodo_fault_data()
    return teleqna


if __name__ == "__main__":
    data = load_all_data()
    print("\nSample TeleQnA entry:")
    print(data[0]["text"][:400])