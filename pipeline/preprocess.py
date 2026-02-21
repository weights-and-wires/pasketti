"""
Build the HuggingFace dataset from training JSONL + FLAC audio files and
push it to the Hub.

Uses datasets.Dataset.from_generator() with an Audio() feature so the
resulting parquet shards carry proper HF metadata and the dataset viewer
renders an inline audio player for every row.
"""

import json
from pathlib import Path

from datasets import Audio, Dataset, Features, Value

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR.parent.parent / "dataset"
JSONL_PATH = DATASET_DIR / "train_word_transcripts.jsonl"
REPO_ID = "weights-and-wires/pasketti"

FEATURES = Features(
    {
        "utterance_id": Value("string"),
        "child_id": Value("string"),
        "session_id": Value("string"),
        "audio_path": Value("string"),
        "audio_duration_sec": Value("float64"),
        "age_bucket": Value("string"),
        "md5_hash": Value("string"),
        "filesize_bytes": Value("int64"),
        "orthographic_text": Value("string"),
        "audio": Audio(),
    }
)


def gen():
    with JSONL_PATH.open() as fh:
        for line in fh:
            rec = json.loads(line)
            audio_file = DATASET_DIR / rec["audio_path"]
            yield {
                "utterance_id": rec["utterance_id"],
                "child_id": rec["child_id"],
                "session_id": rec["session_id"],
                "audio_path": rec["audio_path"],
                "audio_duration_sec": rec["audio_duration_sec"],
                "age_bucket": rec["age_bucket"],
                "md5_hash": rec["md5_hash"],
                "filesize_bytes": rec["filesize_bytes"],
                "orthographic_text": rec["orthographic_text"],
                "audio": {"bytes": audio_file.read_bytes(), "path": audio_file.name},
            }


if __name__ == "__main__":
    print(f"Building dataset from {JSONL_PATH} ...")
    ds = Dataset.from_generator(gen, features=FEATURES)
    print(f"Dataset ready: {ds}")
    print(f"Pushing to {REPO_ID} (shards ~500 MB each) ...")
    ds.push_to_hub(REPO_ID, max_shard_size="500MB")
    print("Done!")
