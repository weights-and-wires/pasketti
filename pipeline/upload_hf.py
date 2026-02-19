"""Upload the dataset README card to the Hugging Face repo.

The actual parquet data is pushed by preprocess.py via datasets.push_to_hub().
This script only manages the README / dataset card.
"""

from huggingface_hub import HfApi

REPO_ID = "weights-and-wires/pasketti"

README_CONTENT = """\
---
license: other
task_categories:
  - automatic-speech-recognition
language:
  - en
tags:
  - children-speech
  - asr
  - driven-data
  - pasketti
size_categories:
  - 10K<n<100K
---

# On Top of Pasketti — Children's Word ASR (Training Data)

Training dataset for the [DrivenData "On Top of Pasketti" Children's Word ASR competition](https://www.drivendata.org/competitions/308/childrens-word-asr/).

## Contents

**95,572 rows** split across multiple parquet shards (~450 MB each) in `data/`, with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `utterance_id` | string | Unique identifier for each utterance |
| `child_id` | string | Anonymized speaker identifier |
| `session_id` | string | Recording session identifier |
| `audio_path` | string | Original relative path to the `.flac` file |
| `audio_duration_sec` | float64 | Duration of the audio clip in seconds |
| `age_bucket` | string | Age range: `3-4`, `5-7`, `8-11`, `12+`, or `unknown` |
| `md5_hash` | string | MD5 checksum of the original audio file |
| `filesize_bytes` | int64 | Size of the original audio file in bytes |
| `orthographic_text` | string | Normalized orthographic transcription |
| `audio` | Audio | Embedded FLAC audio (playable in the dataset viewer) |

## Quick Start

```python
from datasets import load_dataset

ds = load_dataset("weights-and-wires/pasketti", split="train")

for sample in ds:
  audio = sample["audio"]
  waveform = audio["array"]           # numpy float32 array
  sr       = audio["sampling_rate"]   # original sampling rate
  text     = sample["orthographic_text"]
  # ... use with your ASR model
```

To resample all audio to a fixed rate (e.g. 16 kHz for Whisper):

```python
from datasets import Audio

ds = ds.cast_column("audio", Audio(sampling_rate=16_000))
```

## Statistics

| Stat | Value |
|------|-------|
| Total utterances | 95,572 |
| Total audio duration | ~185.4 hours |
| Audio format | FLAC (embedded) |
| Download size | ~12.6 GB |

## Age Distribution

| Age Bucket | Count | % |
|------------|-------|---|
| 3-4 | 10,112 | 10.6% |
| 5-7 | 11,490 | 12.0% |
| 8-11 | 73,970 | 77.4% |
"""

if __name__ == "__main__":
  api = HfApi()
  api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True)

  print("Uploading README.md ...")
  api.upload_file(
    path_or_fileobj=README_CONTENT.encode(),
    path_in_repo="README.md",
    repo_id=REPO_ID,
    repo_type="dataset",
    commit_message="Update dataset README",
  )
  print("Done!")
