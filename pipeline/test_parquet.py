"""Quick sanity check: stream the first few rows from the HF dataset."""

from datasets import load_dataset

REPO_ID = "weights-and-wires/pasketti"

ds = load_dataset(REPO_ID, split="train", streaming=True)

for i, sample in enumerate(ds):
  if i >= 3:
    break
  audio = sample["audio"]
  print(f"Row {i}:")
  print(f"  utterance_id:  {sample['utterance_id']}")
  print(f"  orthographic:  {sample['orthographic_text']}")
  print(f"  age_bucket:    {sample['age_bucket']}")
  print(f"  audio sr:      {audio['sampling_rate']}")
  print(f"  audio samples: {len(audio['array']):,}")
  print()
