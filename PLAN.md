# On Top of Pasketti: Children's Word ASR — Competition Plan

## Table of Contents

- [Competition Summary](#competition-summary)
- [Dataset at a Glance](#dataset-at-a-glance)
- [ASR Crash Course for LLM Practitioners](#asr-crash-course-for-llm-practitioners)
- [Phase 0: Environment & Zero-Shot Baseline](#phase-0-environment--zero-shot-baseline)
- [Phase 1: Data Understanding & Preprocessing](#phase-1-data-understanding--preprocessing)
- [Phase 2: Fine-Tuning Whisper](#phase-2-fine-tuning-whisper)
- [Phase 3: Improving the Model](#phase-3-improving-the-model)
- [Phase 4: Children-Specific Optimizations](#phase-4-children-specific-optimizations)
- [Phase 5: Submission Packaging](#phase-5-submission-packaging)
- [Priority Ranking](#priority-ranking)
- [Useful Resources](#useful-resources)

---

## Competition Summary

| Field | Detail |
|-------|--------|
| **Goal** | Predict the words spoken by children in short audio clips |
| **Metric** | Word Error Rate (WER) — lower is better |
| **Secondary Metric** | Noisy WER (WER on classroom-noise subset) — for bonus prize |
| **Submission** | Code execution — you submit model + inference code in a container |
| **Prize Pool** | $120,000 across both tracks (Word + Phonetic) |

The metric formula:

$$
\text{WER} = \frac{S + D + I}{N}
$$

Where $S$ = substitutions, $D$ = deletions, $I$ = insertions, $N$ = total reference words.

Before scoring, both predictions and ground truth are normalized using Whisper's `EnglishTextNormalizer`, which handles punctuation removal, contraction expansion, number standardization, etc. The scoring script is provided at `metric/score.py`.

---

## Dataset at a Glance

### Training Data

| Statistic | Value |
|-----------|-------|
| Total utterances | 95,572 |
| Total audio duration | ~185.4 hours |
| Min duration | 0.08s |
| Max duration | ~1,348s (22+ minutes — outlier!) |
| Mean duration | 6.98s |
| Mean word count per utterance | 13.5 |
| Max word count | 455 |
| Empty transcripts | 0 |

### Age Distribution (Training)

| Age Bucket | Count | Percentage |
|------------|-------|------------|
| 3–4 | 10,112 | 10.6% |
| 5–7 | 11,490 | 12.0% |
| 8–11 | 73,970 | 77.4% |

> **Key observation:** The dataset is heavily skewed toward 8–11 year olds. Younger children (3–4) with the most non-standard speech patterns are underrepresented. This imbalance will need to be addressed during training.

### Test Data

| Submission File | Utterance Count |
|-----------------|----------------|
| `submission_format_aqPHQ8m.jsonl` | 205,274 |
| `submission_format_z2HCh3r.jsonl` | 9,000 |
| **Total** | **214,274** |

> The test set is ~2.2x larger than training. Some test data sources may not appear in training at all, so generalization is critical.

### Audio Format

- **Training audio:** Variable sample rates, mono or stereo, `.flac` format
- **Test audio:** Normalized to **16 kHz, mono**, `.flac` format
- Audio has been scrubbed of personally identifying information and adult speech

### Label Characteristics

- Normalized orthographic transcriptions (intended words, not verbatim)
- Disfluencies (false starts, repetitions, stutters) are generally **omitted**
- Developmentally typical non-standard forms are **preserved** (e.g., "goed" instead of "went", "tooths" instead of "teeth")
- Environmental noises and non-lexical sounds are **not labeled**
- Occasional non-English words within English utterances are kept (e.g., "my abuela got me those")

---

## ASR Crash Course for LLM Practitioners

If you come from LLM pre-training/fine-tuning, here are the key conceptual mappings:

### Architecture Analogy

| LLM Concept | ASR (Whisper) Equivalent |
|-------------|--------------------------|
| Tokenizer | Feature extractor (converts raw audio → log-mel spectrogram) |
| Input tokens | 80-channel log-mel spectrogram frames (computed from raw waveform) |
| Transformer decoder | Same — Whisper uses an encoder-decoder transformer |
| Text generation (autoregressive) | Transcript generation (autoregressive, identical decoding) |
| Fine-tuning with LoRA/QLoRA | Exactly the same — apply PEFT adapters to attention layers |
| `Trainer` / `SFTTrainer` | `Seq2SeqTrainer` (nearly identical API) |
| Perplexity / cross-entropy loss | Cross-entropy loss on transcript tokens (same!) |
| BLEU / ROUGE | WER (Word Error Rate) — edit distance at word level |

### How Whisper Works

1. **Audio → Waveform:** Load the `.flac` file as a 1D float array at 16kHz
2. **Waveform → Log-Mel Spectrogram:** A short-time Fourier transform (STFT) followed by a mel filterbank produces an 80-channel spectrogram. This is analogous to "tokenization" — it converts raw signal into a structured representation
3. **Encoder:** A standard Transformer encoder processes the spectrogram. Whisper's encoder handles **30-second windows** — audio is padded or chunked to fit
4. **Decoder:** A standard Transformer decoder autoregressively generates text tokens, exactly like GPT generates text. It attends to both the encoder output and its own previous tokens
5. **Decoding strategies:** Beam search, temperature sampling, etc. — identical to LLM decoding

### Two ASR Paradigms

| | Seq2Seq (Whisper) | CTC (wav2vec2, Conformer) |
|---|---|---|
| **Architecture** | Encoder-decoder | Encoder-only + linear head |
| **How it works** | Decoder generates tokens autoregressively | Each audio frame is independently classified into a character/token |
| **Strengths** | Higher accuracy, handles punctuation/formatting | Faster inference, no hallucination risk |
| **Weaknesses** | Can hallucinate on silence/noise, slower | Needs external language model for best results |
| **Analogy** | Like a seq2seq translation model | Like a token classifier (NER-style) |

### Key ASR-Specific Concepts

- **Sample rate:** Number of audio samples per second. 16kHz = 16,000 samples/second. All audio must be resampled to the model's expected rate (16kHz for Whisper)
- **Log-mel spectrogram:** A 2D time-frequency representation of audio. Think of it as the "image" version of sound — models process this, not raw waveform
- **Spectrogram dimensions:** For Whisper, the input is `(80, 3000)` — 80 mel frequency bins × 3000 time frames (covering 30 seconds)
- **WER hallucination:** Unlike LLMs where hallucination means factually wrong text, in ASR "hallucination" means the model generates text when the audio is silent or noisy. Whisper is known to do this — it may repeat phrases or invent words on silence
- **CTC blank token:** In CTC models, a special "blank" token is predicted for frames that don't correspond to any output character. This handles the alignment problem without attention

---

## Phase 0: Environment & Zero-Shot Baseline

**Goal:** Get a working end-to-end pipeline and establish a baseline WER score.

**Timeline:** Days 1–3

### Step 0.1: Hardware Requirements

ASR fine-tuning is GPU-intensive due to the audio encoder. Recommended setups:

| Setup | VRAM | Can Train | Notes |
|-------|------|-----------|-------|
| A100 80GB | 80GB | Full fine-tune of large-v3 | Ideal |
| A100 40GB / A6000 | 40–48GB | Full fine-tune with gradient checkpointing | Good |
| RTX 4090 / A10 | 24GB | LoRA/QLoRA fine-tune of large-v3 | Viable |
| RTX 3090 / T4 | 16–24GB | LoRA of medium or small | Budget option |

Cloud options: RunPod, Lambda Labs, Vast.ai, or Google Colab Pro+ (A100).

### Step 0.2: Install Dependencies

```bash
# Core dependencies
pip install torch torchaudio transformers datasets accelerate
pip install jiwer soundfile librosa
pip install peft bitsandbytes  # For LoRA/QLoRA

# Data augmentation (for later phases)
pip install audiomentations

# Evaluation (matches competition scoring)
pip install jiwer pandas
```

### Step 0.3: Zero-Shot Whisper Baseline

Run `openai/whisper-large-v3` (or `whisper-large-v3-turbo` for speed) without any fine-tuning on the training data, then compute WER. This tells you:

- How well a general-purpose ASR model handles children's speech out-of-the-box
- Your expected gap to close with fine-tuning
- Whether your pipeline (audio loading → inference → WER calculation) works correctly

**Pseudocode for the baseline:**

```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio, json, jiwer

processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")

# For each utterance in the validation set:
#   1. Load audio with torchaudio.load()
#   2. Resample to 16kHz if needed
#   3. Extract input features with processor()
#   4. Generate transcription with model.generate()
#   5. Decode tokens with processor.batch_decode()

# Compute WER using the competition's normalizer
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
normalizer = EnglishTextNormalizer()
# Normalize both predictions and references before computing WER
```

**Expected zero-shot WER:** Likely 15–30% on children's speech (adult speech typically gets 3–5% with Whisper-large-v3). The gap is your opportunity.

### Step 0.4: Validate Scoring Locally

Use the competition's `metric/score.py` to ensure your local WER matches what the leaderboard will compute. Pay special attention to:

- The `EnglishTextNormalizer` is applied before WER calculation
- Empty predictions should be handled (predict empty string `""` for silence)
- The JSONL output format must have exactly `utterance_id` and `orthographic_text` fields

---

## Phase 1: Data Understanding & Preprocessing

**Goal:** Clean and prepare the training data for optimal fine-tuning.

**Timeline:** Days 3–5

### Step 1.1: Audit and Filter Training Data

Key issues to investigate and address:

1. **Extreme duration outliers:** The max duration is ~1,348 seconds (22+ minutes). Whisper processes 30-second windows. Audio longer than 30s needs to be either:
   - **Chunked** using Whisper's built-in `pipeline("automatic-speech-recognition", chunk_length_s=30)` during inference — but for training, you want clean segments
   - **Filtered out** if unreasonably long (e.g., >60s) — these are likely annotation errors or full-session recordings
   
2. **Very short utterances** (<0.5s): Often single words ("hm", "good", "yes"). These are valid and should be kept, but may need padding attention during batching.

3. **Duration distribution bins for training decisions:**
   - <1s: ~X% (single words)
   - 1–10s: ~Y% (typical utterances — sweet spot for Whisper)
   - 10–30s: ~Z% (longer narratives)
   - >30s: outliers to handle

**Recommended filter:** Keep utterances with `0.1s ≤ duration ≤ 30s` for fine-tuning. For utterances >30s, consider segmenting them or excluding them.

### Step 1.2: Resample All Training Audio to 16kHz Mono

Test audio is standardized to 16kHz mono, but training audio varies. Train-test mismatch in sample rate can silently degrade performance.

```python
import torchaudio

def resample_audio(audio_path, target_sr=16000):
    waveform, sr = torchaudio.load(audio_path)
    # Convert stereo to mono if needed
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
    return waveform, target_sr
```

You can either pre-process all audio files and save them, or do this on-the-fly in your dataset `__getitem__`. On-the-fly is more flexible but slower; pre-processing is recommended if disk space allows.

### Step 1.3: Analyze and Address Age Imbalance

The dataset is 77% age 8–11. Younger children (3–4, 5–7) have more non-standard speech and are arguably harder to transcribe — exactly the cases where you want more training signal.

**Strategies:**

| Strategy | How | Trade-off |
|----------|-----|-----------|
| **Oversampling** | Repeat 3–4 and 5–7 utterances 2–3x during training | Simple, risk of overfitting on repeated samples |
| **Weighted sampling** | Use a `WeightedRandomSampler` to equalize age bucket probability | Better than oversampling, easy to implement |
| **Curriculum learning** | Train on all data first, then fine-tune with emphasis on younger age groups | More complex, potentially best results |
| **Do nothing** | Let the natural distribution stand | May be fine if test distribution matches training |

Without knowing the test age distribution, **weighted sampling** is a safe default.

### Step 1.4: Create a Proper Validation Split

**Critical:** Split by `child_id`, NOT randomly by utterance.

Why: A single child has multiple utterances across sessions. If the same child appears in both train and validation, the model memorizes speaker-specific characteristics → inflated val scores → nasty surprise on the leaderboard.

```python
# Group utterances by child_id
# Assign ~5% of CHILDREN to validation
# All of a child's utterances go to the same split
# Stratify by age_bucket to maintain distribution
```

This mirrors real-world deployment: the model will encounter **new children** it has never heard before.

---

## Phase 2: Fine-Tuning Whisper

**Goal:** Fine-tune a pre-trained Whisper model on the children's speech training data.

**Timeline:** Days 5–14

This is the single highest-impact step. If you come from LLM fine-tuning, the workflow is nearly identical except the input is audio instead of text.

### Step 2.1: Choose Your Base Model

| Model | Parameters | VRAM (full FT) | VRAM (LoRA) | Quality |
|-------|-----------|-----------------|-------------|---------|
| `openai/whisper-large-v3` | 1.55B | ~40GB | ~16GB | Best |
| `openai/whisper-large-v3-turbo` | 809M | ~24GB | ~10GB | Nearly as good, 2x faster |
| `openai/whisper-medium` | 769M | ~20GB | ~10GB | Good |
| `openai/whisper-small` | 244M | ~8GB | ~4GB | Decent |

**Recommendation:** Start with `whisper-large-v3-turbo` for the best speed/quality trade-off during experimentation. Use `whisper-large-v3` for your final submission.

### Step 2.2: Set Up the HuggingFace Training Pipeline

The key components:

1. **`WhisperProcessor`** — Handles both feature extraction (audio → mel spectrogram) and tokenization (text → token IDs). This is your combined "tokenizer + feature extractor."

2. **`WhisperForConditionalGeneration`** — The model. Encoder processes mel spectrograms, decoder generates transcript tokens.

3. **`Seq2SeqTrainer`** — HuggingFace's trainer for encoder-decoder models. Virtually identical to `Trainer` but handles generation during evaluation.

4. **`Seq2SeqTrainingArguments`** — Same as `TrainingArguments` with extra seq2seq-specific options like `predict_with_generate=True`.

**Key training setup:**

```python
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

# Data collator for Whisper — pads audio features and labels
# Need a custom collator that:
#   1. Pads input_features to same length in a batch
#   2. Pads labels (token IDs) and replaces padding with -100

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-children-ft",
    per_device_train_batch_size=16,       # Adjust based on VRAM
    gradient_accumulation_steps=2,         # Effective batch = 32
    learning_rate=1e-5,                    # Standard for Whisper FT
    warmup_steps=500,
    max_steps=5000,                        # ~1-2 epochs over 95K samples
    fp16=True,                             # or bf16=True on Ampere+
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=500,
    logging_steps=25,
    predict_with_generate=True,            # Run generation during eval
    generation_max_length=225,
    report_to="wandb",                     # Optional but recommended
    gradient_checkpointing=True,           # Saves ~40% VRAM
    dataloader_num_workers=4,
)
```

### Step 2.3: LoRA/QLoRA Fine-Tuning (If GPU-Constrained)

Since you're familiar with PEFT from LLM work, this is straightforward:

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=32,                          # Rank — 16-64 is typical
    lora_alpha=64,                 # Alpha — usually 2x rank
    target_modules=[               # Apply to attention layers
        "q_proj", "v_proj",        # Decoder attention (minimum)
        "k_proj", "o_proj",        # Can also include these
        # Optionally include encoder attention for better adaptation
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_2_SEQ_LM",     # Critical — tells PEFT this is seq2seq
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # Should be ~1-3% of total
```

**LoRA considerations for ASR:**
- Applying LoRA only to the **decoder** adapts the language generation (what words to predict) — this is usually sufficient
- Applying LoRA to **both encoder and decoder** also adapts acoustic feature extraction (how to interpret children's voice characteristics) — better for domain shift like children's speech
- Start with decoder-only LoRA, then try encoder+decoder if you have capacity

### Step 2.4: Key Training Hyperparameters

| Parameter | Recommended Value | Rationale |
|-----------|-------------------|-----------|
| Learning rate | 1e-5 (full FT), 2e-4 (LoRA) | Standard for Whisper; LoRA uses higher LR like in LLM fine-tuning |
| Batch size (effective) | 32–64 | Larger batches stabilize seq2seq training |
| Warmup | 500 steps | Prevents early divergence |
| Weight decay | 0.01 | Standard regularization |
| Max steps | 5,000–10,000 | ~1-3 epochs; monitor val WER for early stopping |
| Gradient checkpointing | On | Saves ~40% VRAM at ~20% speed cost |
| fp16/bf16 | On | Essential for large models; bf16 preferred on Ampere+ |
| Label smoothing | 0.0 | Whisper authors did not use it; can experiment with 0.1 |
| Forced decoder IDs | Set language="en", task="transcribe" | Tells Whisper to do English transcription specifically |

### Step 2.5: Data Collator

Whisper needs a special data collator because:
- Audio features (mel spectrograms) need padding to the longest in the batch
- Labels (token sequences) need padding and replacement of pad tokens with `-100` (so they're ignored in loss)

```python
import torch
from dataclasses import dataclass

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor
    
    def __call__(self, features):
        # Pad input features (audio)
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        
        # Pad labels (text tokens)
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        
        # Replace padding with -100 for loss masking
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels
        return batch
```

### Step 2.6: Monitor Training

Track these metrics during training:

- **Training loss** — should decrease steadily
- **Validation WER** — the actual metric. Compute using `jiwer.wer()` with the Whisper normalizer applied
- **Learning rate schedule** — verify warmup is working
- **Generation samples** — periodically decode a few val examples and inspect the text. Look for hallucinations, repeated phrases, or systematic errors

**Common failure modes to watch for:**
- WER plateaus early → learning rate too low, or model already saturated on this data
- WER spikes then recovers → normal with seq2seq models, keep training
- Model generates repetitive text ("the the the the...") → reduce max generation length, try higher beam penalty
- Model hallucinates on silence → add some silent/noise-only clips with empty transcripts to training

---

## Phase 3: Improving the Model

**Goal:** Push WER lower through augmentation, alternative models, and ensembling.

**Timeline:** Days 14–21

### Step 3.1: Data Augmentation for Noise Robustness

The competition has a **Noisy Classroom Bonus** evaluated on recordings with background noise, crosstalk, and varying audio quality. Augmentation during training is critical for this.

Use the `audiomentations` library to apply on-the-fly augmentation to training audio:

```python
from audiomentations import (
    Compose, AddGaussianNoise, AddBackgroundNoise,
    TimeStretch, PitchShift, Shift, Gain,
    ClippingDistortion, RoomSimulator
)

augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.3),
    TimeStretch(min_rate=0.9, max_rate=1.1, p=0.2),     # Speed perturbation
    PitchShift(min_semitones=-2, max_semitones=2, p=0.2), # Pitch variation
    Gain(min_gain_db=-6, max_gain_db=6, p=0.3),           # Volume variation
    ClippingDistortion(max_percentile_threshold=10, p=0.1),
    # RoomSimulator(p=0.2),  # Simulates room reverb — expensive but effective
])
```

**Why each augmentation helps:**

| Augmentation | Simulates | Helps With |
|-------------|-----------|------------|
| Gaussian noise | Microphone hiss, electrical noise | General robustness |
| Background noise | Classroom chatter, environmental sounds | Noisy WER bonus |
| Time stretch | Variable speaking rate (children speak irregularly) | Speed variation |
| Pitch shift | Different voice pitches (ages 3–12 vary widely) | Speaker variation |
| Gain | Varying distances from microphone | Volume robustness |
| Clipping | Poor-quality recordings, volume peaks | Bad hardware |
| Room simulator | Reverberation in classrooms | Real-world acoustics |

**SpecAugment** (frequency and time masking on the spectrogram) is another powerful technique. Whisper's feature extractor doesn't apply it by default, but you can add it manually:

```python
# After computing mel spectrogram, randomly mask:
# - 1-2 frequency bands (height of spectrogram)
# - 1-2 time bands (width of spectrogram)
# This is well-established in ASR and acts as regularization
```

### Step 3.2: Try Alternative / Additional Models

Don't put all eggs in the Whisper basket. Other strong ASR models to consider:

#### NVIDIA NeMo Models

- **`nvidia/parakeet-tdt-0.6b-v2`** — A CTC-based model, very fast, strong accuracy
- **`nvidia/canary-1b`** — Attention-based, supports multiple languages
- These use a **Conformer** encoder (CNN + Transformer hybrid) which is often better than pure Transformer for audio

#### wav2vec2 / HuBERT

- **`facebook/wav2vec2-large-960h`** — Pre-trained on 960h of Librispeech
- CTC-based, so no hallucination risk
- Can be combined with a language model for better results
- Easier to fine-tune on smaller GPUs (no decoder)

#### Whisper Variants

- **`distil-whisper/distil-large-v3`** — Distilled Whisper, 2x faster with minimal quality loss
- **Fine-tuned community models** — Search HuggingFace for "whisper children" or "whisper kids" — someone may have already fine-tuned on children's speech data

### Step 3.3: Ensemble Multiple Models

Ensembling is one of the most reliable ways to reduce WER in ASR competitions. Options:

#### Option A: ROVER (Recognizer Output Voting Error Reduction)

- Take transcriptions from N models for each utterance
- Align them using dynamic time warping
- At each position, take the majority vote word
- The `asr-eval` or `sctk` tools implement this

#### Option B: N-best List Rescoring

- Generate N-best hypotheses from each model (using beam search)
- Score all hypotheses with a separate language model
- Pick the best-scoring hypothesis across all models

#### Option C: Simple Heuristic Selection

- Run 2–3 models on each utterance
- If they all agree → high confidence, use any
- If they disagree → use the model that historically has lower WER on similar utterances (based on duration, estimated difficulty)

**Expected improvement:** 5–15% relative WER reduction from 2–3 model ensemble.

### Step 3.4: Experiment with Decoding Parameters

Whisper's `model.generate()` has many knobs, just like LLM generation:

| Parameter | Default | Experiment With | Effect |
|-----------|---------|-----------------|--------|
| `num_beams` | 1 (greedy) | 5 | Better quality, 5x slower |
| `temperature` | 0 | 0.0–0.2 | Affects fallback when beam search fails |
| `compression_ratio_threshold` | 2.4 | 1.8–2.4 | Detects repetitive/hallucinated text |
| `logprob_threshold` | -1.0 | -0.8 to -1.5 | Detects low-confidence predictions |
| `no_speech_threshold` | 0.6 | 0.3–0.8 | Sensitivity to silence detection |
| `length_penalty` | 1.0 | 0.8–1.2 | Penalize/encourage longer outputs |
| `repetition_penalty` | 1.0 | 1.1–1.3 | Reduce repeated tokens |

> **From Whisper's own inference:** When beam search gives high compression ratio (repetition) or low log-probability, Whisper falls back to temperature-based sampling. Understanding and tuning these fallback thresholds is important for robust inference.

---

## Phase 4: Children-Specific Optimizations

**Goal:** Tailor the model specifically to the nuances of children's speech.

**Timeline:** Days 21–28

### Step 4.1: Language Model Rescoring / Constrained Decoding

Children use simpler, more predictable vocabulary than adults. You can exploit this:

1. **Build an n-gram language model** on the training transcripts:
   ```bash
   # Using KenLM (fast n-gram LM)
   pip install https://github.com/kpu/kenlm/archive/master.zip
   # Train a 4-gram LM on training transcripts
   ```

2. **Shallow fusion:** During beam search, interpolate the Whisper decoder scores with the n-gram LM scores:
   $$
   \text{score} = \log P_{\text{whisper}}(y|x) + \lambda \cdot \log P_{\text{LM}}(y)
   $$
   Where $\lambda$ is a weight (typically 0.1–0.5) tuned on validation.

3. **Alternatively, use GPT-2 or a small LLM** as the language model — you could even fine-tune a small GPT-2 on the training transcripts for a very strong children's speech LM.

### Step 4.2: Prompt Engineering for Whisper

Whisper supports `initial_prompt` and `prefix` during generation. These bias the decoder toward certain patterns:

```python
# Experiment with different prompts
model.generate(
    input_features,
    language="en",
    task="transcribe",
    # initial_prompt biases the model's word expectations
    initial_prompt="A child is speaking.",
    # OR more specific:
    initial_prompt="A young child is speaking in a classroom.",
)
```

This is lightweight (no training needed) and can help with:
- Biasing toward child-typical vocabulary
- Reducing hallucinations
- Setting the expected register of speech

### Step 4.3: Handle Edge Cases

#### Very short utterances (<1s)

- Often single words: "hm", "yes", "good"
- Whisper may hallucinate on very short audio (pad with silence)
- Consider a separate small model specialized for short clips, or increase `no_speech_threshold`

#### Non-standard child speech forms

- "goed", "tooths", "runned" — these are CORRECT according to the ground truth
- Whisper's general English LM may "correct" these to standard forms
- Fine-tuning on the training data should teach the model to preserve child forms
- Verify on validation that the model produces "goed" not "went" in appropriate contexts

#### Embedded non-English words

- "my abuela got me those" — the Spanish word should be preserved
- Whisper-large-v3 handles multilingual speech, so this should work if you don't force English-only tokenization too aggressively

#### Silence / noise-only clips

- Some test clips may contain no speech
- The model should output an empty string `""` for these
- Tune `no_speech_threshold` to correctly identify silence without being too aggressive

### Step 4.4: Specialized Models by Age Group

Since speech characteristics differ dramatically by age:

- 3–4 year olds: Short utterances, frequent mispronunciations, limited vocabulary
- 5–7 year olds: Developing fluency, some reading tasks
- 8–11 year olds: More adult-like speech, longer utterances

You could train separate models or LoRA adapters per age group and route at inference time. However, this requires knowing the age of test speakers (which is not provided at inference). This is only viable if you build an age classifier from the audio.

---

## Phase 5: Submission Packaging

**Goal:** Package your model for containerized code execution.

**Timeline:** Days 28–30

### Step 5.1: Understand the Submission Format

This is a **code execution** competition. You submit:
- Your trained model weights
- An inference script
- A Docker-compatible environment specification

Your code must:
1. Read the submission format JSONL (which lists `utterance_id`s to predict)
2. Load the corresponding audio from the `dataset/audio/` directory
3. Run your model
4. Output a JSONL file with `utterance_id` and `orthographic_text` for each utterance

### Step 5.2: Optimize Inference Speed

With **214K test utterances**, inference speed matters. Runtime limits in code execution competitions are typically 8–12 hours on a single GPU.

**Speed optimizations:**

| Technique | Speedup | Effort |
|-----------|---------|--------|
| fp16 inference | ~2x | Trivial — just load model in `torch.float16` |
| `torch.compile()` | ~1.3–1.5x | One line of code, but may not work with all models |
| Flash Attention / SDPA | ~1.5–2x | Use `attn_implementation="sdpa"` when loading model |
| Batch inference | ~2–4x | Group audio by similar length, process in batches |
| `distil-whisper` or `turbo` | ~2x | Use a faster model variant |
| `whisper.cpp` / `faster-whisper` | ~3–4x | CTranslate2-based, much faster than HuggingFace |

**`faster-whisper`** deserves special mention: it's a CTranslate2 reimplementation of Whisper that is significantly faster than the HuggingFace version. You can convert your fine-tuned model to CTranslate2 format for submission.

```bash
pip install faster-whisper
# Convert HuggingFace model to CTranslate2
ct2-opus-mt-converter --model openai/whisper-large-v3 --output_dir whisper-ct2
```

### Step 5.3: Batched Inference Strategy

```python
# Sort utterances by audio duration for efficient batching
# (avoids excessive padding waste)
utterances.sort(key=lambda x: x["audio_duration_sec"])

# Process in batches of similar length
for batch in chunk(utterances, batch_size=16):
    audio_batch = [load_audio(u["audio_path"]) for u in batch]
    features = processor(audio_batch, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        predicted_ids = model.generate(**features.to(device))
    transcriptions = processor.batch_decode(predicted_ids, skip_special_tokens=True)
```

### Step 5.4: Output Format

The output JSONL must exactly match:

```json
{"utterance_id": "U_00014f97ad2021ef", "orthographic_text": "predicted words here"}
{"utterance_id": "U_000191622021fce4", "orthographic_text": "another prediction"}
```

- One line per utterance
- Every `utterance_id` from the submission format must appear
- `orthographic_text` can be empty string `""` for silence
- **Do NOT apply the Whisper normalizer to your output** — the scoring script applies it

### Step 5.5: Pre-Submission Checklist

- [ ] All `utterance_id`s from submission format are present in output
- [ ] No extra `utterance_id`s in output
- [ ] JSONL is valid (one JSON object per line, UTF-8 encoded)
- [ ] Predictions are raw text (normalizer is applied during scoring, not by you)
- [ ] Model loads and runs within the container's time limit
- [ ] Inference completes on all 214K utterances within the time budget
- [ ] Docker/container builds and runs locally before submitting

---

## Priority Ranking

Ordered by **impact-to-effort ratio** — do the top items first:

| # | Action | Impact | Effort | When |
|---|--------|--------|--------|------|
| 1 | **Fine-tune Whisper-large-v3(-turbo) on training data** | Huge | Medium | Phase 2 |
| 2 | **Noise augmentation during training** | High | Low | Phase 3 |
| 3 | **Proper child_id-based val split** | Medium | Low | Phase 1 |
| 4 | **Ensemble 2–3 diverse models** | High | Medium | Phase 3 |
| 5 | **Decoding parameter tuning** | Medium | Low | Phase 3 |
| 6 | **Whisper prompt engineering** | Low–Medium | Very Low | Phase 4 |
| 7 | **LM rescoring** | Medium | Medium | Phase 4 |
| 8 | **faster-whisper for inference speed** | Medium | Low | Phase 5 |
| 9 | **Alternative architectures (NeMo, wav2vec2)** | Medium | High | Phase 3 |
| 10 | **Age-specific models** | Low–Medium | High | Phase 4 |

---

## Useful Resources

### Tutorials & Guides

- [Fine-Tune Whisper with HuggingFace](https://huggingface.co/blog/fine-tune-whisper) — The canonical tutorial for Whisper fine-tuning. Covers the full pipeline from data to training.
- [HuggingFace Audio Course](https://huggingface.co/learn/audio-course) — Free course covering ASR fundamentals, Whisper, CTC models, and more.
- [Whisper Paper (Radford et al., 2022)](https://arxiv.org/abs/2212.04356) — Original paper. Section 2 (Approach) and Section 4 (Analysis) are most relevant.

### Libraries

- [`transformers`](https://github.com/huggingface/transformers) — Whisper, wav2vec2, and all model implementations
- [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper) — CTranslate2-based fast Whisper inference
- [`audiomentations`](https://github.com/iver56/audiomentations) — Audio data augmentation
- [`jiwer`](https://github.com/jitsi/jiwer) — WER computation
- [`peft`](https://github.com/huggingface/peft) — LoRA/QLoRA adapters
- [`NeMo`](https://github.com/NVIDIA/NeMo) — NVIDIA's ASR toolkit with Conformer/Parakeet models

### Competition-Specific

- [DrivenData Competition Page](https://www.drivendata.org/competitions/308/childrens-word-asr/page/973/)
- `metric/score.py` — Local scoring script with Whisper normalizer + WER computation
- Submission format examples: `dataset/submission_format_aqPHQ8m.jsonl`, `dataset/submission_format_z2HCh3r.jsonl`

### Relevant Papers

- [Children's Speech Recognition: Past, Present and Future (2024)](https://arxiv.org/abs/2402.09387) — Survey of children's ASR challenges
- [Whisper (Radford et al., 2022)](https://arxiv.org/abs/2212.04356) — The base model
- [LoRA (Hu et al., 2021)](https://arxiv.org/abs/2106.09685) — Low-Rank Adaptation
- [SpecAugment (Park et al., 2019)](https://arxiv.org/abs/1904.08779) — Spectrogram augmentation for ASR
