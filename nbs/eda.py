#!/usr/bin/env python
# coding: utf-8

# ### **Load a small slice**

# In[1]:


from pprint import pprint
from datasets import load_dataset


# In[2]:


DATASET_NAME = "weights-and-wires/pasketti"
SPLIT = "train"


# In[3]:


ds = load_dataset(DATASET_NAME, split=SPLIT, streaming=True)


# In[4]:


sample = next(iter(ds))


# In[5]:


pprint({k: v for k, v in sample.items() if k != "audio"})
print("Audio keys:", sample["audio"])
print("Waveform shape:", sample["audio"]["array"].shape)
print("Sample rate:", sample["audio"]["sampling_rate"])


# In[6]:


ds_small = load_dataset(DATASET_NAME, split="train[:500]")


# ### **Understanding labels**
# > Single words or short utterances

# In[7]:


import pandas as pd


# In[8]:


df = ds_small.to_pandas().drop(columns=["audio"])
df


# ### **Audio duration distribution**

# In[9]:


import matplotlib.pyplot as plt


# In[10]:


df["audio_duration_sec"].hist(bins=60)
plt.xlabel("Duration (seconds)")
plt.title("Utterance Duration Distribution")
plt.show()


# In[11]:


print(df["audio_duration_sec"].describe())
print("Clips under 0.5s:", (df["audio_duration_sec"] < 0.5).sum())
print("Clips over 3s:", (df["audio_duration_sec"] > 3).sum())


# ### **Breakdown by age bucket**

# In[12]:


print(df["age_bucket"].value_counts())


# ### **Waveforms**

# In[13]:


import random
import numpy as np
import IPython.display as ipd


# In[14]:


def plot_sample(sample):
    audio = sample["audio"]
    print("Text:", sample["orthographic_text"])
    print("Age:", sample["age_bucket"])
    ipd.Audio(audio["array"], rate=audio["sampling_rate"])

    plt.figure(figsize=(12, 3))
    plt.plot(audio["array"])
    plt.title(f"\"{sample['orthographic_text']}\" — {sample['age_bucket']}")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.show()


# In[15]:


sample = ds_small[random.randint(0, len(ds_small) - 1)]
plot_sample(sample)


# In[16]:


sample = ds_small[random.randint(0, len(ds_small) - 1)]
plot_sample(sample)


# In[17]:


sample = ds_small[random.randint(0, len(ds_small) - 1)]
plot_sample(sample)


# ### **Silence/Energy check**
# 
# A lot of ASR failure comes from bad clips, too much leading/trailing silence or near silence throughout.

# In[18]:


import io
import librosa


# In[19]:


def rms_energy(waveform):
    return np.sqrt(np.mean(waveform ** 2))


# In[20]:


def decode_and_get_rms(audio_dict):
    try:
        audio_bytes = audio_dict.get("bytes")
        if audio_bytes is None:
            return np.nan

        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
        return rms_energy(y)
    except Exception as e:
        print(f"Error decoding {audio_dict.get('path')}: {e}")
        return np.nan


# In[21]:


df_full = ds_small.to_pandas()  # keep audio column this time
df_full["rms"] = df_full["audio"].apply(decode_and_get_rms)

df_full["rms"].hist(bins=50)
plt.title("RMS Energy Distribution")
plt.show()

# Flag potential bad clips
print("Very quiet clips:", (df_full["rms"] < 0.01).sum())


# ### **Problem 1: Gain/Volume normalization**
# 
# Compare peak amplitude vs RMS to separate truly quiet clips from normally-recorded ones

# In[22]:


df_full.columns


# In[23]:


def audio_stats(waveform):
    rms = np.sqrt(np.mean(waveform ** 2))
    peak = np.max(np.abs(waveform))

    # crest factor:
    # - high value = sparse signal (mostly silence)
    # - low value = consistently loud
    crest_factor = peak / (rms + 1e-9)
    return pd.Series({
        "rms": rms,
        "peak": peak,
        "crest_factor": crest_factor
    })


# In[24]:


def get_crest_factor(audio_dict):
    try:
        audio_bytes = audio_dict.get("bytes")
        if audio_bytes is None:
            return np.nan

        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
        return audio_stats(y)
    except Exception as e:
        print(f"Error decoding {audio_dict.get('path')}: {e}")
        return np.nan


# In[25]:


stats = df_full["audio"].apply(get_crest_factor)
df_full = pd.concat([df_full, stats], axis=1)


# In[26]:


df_full.columns


# **Quiet clips: low peak regardless of content**

# In[27]:


print("Clips with peak < 0.05:", (df_full["peak"] < 0.05).sum())


# ### **Problem 2: Leading/trailing silence**

# In[28]:


def estimate_silence_ratio(audio_dict, frame_ms=20, threshold=0.01):
    try:
        audio_bytes = audio_dict.get("bytes")
        if not audio_bytes:
            return np.nan

        waveform, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
        if len(waveform) == 0:
            return 1.0  # completely silent

        frame_len = int(sr * frame_ms / 1000)
        num_frames = max(1, len(waveform) // frame_len)

        truncated_waveform = waveform[:num_frames * frame_len]
        frames = truncated_waveform.reshape(num_frames, frame_len)
        frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))
        silent_frames = (frame_rms < threshold).sum()

        return silent_frames / num_frames
    except Exception as e:
        return np.nan


# In[29]:


df_full["silence_ratio"] = df_full["audio"].apply(estimate_silence_ratio)

print(df_full["silence_ratio"].describe())
print(f"Clips >50% silence: {(df_full['silence_ratio'] > 0.5).sum()}")


# In[30]:


df_full.columns


# In[31]:


test_audio = df_full["audio"].iloc[0]
audio_bytes = test_audio['bytes']
waveform, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
print(f"Waveform shape: {waveform.shape}, SR: {sr}")


# In[ ]:




