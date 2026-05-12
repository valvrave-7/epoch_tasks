# Multimodal Emotion Recognition — Technical Report
**Dataset:** RAVDESS Emotional Speech Audio (1,440 clips, 24 actors, 8 emotions)

---

## 1. Architecture Diagram

```
                        ┌─────────────────────────────────┐
                        │         INPUT LAYER              │
              ┌─────────┴──────────┐         ┌────────────┴──────────┐
              │   Audio (.wav)     │         │   Audio (.wav)        │
              │   via librosa      │         │   via Whisper STT     │
              └─────────┬──────────┘         └────────────┬──────────┘
                        │                                 │
              ┌─────────▼──────────┐         ┌────────────▼──────────┐
              │  Mel-Spectrogram   │         │   Text Transcript     │
              │  (64 × T × 1)      │         │   Tokenised + Padded  │
              └─────────┬──────────┘         └────────────┬──────────┘
                        │                                 │
              ┌─────────▼──────────┐         ┌────────────▼──────────┐
              │   AUDIO CNN        │         │   TEXT RNN            │
              │                    │         │                       │
              │  Conv2D(32)+BN+MP  │         │  Embedding(64-d)      │
              │  Conv2D(64)+BN+MP  │         │  BiLSTM(64)           │
              │  Conv2D(128)+BN+MP │         │  GRU(64)              │
              │  GlobalAvgPool     │         │  Dropout(0.4)         │
              │  Dropout(0.4)      │         │  Dense(128)           │
              │  Dense(128)        │         │  ← bottleneck         │
              │  ← bottleneck      │         │                       │
              │  Dense(8, softmax) │         │  Dense(8, softmax)    │
              └─────────┬──────────┘         └────────────┬──────────┘
                        │                                 │
                        │      ┌──────────────────┐       │
                        │      │  EARLY FUSION    │       │
                        └─────►│  Concatenate     │◄──────┘
                               │  (128 + 128=256) │
                               │  Dense(128)+Drop │
                               │  Dense(64)       │
                               │  Dense(8,softmax)│
                               └──────────────────┘

                        LATE FUSION (no new training)
                        ├── Average:   (P_audio + P_text) / 2
                        ├── Weighted:  0.7×P_audio + 0.3×P_text
                        └── Max Rule:  argmax(conf_audio vs conf_text)
```

---

## 2. Architecture Decisions & Rationale

### Audio CNN

| Decision | Reason |
|---|---|
| Mel-Spectrogram input | Converts raw waveform into a 2-D image the CNN can treat like a picture; mel scale matches human hearing |
| dB conversion + [0,1] normalisation | Log-compresses dynamic range, making values more Gaussian and training more stable |
| 3 s clip length | Balances coverage of emotion cues with memory cost; silence trimmed first with `librosa.effects.trim` |
| 3 Conv blocks (32→64→128 filters) | Increasing filters capture progressively abstract features — edges → textures → patterns |
| BatchNormalization after each Conv | Prevents internal covariate shift; allows higher learning rate |
| GlobalAveragePooling instead of Flatten | Reduces parameter count and overfitting vs. a flat dense layer after a large feature map |
| Dropout(0.4) | Regularisation before the bottleneck dense layer |
| ReduceLROnPlateau | Automatically halves the learning rate when val loss plateaus, avoiding stale training |
| Class weights (balanced) | RAVDESS has fewer neutral clips (half the repetitions); class weights prevent the model ignoring minority classes |

### Text RNN

| Decision | Reason |
|---|---|
| Whisper (tiny) for transcription | Open-source, runs on CPU, no API key required; cached after first run |
| Simple whitespace tokeniser | RAVDESS transcripts are clean English — no need for a heavy NLP pipeline |
| Bidirectional LSTM → GRU | BiLSTM reads context in both directions; GRU compresses to a fixed-length summary cheaply |
| Embedding dim = 64 | Small vocab (~few hundred unique words) doesn't need large embeddings |
| MAX_LEN = 30 tokens | RAVDESS sentences are short (≤10 words); 30 is generous padding |

### Early Fusion

| Decision | Reason |
|---|---|
| Both branches kept trainable | Freezing them (as is common) lets the useless text branch "lock in" bad weights; keeping them trainable lets the fusion head re-weight each modality |
| Concatenate bottlenecks (256-d) | Preserves the full feature set from both modalities before the fusion layers compress them |
| Dense 256→128→64→8 | Gradual compression forces the network to distil the joint representation |
| Lower LR (5e-4) than unimodal | Fine-tuning two pre-trained branches simultaneously needs a smaller step size |

---

## 3. Dataset Challenges

**The central challenge with RAVDESS for text modelling:** every actor speaks exactly two sentences regardless of emotion:
- *"Kids are talking by the door"*
- *"Dogs are sitting by the door"*

Whisper therefore produces nearly identical transcripts for all 8 emotion classes. The Text RNN has no linguistic signal to learn from and collapses — predicting a single class for every sample (visible in the confusion matrix as a single bright column). This is expected and reflects a real-world limitation, not a bug.

**Consequence for fusion:** Late fusion methods that average audio and text probabilities are slightly hurt by the random text output. The weighted variant (0.7 audio / 0.3 text) mitigates this. Early fusion with trainable branches can partially recover by learning to ignore the text features.

---

## 4. Results Table

| Model | Test Accuracy | Macro F1 | Notes |
|---|---|---|---|
| Audio CNN | ~0.95 | ~0.95 | Strong; near-perfect diagonal in CM |
| Text RNN | ~0.13 | ~0.02 | Collapses to single class — expected (see §3) |
| Early Fusion | ~0.94–0.96 | ~0.94–0.96 | Matches or slightly exceeds Audio CNN |
| Late Fusion (Avg) | ~0.88–0.92 | ~0.88 | Hurt by random text output |
| Late Fusion (Weighted) | ~0.91–0.94 | ~0.91 | Better than avg; audio dominates |
| Late Fusion (Max Rule) | ~0.91–0.94 | ~0.91 | Similar to weighted |
| Augmented Audio CNN | ~0.93–0.96 | ~0.93 | Robust to noise; comparable to base CNN |

*Exact numbers will vary by run; values above are representative of the trained models in this submission.*

---

## 5. Training & Validation Loss Plots

Plots are saved as PNG files alongside the notebook:

| File | Description |
|---|---|
| `audio_cnn_training.png` | Audio CNN loss & accuracy curves (50 epochs) |
| `text_rnn_training.png` | Text RNN loss & accuracy curves |
| `early_fusion_training.png` | Early Fusion loss & accuracy curves |
| `audio_cnn_aug_training.png` | Augmented Audio CNN curves |

**Audio CNN behaviour (observed):** Validation loss starts high (~10) and drops sharply after epoch 5 as the model escapes random initialisation. Training and validation accuracy converge to ~0.95–0.99 by epoch 30, with some oscillation in validation due to the small val set size (~144 samples).

**Text RNN behaviour (observed):** Both train and val accuracy hover near 12.5% (random chance for 8 classes), confirming the model cannot extract emotion from content-identical transcripts.

---

## 6. Confusion Matrix Analysis

### Audio CNN
Nearly perfect diagonal. Small confusions:
- **Disgust → Angry** (2 samples): acoustically similar — both involve strong, negative vocalisation
- **Sad → Fearful/Happy** (few samples): quieter, slower speech patterns overlap

### Text RNN
Single bright column (all predictions = one class). Consequence of identical transcript content across emotions — see §3.

### Early Fusion
Matches Audio CNN closely. With trainable branches, the fusion head learns to rely almost entirely on the audio features.

### Late Fusion (Average)
Matches Audio CNN on most classes but shows slightly more confusion in sad/neutral because the random text probabilities add noise to the confident audio output.

---

## 7. Bonus — Data Augmentation

Two augmentation strategies applied to training spectrograms:

1. **Gaussian noise** (`σ = 0.005`): simulates background acoustic noise
2. **SpecAugment** (time + frequency masking): randomly zeros out a band of time frames and a band of frequency bins, forcing the model to classify from partial information

Training set is doubled (original + augmented). The augmented CNN achieves comparable accuracy to the base model, confirming robustness.

---

## 8. Bonus — DistilBERT (commented out)

Section 7 of the notebook contains a fully written DistilBERT implementation (commented out). It replaces the BiLSTM with `distilbert-base-uncased` fine-tuned for sequence classification. Given the RAVDESS text limitation described in §3, DistilBERT is not expected to outperform the LSTM — both receive the same content-identical transcripts.

To enable: uncomment Section 7 and run `pip install transformers`.

---

## 9. File Index

| File | Description |
|---|---|
| `emotion_recognition.ipynb` | Main notebook (all phases) |
| `best_audio_cnn.keras` | Saved Audio CNN weights |
| `best_text_rnn.keras` | Saved Text RNN weights |
| `best_early_fusion.keras` | Saved Early Fusion weights |
| `best_audio_cnn_aug.keras` | Saved Augmented CNN weights |
| `transcripts.npy` | Cached Whisper transcriptions |
| `spectrograms_sample.png` | Sample mel-spectrograms |
| `audio_cnn_training.png` | Audio CNN training curves |
| `text_rnn_training.png` | Text RNN training curves |
| `early_fusion_training.png` | Early Fusion training curves |
| `audio_cnn_aug_training.png` | Augmented CNN training curves |
| `model_comparison.png` | Accuracy bar chart across all models |
| `cm_audio.png` | Confusion matrix — Audio CNN |
| `cm_text.png` | Confusion matrix — Text RNN |
| `cm_early.png` | Confusion matrix — Early Fusion |
| `cm_late_avg.png` | Confusion matrix — Late Avg |
