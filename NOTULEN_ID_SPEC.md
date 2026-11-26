# Notulen ID v2 — handoff spec for the new repo

Indonesian-only, noise-robust, in-browser meeting transcription.

This document is the handoff for a NEW repo built from zero. Sections 1–13 are the spec and are saved as `/home/eve/project-speech/NOTULEN_ID_SPEC.md` (path confirmed by the user). The spec says everything that must be built and why. The new AI session may also read the v1 notebook, read-only, as a source of code to copy (section 11); nothing in this spec depends on it.

## 1. Context

- **Product.** "UKM Live Notulen": a real-time meeting-minutes (notulen) tool for Indonesian student clubs (UKM = Unit Kegiatan Mahasiswa). It transcribes a meeting live, labels who spoke, and must keep working when the meeting happens in a noisy place.
- **Where the model runs (hard constraint, drives every technical choice).** In the user's browser through Transformers.js (WebGPU, WASM fallback) with ONNX weights downloaded once from the Hugging Face Hub and cached. No external ASR API.
- **Who builds what.** The AI builds ONE notebook (data, training, evaluation, export). The user runs it in Google Colab on a T4 GPU and pastes errors back. The web app (Next.js on Vercel) is the user's own work and is out of scope for the AI.
- **Origin.** v1 (`poevy/rapatin-speech-transcript`) was the final project for COMP6822001 Speech Recognition. The course rewards from-scratch fundamentals, and the user chose to keep those parts.
- **What v2 changes (only these two, decided by the user on 2026-11-26):**
  1. Indonesian only. No code-switching, no English data, no synthetic ID+EN concatenation.
  2. No translation feature. v1's translation layer (MarianMT, Xenova, sacrebleu) is dropped.
- **Consequence.** Noise robustness moves from a side experiment to a core requirement.
- **Other user decisions for v2.** Deployment stays in-browser. Keep the manual MFCC + 13-vs-12 experiment, the 3-model comparison, the streaming simulation and speaker labels. Cover babble, overlapping speech and environmental noise (room echo not required). Public data only; no own recordings.

## 2. Requirements

| # | Requirement |
|---|---|
| R1 | Transcribe Indonesian meeting speech from a single microphone, live. |
| R2 | Robust to three noise types, trained AND tested: babble, overlapping speech, environmental. |
| R3 | Runs fully in the browser: only Transformers.js-supported models small enough for it (Whisper-tiny 39M, Whisper-base 74M, Wav2Vec2-base 95M). |
| R4 | Notebook contains the fundamentals: manual MFCC, 13-vs-12 experiment, WER/CER/RTF, error analysis. |
| R5 | Compare three fine-tuned models on identical data and identical evaluation: whisper-tiny, whisper-base, wav2vec2-base CTC. |
| R6 | Streaming simulation: partial transcripts, finalization latency, RTF. |
| R7 | Speaker labels ("who said what"). |
| R8 | Export each model to ONNX (q8), verify it, upload to the Hub. |
| R9 | Public data only. |

Out of scope: web app, translation, code-switching, room echo, own recordings.

**What "noise proof" must mean, measurably:**

- (a) For each Whisper model, fine-tuned WER at or below its pretrained WER on every noise condition, judged with bootstrap confidence intervals (section 7.7). Wav2Vec2 has no useful pretrained baseline, so for it the evidence is its own noisy-versus-clean gap compared with the Whisper models'.
- (b) Whisper-base trained with noise augmentation beats whisper-base trained clean-only under noise, and the cost on clean WER is small and reported.
- (c) Results are WER-versus-SNR curves per noise type, on noise that was held out from training, and the notebook prints sample hypotheses for noisy conditions. v1 never looked at its noisy outputs and could not explain them.

## 3. Data

### 3.1 Indonesian speech for training

All of it is read speech recorded in quiet conditions. Sizes and licences were checked against primary pages on 2026-11-26 (section 12).

| Source | Size | Style | Licence | Access |
|---|---|---|---|---|
| FLEURS `id_id`, train split | 2,579 clips, about 9.1 h | read sentences | CC-BY 4.0 | ungated on Hugging Face: `google/fleurs` |
| Indspeech_Teldialog LVCSR | 40,000 clips, about 36 h, 400 speakers | read telephone-style sentences | CC BY-NC-SA 4.0 | GitHub `s-sakti/data_indsp_teldialog_lvcsr` |
| Indspeech_News LVCSR | 44,000 clips, about 43 h, 400 speakers | read news sentences | CC BY-NC-SA 4.0 | GitHub `s-sakti/data_indsp_news_lvcsr` |

- Together about 88 h from about 800 speakers. v1 had about 9 h of Indonesian.
- FLEURS has no speaker id column (only `gender`). The Indspeech corpora have speaker labels; confirm their format when loading. Every split that needs speakers (test, validation, interferers) is built from the Indspeech corpora.
- **Licence rules for the Indspeech corpora:** non-commercial, share-alike, re-hosting forbidden. Never upload that audio to the Hub or to a public Drive folder. Name the licence in the model card, and check the course's data-use policy before publishing weights trained on it.
- **No spontaneous speech available.** No public Indonesian corpus of spontaneous, conversational or meeting speech with a usable licence was found (only Javanese and Mandar sets, non-commercial). So the style gap between read sentences and real meeting talk is a known limitation for the report. Acoustic robustness is what noise training can address.
- **Common Voice is not a dependency.** Mozilla removed it from Hugging Face (the dataset pages are stubs) and serves it only through the Mozilla Data Collective: CC0, but it needs an account plus API credentials. Treat it as an optional extra if the user registers there.
- **Memory.** About 88 h of audio. Keep loading lazy: arrow cache, or file paths decoded inside the collator. Never decode all of it into RAM (Colab has about 12 GB).
- **Labels.** Whisper keeps the original casing and punctuation. CTC uses `norm(text)`: lowercase, no punctuation.
- **Length.** Drop or split clips longer than 30 s (Whisper's window).
- **Optional, helps the style gap.** For about 30% of the training data, join consecutive sentences of the same speaker into 10–25 s turns. Meeting turns are long; read sentences are short.

### 3.2 Speakers and splits

Assign whole speakers, never clips, to each role:

| Role | Where it comes from |
|---|---|
| Training | all FLEURS train; the Indspeech speakers not listed below |
| Validation (clean, for best-checkpoint selection) | FLEURS validation split (350 clips), subsampled to 150–200 clips; plus 3–5 held-out Indspeech speakers per corpus |
| Validation (noisy, fixed seed, for tuning noise hyperparameters and decode arguments) | the same validation clips, corrupted once per condition |
| Test speakers | about 10 held-out Indspeech speakers per corpus, plus the FLEURS test split |
| Interferer speakers for training noise | drawn from training speakers |
| Interferer speakers for test noise | about 10 further held-out speakers per corpus, different from test speakers |

- No test clip and no test speaker may appear in training, validation or the training noise.

### 3.3 Test sets (`TEST_SETS`)

- `fleurs_id`: FLEURS `id_id` test split (687 clips, about 2.4 h), capped at 200 clips.
- `indspeech_holdout`: all clips of the held-out Indspeech test speakers.
- `cv_id`: only if the Mozilla Data Collective route is used.
- Report the clip counts.

### 3.4 Noise banks, each split into a training part and a held-out test part

| Noise type | How it is built |
|---|---|
| Environmental | MUSAN noise subset (929 files, about 6 h, no intelligible speech, 16 kHz, CC BY 4.0; check the per-folder licences) plus the DEMAND 16 kHz zips (17–18 environments including OMEETING, OOFFICE and OHALLWAY, about 1.8 GB). DEMAND's licence is listed as CC BY-SA 3.0 on its page and CC BY 4.0 in its Zenodo metadata: cite both. |
| Babble | synthesised: 4–8 Indonesian speech clips, random offsets, different speakers when labels allow |
| Overlap | 1–2 interfering Indonesian utterances placed so they overlap at least half of the target clip |

- Training augmentation draws only from the training parts: about 80% of the environmental files (or 14 of the DEMAND environments) and the training interferer speakers.
- Test conditions use only the held-out parts: the remaining environmental files and environments, and the test interferer speakers. This keeps test noise out of the training distribution.
- The interferer pool kept in memory is about 300 clips.
- Not used: WHAM! (non-commercial, 17–35 GB) and FSD50K (44.1 kHz only, about 25 GB, mixed licences).

## 4. Noise model (functions to write)

- `add_noise(audio, kind, snr_db, rng, bank)` returns float32. `kind` is one of `babble`, `overlap`, `env`. `bank` is the training or the test bank (section 3.4).
- SNR is defined on speech-active frames: 10·log10(mean power of the clean target over its speech frames ÷ mean power of the noise over the same frames). The speech frames come from `vad_from_c0` run on the clean target.
- Noise starts at a random offset. If the mixture's peak exceeds 1.0, scale the whole mixture down. That is safe: Whisper's log-mel is invariant to overall gain (log10 of power, clamped relative to the clip's own maximum, and `do_normalize` defaults to false in transformers 4.46.3), and Wav2Vec2's extractor normalises to zero mean and unit variance.
- `make_noisy_set(ds, cond, seed)` builds the noisy copy of a test set. It is deterministic: one seed per (test set, condition).
- **Evaluation loop order:** test sets and conditions on the outside, models on the inside. Build each noisy set once, evaluate every loaded model on it, then discard it. Only one noisy set is in memory at a time. Load all models once, in fp16 (about 1.5 GB together).
- Condition grid: `clean`, plus each noise type at 20, 10 and 5 dB. 0 dB is optional.

**Training-time augmentation (defaults, tuned on the noisy validation set):**

- With probability `p_noise = 0.6` a sample is corrupted, otherwise it stays clean.
- Noise type is uniform; SNR is uniform in [0, 20] dB.
- Applied on the fly inside the data collator. Nothing is written to disk.

## 5. Models and training

Fairness framing: same dataset, same augmentation, same evaluation, same optimizer-step budget, with architecture-appropriate hyperparameters documented in a table in the notebook and the report.

| | Whisper-tiny | Whisper-base | Wav2Vec2-base CTC |
|---|---|---|---|
| Start | `openai/whisper-tiny` | `openai/whisper-base` | `facebook/wav2vec2-base` + new character head |
| Parameters | 39M | 74M | 95M |
| Learning rate | 1e-5 | 1e-5 | 1e-4 |
| Batch | 16 | 16 | 8 × grad-accum 2 (T4 memory) |
| Loss | seq2seq cross-entropy | same | CTC, reduction `mean` |
| Output | cased, punctuated | same | lowercase, no punctuation |
| Language | forced Indonesian | same | n/a |

- **Step budget.** Decide `MAX_STEPS` once, from whisper-base's validation-WER curve (start at 1500, where v1 took 20–40 minutes per run), and apply the same value to all three models. Publish the validation-WER-versus-step curve of every model, so a reader can see whether one was still improving.
- **Checkpoint selection.** Evaluate and save every 500 steps; keep the 2 newest; reload the checkpoint with the lowest validation WER at the end.
- **Decoding must be consistent.** The Trainer's own evaluation generates from `model.generation_config`, not from `GEN_KW`. So in every fine-tune cell set `generation_config.language`, `.task = "transcribe"`, `forced_decoder_ids = None`, and copy the kept decode arguments (section 7.5) into `generation_config`. Otherwise the best checkpoint is selected with a different decoder than the one reported.
- **Ablation** for requirement (b), whisper-base only: the same recipe with `p_noise = 0`.
- Save the best checkpoint to `/content` and to Drive as soon as training ends.
- Note for the report: Wav2Vec2-base starts from English-centric pretraining, so this is a paradigm comparison, not a contest against Indonesian-pretrained checkpoints.
- Possible stronger start for the CTC model, to evaluate, not assumed: `huyisme-005/wav2vec2-base-10lang-ssl` (Apache-2.0, base size, self-supervised on 10 languages including Indonesian; a community upload without a CTC head).
- `indonesian-nlp/wav2vec2-indonesian-javanese-sundanese` (Apache-2.0, CTC for id/jv/su) can give a reference WER in the report if its size fits Colab. It is not a browser candidate.

## 6. Notebook outline

One notebook, comments and markdown in Indonesian, run top to bottom on Colab with a T4.

1. **Setup.** Pinned packages, config block, Drive mount.
2. **Data.** The three corpora, speaker roles, `TEST_SETS`, validation sets.
3. **Noise.** Noise banks and their train/test split, `add_noise`, plots of one clip clean versus each noise type.
4. **MFCC from scratch.** Manual NumPy pipeline, validated against librosa, plus the online `vad_from_c0`.
5. **13 vs 12 coefficients.** Redesigned experiment (section 7.2).
6. **Baseline.** Pretrained whisper-tiny and whisper-base on clean and noisy test sets. Defines the shared harness: `norm`, `load_whisper`, `transcribe`, `evaluate`, `GEN_KW`.
7. **Fine-tune.** whisper-base, the clean-only ablation, whisper-tiny, wav2vec2-CTC.
8. **Evaluation.** Combined table with confidence intervals, curves, sample hypotheses, error analysis (section 7.7).
9. **Streaming simulation.** Redesigned loop (section 7.3).
10. **Speaker labels.** Redesigned tracker (section 7.4).
11. **Export.** ONNX + q8 + verification + Hub upload, per model (section 8).

## 7. Redesigns required (v1 flaw → v2 behaviour)

### 7.1 Validation hygiene
- v1: the validation set was the first 100 of the 150 test clips, and it was noise-augmented, because the Trainer uses ONE collator for training and evaluation (transformers 4.46.3). Best-model selection was contaminated.
- v2: validation comes from held-out splits and speakers. Each dataset gets an `is_eval` column. With `remove_unused_columns=False` the collator receives every column, so it reads `is_eval` from each example, augments only when it is false, and returns only the model inputs (`input_features` or `input_values`, plus `labels`). The extra columns never reach the model.

### 7.2 13 vs 12 experiment
- v1: speaker identification scored 1.000 for both feature sets on 30 English LibriSpeech clips. Ceiling, no conclusion.
- v2: use the Indspeech speaker labels. Pick 20–30 held-out speakers with at least 8 clips each. Run identification (diagonal GMM per speaker, as v1) and verification with same-speaker and different-speaker pairs (hundreds of pairs, cosine on the embedding of section 7.4). Compare `c1..c12` with `c0..c12` under clean, babble 10 dB, and 2-second clips. Report accuracy and equal error rate. The conclusion may be either way; "c0 does not help speaker ID but is needed for VAD" is a valid result.

### 7.3 Voice activity detection and streaming

**VAD, specified.** c0 from `mfcc_manual` is log frame energy up to a scale factor.

- For each 0.5 s chunk compute `p10 = percentile(c0, 10)`. The noise floor follows it down quickly and up slowly, with v1's rule: `0.7*floor + 0.3*p10` when `p10 < floor`, otherwise `floor + 0.02*(p10 - floor)`.
- A frame is speech when `c0 > floor + margin` (margin 3.0 to start). Speech runs shorter than `min_frames = 5` are dropped.
- Compute c0 once per chunk and keep it with the segment. v1 recomputed the features over the whole buffer on every call.

**Loop.** `chunk_s = 0.5`, `update_s = 1.0`, `silence_s = 0.6`, `max_seg_s = 20`.

- A segment opens when speech is detected and closes after `silence_s` of trailing silence.
- Partials run every `update_s` on the segment's audio only (grow-only).
- At the cap, cut at the lowest-energy frame in the last 2 s and carry the remainder into the next segment. v1 cut mid-sentence.
- Guards: call the model only when the segment holds at least 0.5 s of speech; drop empty outputs.

**v1 flaws this fixes:** `silence_s` of 0.6 and 1.0 behaved identically; the reported final latency excluded the silence wait; the first partial used 0.5 s of audio and was often a hallucination; a 9 s utterance caused 11 Whisper calls covering 60 s of audio.

**Metrics:** partial latency, finalization latency measured from the END of speech including the silence wait, RTF, model calls per minute of audio, false activations (a finalized non-empty segment from input with no target speech).

**Limitation to state:** an energy VAD cannot tell speech from babble; under babble the floor rises and detection degrades. So the false-activation target of 0 applies to silence and to non-speech environmental noise only. Babble-only input is measured and reported, not targeted.

**Tests whose numbers are printed:** `silence_s` 0.6 versus 1.0 gives different finalization times; an input with babble at 10 dB is still segmented into the right number of segments.

### 7.4 Speaker labels
- v1: the tracker (cosine threshold 0.6) labelled every segment "Speaker 1".
- v2 embedding: mean and std of manual MFCCs over voiced frames, after z-normalising each coefficient with statistics computed on the calibration speakers (not per utterance: per-utterance mean removal would zero the mean). The embedding must be normalised so that cosine similarity is meaningful.
- v2 threshold: calibrated on real same-speaker and different-speaker pairs from the validation speakers (the value at the equal error rate), not guessed. Calibration speakers are never the speakers used for scoring.
- v2 online rule: assign to the nearest speaker centroid above the threshold, otherwise create a new speaker; require 1 s of speech before assigning.
- Evaluation: a synthetic meeting built from clips of N = 5 held-out test speakers. Report segment labelling accuracy after optimal one-to-one mapping of predicted to true labels (`scipy.optimize.linear_sum_assignment`), and speakers found versus true.

### 7.5 Decoding arguments
- v1's anti-hallucination arguments (`repetition_penalty=1.2`, `no_repeat_ngram_size=3`, `no_speech_threshold=0.6`, `logprob_threshold=-1.0`) were never benchmarked.
- v2: measure WER with and without them on the clean and the noisy VALIDATION sets only, keep what helps, and never tune on test.

### 7.6 Artifacts
- v1 saved evaluation CSVs only to `/content`, and they were lost on a runtime reset.
- v2: every result file (CSVs, models, ONNX folders) is copied to Drive in the same cell that creates it.

### 7.7 Evaluation protocol
- All systems, Whisper and CTC, are scored on `norm()`: WER would not be comparable otherwise, because CTC outputs no case or punctuation.
- Every WER gets a 95% bootstrap confidence interval (resample clips 1,000 times). Requirement (a) fails only when the fine-tuned model is worse beyond the interval; borderline results are reported as such.
- For every condition print 5 reference/hypothesis pairs.
- Error analysis (`error_table`: substitutions, deletions, insertions, most-confused words) runs on clean and on the worst noise condition.

## 8. Export and deployment

- Stack that worked in v1: `optimum` `main_export`, then `onnxruntime` dynamic quantization to uint8, then the Transformers.js folder layout, then upload.
- Export task strings: `automatic-speech-recognition-with-past` for Whisper (gives `encoder_model.onnx` + `decoder_model_merged.onnx`); `automatic-speech-recognition` for Wav2Vec2-CTC (single `model.onnx`). The CTC export was never run in v1.
- Whisper layout: root holds `config.json`, `generation_config.json`, tokenizer files and `preprocessor_config.json`; `onnx/` holds `encoder_model.onnx`, `decoder_model_merged.onnx` and their `_quantized` copies.
- Wav2Vec2-CTC layout: root holds `config.json`, `preprocessor_config.json`, `vocab.json`, `tokenizer_config.json`, `special_tokens_map.json`; `onnx/` holds `model.onnx` and `model_quantized.onnx`.
- Verify each export: WER of ONNX versus PyTorch on 30–50 clips, for the fp32 files AND for the q8 files.
- **Known defect to fix first.** In v1 the q8 step did not shrink the decoder at all: whisper-tiny 118.6 MB versus 118.5 MB fp32; whisper-base 208.6 MB. The encoder was fine (32.9 → 10.1 MB).
  - The token-embedding table dominates the decoder (about 80 of tiny's 118.5 MB, about 106 of base's 208.6 MB) and is read by a Gather op, which `quantize_dynamic` leaves in float32. Even with every MatMul quantised, the decoder cannot go below about 90 MB (tiny) and 130 MB (base) with that method.
  - So v1's 118.6 MB means no MatMul was quantised either. Diagnose by counting the quantised nodes in the decoder graph and comparing with the encoder.
  - Smaller files are possible: v1's Hub repo holds a 79.9 MB base decoder, below that floor. Find out how it and the public `onnx-community/whisper-*` decoders were produced (another quantization path, or weights stored as fp16/int8) and reproduce it.
  - Acceptance: the decoder's MatMul weights are really quantised; every file size is reported; q8 WER is within a few points of fp32 WER.
- Hub repos, public (the browser must download anonymously), names confirmed by the user: `{username}/whisper-tiny-notulen-id`, `{username}/whisper-base-notulen-id`, `{username}/wav2vec2-base-notulen-id`. The username comes from the login token. Add a short model card after each upload; it must name the training data and licences.
- Upload flow that worked: `login(new_session=True)` (the user pastes a Write token in the Colab prompt), `HfApi().whoami()["name"]`, `create_repo(..., exist_ok=True)`, `upload_folder(folder_path=<Drive folder>, repo_id=...)`.
- Latency in Colab is measured on a GPU. The real-time numbers that matter are measured in the browser by the user.

## 9. Environment facts and library pitfalls

- **Local machine (AI side).** WSL2, no GPU, 6 GB RAM, only numpy and scipy installed. The AI cannot run training, evaluation or ONNX tools. Every new cell is untested until the user runs it in Colab.
- **Colab.** T4 with about 14.5 GB GPU RAM and about 12 GB system RAM, Python 3.13. Runtimes get lost often (RAM exhaustion, idle disconnects).
- **Pinned stack (keep unless there is a reason).** `transformers==4.46.3`, `datasets==3.1.0`, `optimum[onnxruntime]==1.23.3`, `accelerate==1.1.1`, plus `jiwer librosa soundfile scikit-learn`. Then `pip uninstall -y diffusers`.
- pip's dependency-conflict warnings about `gradio`, `gradio-client`, `gcsfs` and `diffusers` are harmless. Only if the import cell fails: Runtime → Restart session, then continue from the next cell.
- **Symptom → fix list, all seen in Colab:**
  1. `TypeError: '>' not supported between 'NoneType' and 'float'` inside `_retrieve_avg_logprobs`: pass `temperature=0.0` whenever `logprob_threshold` or `no_speech_threshold` is set.
  2. Do not set `forced_decoder_ids`; set it to `None` on the model and pass `language=` and `task=` to `generate`.
  3. The "attention mask is not set ... pad token is same as eos token" warning is harmless.
  4. Wav2Vec2: `facebook/wav2vec2-base` ships `vocab_size=32` and no head, so loading it plainly gives `Label values must be <= vocab_size: 32`. Load with `vocab_size=len(ctc_tok), ctc_loss_reduction="mean"`. The "Some weights ... were not initialized: lm_head" warning is expected.
  5. Wav2Vec2 runs out of GPU memory at batch 16 with long clips (failed in `group_norm`). Use batch 8 and `gradient_accumulation_steps=2`.
  6. Wav2Vec2-base feature extractor: `do_normalize=True`, `return_attention_mask=False`. Build labels with `filter` and `map` on the text column only (`input_columns=["text"]`), and decode labels with `group_tokens=False`.
  7. Before a training cell, free GPU memory by deleting the previous fp32 training objects (`model`, `trainer`, processors) and calling `gc.collect()` and `torch.cuda.empty_cache()`. Do not delete objects that later sections need.
  8. ONNX export: set `optimum.utils.import_utils._diffusers_available = False`, remove `diffusers*` and `optimum.exporters*` from `sys.modules`, wrap `torch.onnx.export` so it always gets `dynamo=False`, load the model with `attn_implementation="eager"` into a temporary folder, then `main_export`.
  9. Every export, verify and upload cell must run on a fresh kernel: resolve paths with `globals().get(...)` and fall back to the Drive copy. Never assume a variable exists.
- **After a runtime reset:** `/content` and all Python variables are gone; Drive keeps whatever was copied there. The export path then needs only the setup cells and the export → verify → upload cells. A verify cell that fails with `Incorrect path_or_model_id: '/content/...'` means the export cell was not run in that session.

## 10. Working rules between user and AI

- The user's open Colab copy of the notebook can lag behind GitHub and cannot be reloaded mid-session. For every error the user pastes, give three things: the exact in-session workaround (a new cell or a one-line edit), the fix committed and pushed to GitHub, and which cells must be re-run after a reset.
- Diagnose from the traceback and the pinned library source before editing.
- Git: the AI edits the notebook locally and runs `git pull` first. The user may also edit in Colab ("Save a copy in GitHub"), so pull before every local edit. Never force-push. Commit and push only with the user's approval; the user has approved pushing agreed fixes to `main`.
- Notebook JSON: never reformat. Before writing, load the file, dump it with `json.dumps(nb, indent=2, ensure_ascii=False)` and confirm the bytes are identical (v1 had no trailing newline). After editing, `git diff --numstat` must show only the intended lines.
- Build in stages, one model end to end first (whisper-base), let the user confirm it in Colab, then whisper-tiny, then wav2vec2, then the combined table and the exports.
- Notebook text in Indonesian. Chat with the user in English, briefly and concretely: the user asks "what should I fix and where?" and wants exactly that.

## 11. Old repo: what to reuse, what to drop

- v1 notebook: `/home/eve/project-speech/rapatin-speech-transcript/UKM_Live_Notulen_ASR_(2).ipynb`, also on GitHub at `https://github.com/poevy/rapatin-speech-transcript`. Local handoff: `PROJECT_CONTEXT.txt` in that folder. Reference only; the new repo starts empty.
- **Ran in Colab, copy and adapt:**
  - `load_fleurs` (as is), `norm`, `load_whisper`, `transcribe`.
  - `evaluate`: change it to take ready-made audios and references, so that noise is applied outside it.
  - `hz_to_mel`, `mel_to_hz`, `mel_filterbank`, `mfcc_manual` (as is; validated against librosa in v1), `vad_from_c0` (take the floor as an argument, section 7.3).
  - `Collator`, `compute_metrics` (whisper-base and whisper-tiny trained with them), `CollatorCTC`, `compute_metrics_ctc` (training started and ran; completion of all steps was not reported). Both collators need `is_eval` and `add_noise`.
  - `simulate_stream` and `SpeakerTracker` (ran, with the flaws in sections 7.3 and 7.4: rewrite the internals, keep the signature idea).
  - `error_table` (as is), and the Whisper export, verify, quantize and upload cells.
- **Written in v1 but never run on a GPU, treat as untested:** `transcribe_ctc`, `evaluate_w2v2`, `load_w2v2`, `model_dir`, and every Wav2Vec2 ONNX cell.
- **Drop:** `make_codeswitch`, the English FLEURS loader, `COMPARE_MODEL`, `NOISE_BANK`, `add_babble`, `load_speaker_data` and `speaker_id_accuracy` (replaced by section 7.2), the whole translation section.

## 12. Dataset facts

### Checked on 2026-11-26 against primary pages

- **FLEURS `id_id`**: train 2,579 clips (about 9.1 h), validation 350 (about 1.2 h), test 687 (about 2.4 h). CC-BY-4.0. No speaker id column. https://huggingface.co/datasets/google/fleurs
- **Common Voice**: removed from Hugging Face. `mozilla-foundation/common_voice_17_0` is a 2.88 kB stub that points to the Mozilla Data Collective (exclusive since October 2025). Latest release there: Scripted Speech v27.0, 21 September 2026. CC0. Download needs an MDC account plus API credentials; the free tier allows 7 downloads a day. Speakers appear in only one split (`client_id`). https://datacollective.mozillafoundation.org/
- **Indspeech_Teldialog LVCSR**: 36.15 h, 40,000 utterances, 400 speakers, read telephone-domain sentences, clean, 16 kHz, CC BY-NC-SA 4.0, re-hosting forbidden. https://github.com/s-sakti/data_indsp_teldialog_lvcsr
- **Indspeech_News LVCSR**: 43.35 h, 44,000 utterances, 400 speakers, read news, clean, 16 kHz, CC BY-NC-SA 4.0, re-hosting forbidden. https://github.com/s-sakti/data_indsp_news_lvcsr
- **Not Indonesian**: Indspeech_News_EthnicSR is Javanese and Sundanese. CML-TTS has no Indonesian.
- **MUSAN**: about 109 h in total, 16 kHz; the noise subset is 929 files, about 6 h, no intelligible speech. CC BY 4.0 with per-folder licences. No ready-made babble subset. https://www.openslr.org/17/
- **DEMAND**: 16 kHz zips exist for 17 of 18 environments, about 1.8 GB. https://zenodo.org/records/1227121
- **WHAM!**: CC BY-NC 4.0. **FSD50K**: 44.1 kHz only, mixed licences.

### Still to check in the new session

- How the two Indspeech corpora are downloaded (URL, archive size, file layout) and how their speaker labels are stored. If a runtime reset costs more than about 15 minutes of re-downloading, cache them on Drive.
- Indonesian validated hours in Common Voice v27 (the catalogue is JavaScript-only), and whether the MDC "Spontaneous Speech" set includes Indonesian. If it does, add it: it would be the only non-read Indonesian speech.
- How the small quantized Whisper decoders on the Hub were produced (section 8).
- Whether `huyisme-005/wav2vec2-base-10lang-ssl` loads with transformers 4.46.3, and the parameter count of `indonesian-nlp/wav2vec2-indonesian-javanese-sundanese`.
- Whether the course allows publishing weights trained on CC BY-NC-SA data.

## 13. Build order and definition of done

Build order: setup + data → noise → MFCC → baseline → whisper-base fine-tune and ablation → evaluation table → 13 vs 12 → streaming → speaker labels → whisper-tiny → wav2vec2 → exports.

Done means:

- The notebook runs top to bottom on a fresh Colab T4 (confirmed by the user), and every export, verify and upload cell runs standalone after a reset.
- The combined table exists with its CSV on Drive, covering pretrained and fine-tuned models on every test set and noise condition, with confidence intervals.
- Requirement (a) holds for every condition, or the failing condition is diagnosed and reported with sample hypotheses.
- The 13-vs-12 experiment reports numbers under at least one hard condition.
- Speaker labels are evaluated with numbers on a 5-speaker meeting.
- Streaming reports zero false activations on silence and on environmental noise, and shows that `silence_s` changes behaviour.
- Three public Hub repos hold q8 models that passed the checks in section 8.
