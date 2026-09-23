# Module 04 — Model Optimization Toolkit

**Status:** In Progress (started out of curriculum order, driven by questions while reading the quantization blog)
**Related:** Llama.cpp, Microsoft Olive, OpenVINO, Apple MLX — full toolkit coverage still to come

## Key Concepts

### What quantization actually is (and isn't)

Quantization does **not** reduce the number of parameters — a 7B model stays a 7B model. It reduces how much storage **each parameter** takes, by converting the data type each number is stored as:

- Base precision: usually **FP16 or FP32** (floating point) — how models are trained
- Quantized precision: **INT8 / INT4** (rarely INT16) — how models are compressed for deployment

Same count of numbers, each one stored more cheaply. That's why VRAM/RAM footprint shrinks (e.g. a 7B model: ~14GB at FP16 → ~4GB at INT4) without deleting any parameters.

### Worked example: INT4 quantization, step by step

Take 5 real weights (FP16, precise): `-0.90, -0.42, 0.03, 0.31, 0.87`

**Step 1 — find the range:** min = -0.90, max = 0.87

**Step 2 — build the "ruler":** INT4 unsigned gives 16 possible values (2⁴ = 16, tick marks 0–15). Scale = (max − min) / 15 = 1.77 / 15 ≈ **0.118** — the distance between adjacent tick marks.

**Step 3 — the ruler:**
```
tick:   0     1     2     3     4     5     6     7     8     9    10    11    12    13    14    15
value: -0.90 -0.78 -0.66 -0.54 -0.43 -0.31 -0.19 -0.07  0.04  0.16  0.28  0.39  0.51  0.63  0.75  0.87
```

**Step 4 — snap each real weight to the nearest tick:**

| Original (FP16) | Nearest tick # | Stored as (4-bit) | Reconstructed value | Error |
|---|---|---|---|---|
| -0.90 | 0 | `0000` | -0.90 | 0.00 |
| -0.42 | 4 | `0100` | -0.43 | 0.01 |
| 0.03 | 8 | `1000` | 0.04 | 0.01 |
| 0.31 | 10 | `1010` | 0.28 | 0.03 |
| 0.87 | 15 | `1111` | 0.87 | 0.00 |

**What's actually stored:** just the 4-bit tick number for each weight, plus **one shared scale factor** (0.118) for the whole group. To use a weight at inference time, it's reconstructed as `tick × scale + min`.

**Why quality degrades slightly:** values that don't land exactly on a tick get rounded — visible above as small errors (0.01–0.03). More bits (INT8) = more ticks = smaller error, at the cost of more memory. This is the core precision-vs-size tradeoff behind every quantization method (GPTQ, AWQ, GGUF, bitsandbytes).

### VRAM sizing formula (weight memory)

`Memory (bytes) = (n_bits / 8) × n_params`

Where **n_bits is the target precision you're quantizing TO** (not the original precision) — this is the number of parameters times how many bytes each one now takes.

**Worked example — 7B parameter model:**

| Precision | n_bits | Calculation | Weight memory |
|---|---|---|---|
| FP32 (original) | 32 | (32/8) × 7B | ~28 GB |
| FP16 | 16 | (16/8) × 7B | ~14 GB |
| INT8 (quantized) | 8 | (8/8) × 7B | ~7 GB |
| INT4 (quantized) | 4 | (4/8) × 7B | ~3.5 GB |

**Common mistake to avoid:** don't plug in the *original* precision (e.g. 32 for FP32) — n_bits is always the *target* precision you're converting to. The ratio between original and target (e.g. 32/8 = 4x) is a separate useful number — the **compression ratio** — but it multiplies the wrong way if used directly in the formula.

**Important catch — this formula is weight memory ONLY.** It does not include activation memory (the temporary numbers computed fresh per inference run — see below). Real total VRAM need = weight memory (this formula) **+** activation memory (additional, not a subdivision of the same total).

### Weight memory vs. activation memory

Both live in the same physical VRAM, but are fundamentally different kinds of content:

| | Weight memory | Activation memory |
|---|---|---|
| What it stores | Fixed, learned parameters (the "spreadsheet") | Temporary values computed during inference (`weight × input` + bias, per neuron) |
| When created | Once, at training time, saved to model file | Fresh, every single inference run — discarded after |
| Size formula | `(n_bits/8) × n_params` — fixed per model | Depends on sequence length, batch size, number of layers, hidden size — varies per request |
| On the ECU analogy | Calibration table (tuned once, stored) | Live computed output for the current engine cycle (recalculated every cycle, not stored) |

Activation memory doesn't have one fixed formula the way weight memory does — it scales with *how the model is being used* (longer input = more activation memory), not just which model it is.

### Symmetric vs. asymmetric quantization

Two independent axes of quantization design, easy to conflate — keep them separate:

**What gets quantized (two targets):**
1. **Weight quantization** — the fixed, learned parameters. Done once, frozen after.
2. **Activation quantization** — the computed outputs (`input × weight + bias`, from the activations concept above). Recalculated live, every inference run.

**How it gets quantized (two schemes), applying to either target:**
- **Symmetric** — forces the representable range to be centered on zero (using the largest magnitude value). Zero always lands exactly on a code, so no extra storage needed — just a scale factor. Simple, cheap in hardware, but wastes codes if the real data isn't actually centered on zero.
- **Asymmetric** — uses the real min/max directly, no forcing. Every code does useful work (finer precision), but zero usually doesn't land on a clean code anymore, so an extra number — the **zero-point** — has to be stored alongside the scale to know which code represents real zero.

**Full worked comparison (5 weights: `-0.20, -0.05, 0.03, 0.31, 0.87`), with two number-line diagrams:**
https://claude.ai/code/artifact/3bba9011-6d23-4675-aa69-b4bfc5b7097c

That example: symmetric forces the range to -0.90..0.90 (scale 0.1286), wasting 6 of 16 codes since the data never goes that low. Asymmetric fits -0.20..0.90 directly (scale 0.0733, zero-point at code 3), using all 16 codes for ~1.75x finer resolution — at the cost of storing that zero-point.

**Which is used where, in real deployments (both used simultaneously, not either/or):**
- **Weights → symmetric.** Trained weights naturally cluster loosely around zero, so waste is small, and it's faster/cheaper in hardware (no zero-point offset needed in the multiply). This is the default for weight quantization in GPTQ, AWQ, bitsandbytes.
- **Activations → asymmetric.** Activations after a ReLU (or similar) are entirely non-negative and heavily skewed — symmetric would waste half the range on negative values that never occur. So production pipelines typically quantize weights symmetrically *and* activations asymmetrically, in the same model, at the same time.

### Calibration — choosing the range, not just the scheme

Both diagrams above assumed the "real" min/max were already known. **Calibration is the step that decides what those min/max should be** — and the honest answer is usually *not* the literal min/max of the data.

**The problem:** real weight/activation distributions usually have a tight bulk near zero plus a few rare outliers far out at the edges. If you naively use the literal min/max (including the outliers), the whole 16-code range gets stretched to cover those rare extremes — wasting most of the resolution on territory almost no real value lives in.

**What calibration does instead:** run a small representative sample of real data through the model, look at where most values actually fall, and deliberately pick a *tighter* clipping range that fits the bulk — clamping the rare outliers to the nearest edge code (accepting some error for just those rare values) in exchange for much finer resolution everywhere else.

**Worked example (21 sampled weights, 2 outliers near ±1.9, 86% of values inside ±0.8), with histogram + before/after range diagrams:**
https://claude.ai/code/artifact/3bba9011-6d23-4675-aa69-b4bfc5b7097c (same page, "Part 2" section)

- Naive range (-1.9 to 1.9): scale = 0.253 per tick — coarse, most of the range wasted on empty territory
- Calibrated range (-0.8 to 0.8, outliers clipped): scale = 0.107 per tick — **~2.4x finer** for the 86% of values that actually matter

**One-sentence definition:** calibration = choosing the clipping range using real sample data, instead of blindly trusting the literal min/max, so a handful of rare outliers don't ruin resolution for everything else.

## Connects to what I already know
- Directly parallels the fbgemm/cuda quantization work already done on vision models in `sdv-edge-gateway` — same core idea (reduce numeric precision to shrink memory/compute), just applied to language model weights instead of CNN weights.
- Distinct from **pruning/distillation** — those actually reduce parameter *count*. Quantization only changes how each parameter is *stored*. Don't conflate the two.

## Open Questions
- How does the "shared scale factor per group" choice (group size) affect the precision/memory tradeoff in practice? (GPTQ/AWQ differ here — revisit once doing the hands-on resources.)

## Resources
- [A Visual Guide to Quantization — Maarten Grootendorst](https://www.maartengrootendorst.com/blog/quantization/) (started 23-Sep-26)
- [Eldar Kurtić — Beginner Friendly Introduction to LLM Quantization: From Zero to Hero](https://www.youtube.com/watch?v=LK2-lrLvhTA)
- [Which Quantization Method Is Best for You?: GGUF, GPTQ, or AWQ | E2E Networks](https://www.e2enetworks.com/blog/which-quantization-method-is-best-for-you-gguf-gptq-or-awq) — hands-on, next up
