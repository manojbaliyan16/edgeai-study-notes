# Module 04 - Model Optimization Toolkit

**Status:** In Progress (started out of curriculum order, driven by questions while reading the quantization blog)
**Related:** Llama.cpp, Microsoft Olive, OpenVINO, Apple MLX - full toolkit coverage still to come

## Key Concepts

### Quantization techniques, quick reference (from the course repo)

The course material lists four techniques:

- **Post-Training Quantization (PTQ):** applied after model training, no retraining needed
- **Quantization-Aware Training (QAT):** builds quantization effects into training itself, for better accuracy
- **Dynamic Quantization:** quantizes weights to INT8, calculates activations on the fly
- **Static Quantization:** pre-computes quantization parameters for both weights and activations

All four are covered below with worked examples (see PTQ vs. QAT and dynamic vs. static further down). One framing note worth flagging: the course lists PTQ/QAT and dynamic/static as four separate parallel techniques, but the mental model I've actually built is that dynamic/static are a choice *within* PTQ - both happen after training, they just differ in *when* the scale/zero-point gets computed. Same substance, just nested one level deeper than how the course presents it.

### What quantization actually is (and isn't)

Quantization doesn't reduce the number of parameters - a 7B model stays a 7B model. What it changes is how much storage each parameter takes, by converting the data type each number is stored as. Base precision is usually FP16 or FP32 (how models get trained); quantized precision drops to INT8 or INT4, rarely INT16 (how they get compressed for deployment). Same count of numbers, each stored more cheaply - that's why VRAM/RAM footprint shrinks (a 7B model goes from ~14GB at FP16 down to ~4GB at INT4) without deleting a single parameter.

### Worked example: INT4 quantization, step by step

Take 5 real weights (FP16, precise): `-0.90, -0.42, 0.03, 0.31, 0.87`

**Step 1 - find the range:** min = -0.90, max = 0.87

**Step 2 - build the "ruler":** INT4 unsigned gives 16 possible values (2⁴ = 16, tick marks 0-15). Scale = (max - min) / 15 = 1.77 / 15 ≈ 0.118, the distance between adjacent ticks.

**Step 3 - the ruler:**
```
tick:   0     1     2     3     4     5     6     7     8     9    10    11    12    13    14    15
value: -0.90 -0.78 -0.66 -0.54 -0.43 -0.31 -0.19 -0.07  0.04  0.16  0.28  0.39  0.51  0.63  0.75  0.87
```

**Step 4 - snap each real weight to the nearest tick:**

| Original (FP16) | Nearest tick # | Stored as (4-bit) | Reconstructed value | Error |
|---|---|---|---|---|
| -0.90 | 0 | `0000` | -0.90 | 0.00 |
| -0.42 | 4 | `0100` | -0.43 | 0.01 |
| 0.03 | 8 | `1000` | 0.04 | 0.01 |
| 0.31 | 10 | `1010` | 0.28 | 0.03 |
| 0.87 | 15 | `1111` | 0.87 | 0.00 |

What actually gets stored is just the 4-bit tick number per weight, plus one shared scale factor (0.118) for the whole group - a weight gets reconstructed at inference time as `tick × scale + min`. The small errors above (0.01-0.03) come from values that don't land exactly on a tick and get rounded off. More bits means more ticks means smaller error, at the cost of more memory - that tradeoff is the core idea behind every quantization method out there (GPTQ, AWQ, GGUF, bitsandbytes).

### VRAM sizing formula (weight memory)

`Memory (bytes) = (n_bits / 8) × n_params`

n_bits here is the target precision you're quantizing *to*, not the original precision - number of parameters times how many bytes each one now takes.

**Worked example - 7B parameter model:**

| Precision | n_bits | Calculation | Weight memory |
|---|---|---|---|
| FP32 (original) | 32 | (32/8) × 7B | ~28 GB |
| FP16 | 16 | (16/8) × 7B | ~14 GB |
| INT8 (quantized) | 8 | (8/8) × 7B | ~7 GB |
| INT4 (quantized) | 4 | (4/8) × 7B | ~3.5 GB |

Easy mistake to make: plugging in the *original* precision (32 for FP32) instead of the target. The ratio between the two (32/8 = 4x) is a real and useful number - the compression ratio - but it multiplies the wrong way if dropped straight into the formula.

One catch worth remembering: this formula is weight memory only. It leaves out activation memory entirely (the temporary numbers computed fresh on every inference run, covered below), so real total VRAM need is weight memory plus activation memory on top, not a subdivision of one total.

### Weight memory vs. activation memory

Both live in the same physical VRAM, but they're fundamentally different kinds of content:

| | Weight memory | Activation memory |
|---|---|---|
| What it stores | Fixed, learned parameters (the "spreadsheet") | Temporary values computed during inference (`weight × input` + bias, per neuron) |
| When created | Once, at training time, saved to model file | Fresh, every single inference run - discarded after |
| Size formula | `(n_bits/8) × n_params`, fixed per model | Depends on sequence length, batch size, number of layers, hidden size - varies per request |
| On the ECU analogy | Calibration table (tuned once, stored) | Live computed output for the current engine cycle (recalculated every cycle, not stored) |

Activation memory doesn't have a single fixed formula the way weight memory does - it scales with how the model is actually being used, not just which model it is.

### Scale and zero-point, plain and simple

Two words that show up everywhere in quantization, stripped down to the bare minimum. Scale is how much real-world distance one "box" (integer step) covers - the price per box, in real units. Zero-point is which box number represents the real value 0.

Simplest possible example: a ruler with only 5 marks (boxes 0-4) covering real values -10 to +10.

```
box:    0     1     2     3     4
value: -10   -5     0     5    10
                     ↑
              zero-point = 2
```

Total real range is 20, split across 4 gaps between the 5 marks, so scale = 20 ÷ 4 = 5 (each box-step is 5 real units). Real zero lands exactly on box 2, so zero-point = 2. To turn a stored box-number back into a real value: `real value = (box number - zero-point) × scale`. Check: box 3 → `(3-2)×5 = 5`.

This is the same scale/zero-point that shows up in the symmetric-vs-asymmetric diagrams below. Symmetric quantization always gets zero-point = 0 for free, since zero lands on a clean code by construction; asymmetric has to store whatever zero-point the real data's min/max actually produces.

### Symmetric vs. asymmetric quantization

Two independent axes of quantization design that are easy to conflate, so worth keeping separate in my head.

**What gets quantized:** weight quantization targets the fixed, learned parameters, done once and frozen after. Activation quantization targets the computed outputs (`input × weight + bias`), recalculated live on every inference run.

**How it gets quantized, for either target:** symmetric forces the representable range to center on zero, using the largest magnitude value - zero always lands exactly on a code so no extra storage is needed, just a scale factor. Simple and cheap in hardware, but wastes codes if the real data isn't actually centered on zero. Asymmetric uses the real min/max directly instead, no forcing - every code does useful work, finer precision - but zero usually doesn't land on a clean code anymore, so an extra number (the zero-point) has to be stored alongside the scale.

**Full worked comparison - 5 weights: `-0.20, -0.05, 0.03, 0.31, 0.87`**

**Symmetric** (forces range to ±0.90, the largest magnitude; signed codes -8..7; scale = 0.1286):
```
code:  -8    -7    -6    -5    -4    -3    -2    -1     0     1     2     3     4     5     6     7
value: -1.03 -0.90 -0.77 -0.64 -0.51 -0.39 -0.26 -0.13  0.00  0.13  0.26  0.39  0.51  0.64  0.77  0.90
       |<------------- wasted: no data this low ------------->|<---------- actually used ---------->|
```
6 of 16 codes (-8 to -3) sit below the real minimum (-0.20) - wasted. Zero lands exactly on code 0, free.

**Asymmetric** (fits the real range -0.20..0.90 directly; unsigned codes 0..15; scale = 0.0733):
```
code:   0     1     2     3     4     5     6     7     8     9    10    11    12    13    14    15
value: -0.20 -0.13 -0.06  0.01  0.09  0.17  0.24  0.31  0.39  0.46  0.54  0.61  0.68  0.76  0.83  0.90
                          ↑
                   zero-point = 3
```
All 16 codes fall inside the real data range, nothing wasted on headroom that's never used - about 1.75x finer resolution than symmetric (0.0733 vs 0.1286 per code), at the cost of storing that zero-point.

In real deployments, both get used at once, not either/or. Weights go symmetric, because trained weights naturally cluster loosely around zero so the waste is small, and it's faster/cheaper in hardware with no zero-point offset in the multiply - this is the default for weight quantization in GPTQ, AWQ, and bitsandbytes. Activations go asymmetric, because activations after a ReLU (or similar) are entirely non-negative and heavily skewed, and symmetric would waste half the range on negative values that never occur. So production pipelines quantize weights symmetrically and activations asymmetrically, in the same model, at the same time.

### Calibration - choosing the range, not just the scheme

Both diagrams above assumed the "real" min/max were already known. Calibration is the step that decides what those min/max should actually be, and the honest answer is usually not the literal min/max of the data.

Real weight/activation distributions tend to have a tight bulk near zero plus a few rare outliers way out at the edges. Naively using the literal min/max (outliers included) stretches the whole 16-code range to cover those extremes, wasting most of the resolution on territory almost no real value ever touches. What calibration does instead: run a small representative sample of real data through the model, see where most values actually fall, and deliberately pick a tighter clipping range that fits the bulk - clamping the rare outliers to the nearest edge code (accepting some error just for those) in exchange for much finer resolution everywhere else.

**Worked example - 21 sampled weights, 2 outliers near ±1.9, 86% of values inside ±0.8**

**Naive range** (literal min/max, -1.9 to 1.9; scale = 0.253):
```
code:   0     1     2     3     4     5     6     7     8     9    10    11    12    13    14    15
value: -1.90 -1.65 -1.39 -1.14 -0.89 -0.63 -0.38 -0.13  0.13  0.38  0.63  0.89  1.14  1.39  1.65  1.90
             |<------------------ most codes land where almost no data lives ------------------>|
```

**Calibrated range** (outliers clipped, -0.8 to 0.8; scale = 0.107):
```
code:   0     1     2     3     4     5     6     7     8     9    10    11    12    13    14    15
value: -0.80 -0.69 -0.59 -0.48 -0.37 -0.27 -0.16 -0.05  0.05  0.16  0.27  0.37  0.48  0.59  0.69  0.80
       |<------------------------- every code now covers real data -------------------------->|
```

Naive range gives a scale of 0.253 per tick, coarse, most of it wasted on empty territory. Calibrated range tightens to 0.107 per tick, roughly 2.4x finer, for the 86% of values that actually matter. In one sentence: calibration means choosing the clipping range from real sample data instead of blindly trusting the literal min/max, so a handful of rare outliers don't ruin resolution for everything else.

### Post-Training Quantization (PTQ) vs. Quantization-Aware Training (QAT)

Everything in this file up to here - the INT4 ruler, the VRAM formula, symmetric/asymmetric, calibration - is PTQ: take an already-trained model, weights and biases fully learned, and quantize it afterward as a separate step before deploying.

QAT is the alternative. Quantization happens *during* training instead of after, so the model learns to be robust to rounding error while it's still training. Usually gives better quality at very low bit-widths, but it's far more expensive since it requires actual retraining, not just a one-time conversion pass on an existing model.

### Dynamic vs. static activation quantization

Both are still PTQ, happening after training. The real difference is when the scale/zero-point for activations gets calculated:

| | Dynamic | Static |
|---|---|---|
| When scale/zero-point is computed | Fresh, live, every single inference call | Once, ahead of time, via calibration |
| What it needs | Nothing upfront | A calibration dataset run through the model beforehand |
| Accuracy | Higher, perfectly fit to each real input | Slightly lower if a real input differs from the calibration sample |
| Speed | Slower, extra calculation every call | Faster, no per-call overhead, fixed numbers reused |

### Weight quantization timing: recompute-every-load vs. quantize-once-save

A separate axis from dynamic/static activation quantization above - this one is about weights, and about how often the quantization math gets redone, not about per-call timing.

With bitsandbytes, the original FP16 weights stay on disk unquantized. Every time `from_pretrained(..., load_in_4bit=True)` runs, bitsandbytes converts them to INT4 fresh, in memory, right then - nothing quantized gets saved back to disk. Cheap enough in practice (seconds), but it is real repeated work, once per process load, not once per inference call and not once ever.

TensorRT, GGUF, and GPTQ work the other way: a separate build/conversion step runs once, produces an already-quantized file, and every future load just reads that file's bytes - no quantization math at load time at all.

### Hands-on: bitsandbytes on Google Colab (T4 GPU)

Ran the FP16 → INT8 → INT4 comparison for `microsoft/Phi-3-mini-4k-instruct` (3.8B params) on a free Colab T4.

**Notebook:** [`notebooks/04-quantization-bitsandbytes.ipynb`](../notebooks/04-quantization-bitsandbytes.ipynb)

**Real measured VRAM vs. the formula prediction:**

| Precision | Measured VRAM | Formula prediction `(n_bits/8)×3.8B` |
|---|---|---|
| FP16 | 7.64 GB | 7.6 GB |
| INT8 | 4.02 GB | 3.8 GB |
| INT4 | 2.44 GB | 1.9 GB |

FP16 matches the formula almost exactly. INT8 and INT4 both come in a bit higher than the pure formula, which makes sense - not every layer actually gets quantized (embeddings and some norm layers tend to stay in higher precision), and the scale/zero-point constants themselves take a little extra storage. The formula gives the theoretical floor; real measured VRAM always sits a bit above it.

Quality check at INT4, prompt: *"Explain what a check engine light means in one sentence."* Response: *"The check engine light illuminates when the vehicle's on-board diagnostics system detects an issue with the engine or related components."* Coherent, correct, and the aggressive 4-bit compression didn't visibly hurt quality for a task this simple.

Key `BitsAndBytesConfig` params used: `load_in_4bit=True` turns on 4-bit quantization at load time. `bnb_4bit_quant_type="nf4"` sets NormalFloat4, Hugging Face's recommended type for normally-distributed weights. `bnb_4bit_compute_dtype=torch.bfloat16` keeps weights stored at 4-bit but temporarily dequantizes to bfloat16 during the forward pass for speed and precision. `bnb_4bit_use_double_quant=True` (optional) quantizes the quantization constants themselves too, saving another ~0.4GB per 1B params.

Both dynamic and static activation quantization need real data at some point - dynamic needs live activations from actual inference, static needs sample activations from calibration runs - the real distinction is recompute constantly vs. compute once and reuse.

## Connects to what I already know
- Directly parallels the fbgemm/cuda quantization work already done on vision models in `sdv-edge-gateway` - same core idea (reduce numeric precision to shrink memory/compute), just applied to language model weights instead of CNN weights.
- Distinct from pruning/distillation, which actually reduce parameter *count*. Quantization only changes how each parameter is *stored* - easy to conflate the two, worth keeping straight.

## Open Questions
- How does the "shared scale factor per group" choice (group size) affect the precision/memory tradeoff in practice? GPTQ and AWQ differ here - revisit once doing the hands-on resources.

## Resources
- [A Visual Guide to Quantization - Maarten Grootendorst](https://www.maartengrootendorst.com/blog/quantization/) (started 23-Sep-26)
- [Eldar Kurtić - Beginner Friendly Introduction to LLM Quantization: From Zero to Hero](https://www.youtube.com/watch?v=LK2-lrLvhTA)
- [Which Quantization Method Is Best for You?: GGUF, GPTQ, or AWQ | E2E Networks](https://www.e2enetworks.com/blog/which-quantization-method-is-best-for-you-gguf-gptq-or-awq) - hands-on, next up
