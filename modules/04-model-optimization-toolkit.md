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

## Connects to what I already know
- Directly parallels the fbgemm/cuda quantization work already done on vision models in `sdv-edge-gateway` — same core idea (reduce numeric precision to shrink memory/compute), just applied to language model weights instead of CNN weights.
- Distinct from **pruning/distillation** — those actually reduce parameter *count*. Quantization only changes how each parameter is *stored*. Don't conflate the two.

## Open Questions
- How does the "shared scale factor per group" choice (group size) affect the precision/memory tradeoff in practice? (GPTQ/AWQ differ here — revisit once doing the hands-on resources.)

## Resources
- [A Visual Guide to Quantization — Maarten Grootendorst](https://www.maartengrootendorst.com/blog/quantization/) (started 23-Sep-26)
- [Eldar Kurtić — Beginner Friendly Introduction to LLM Quantization: From Zero to Hero](https://www.youtube.com/watch?v=LK2-lrLvhTA)
- [Which Quantization Method Is Best for You?: GGUF, GPTQ, or AWQ | E2E Networks](https://www.e2enetworks.com/blog/which-quantization-method-is-best-for-you-gguf-gptq-or-awq) — hands-on, next up
