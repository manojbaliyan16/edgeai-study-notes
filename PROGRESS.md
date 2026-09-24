# Progress Tracker

Started: 2026-09-22

| # | Module | Status | Started | Finished | Notes |
|---|---|---|---|---|---|
| 00 | Introduction to EdgeAI | In Progress | 2026-09-22 | | [modules/00-introduction.md](modules/00-introduction.md) |
| 01 | EdgeAI Fundamentals | Not Started | | | |
| 02 | SLM Model Foundations | Not Started | | | |
| 03 | SLM Deployment Practice | Not Started | | | |
| 04 | Model Optimization Toolkit | In Progress | 2026-09-23 | | [modules/04-model-optimization-toolkit.md](modules/04-model-optimization-toolkit.md) |
| 05 | SLMOps Production | Not Started | | | |
| 06 | AI Agents & Function Calling | Not Started | | | |
| 07 | Platform Implementation | Not Started | | | |
| 08 | Foundry Local Toolkit | Not Started | | | |
| W1 | Workshop | Not Started | | | |
| W2 | WorkshopForAgentic | Not Started | | | |

**Status legend:** Not Started · In Progress · Done

## Log

- **2026-09-22** — Repo created, curriculum mapped from source repo README. Starting Module 00.
- **2026-09-23** — Started Chapter 01 (Module 01) reading; jumped into Module 04 out of order after quantization questions came up while reading the "Visual Guide to Quantization" blog. Covered and logged: VRAM vs RAM, parameters, INT4 worked example, weight vs activation memory, VRAM sizing formula, symmetric vs asymmetric quantization (+ diagram artifact), calibration (+ histogram diagram), PTQ vs QAT, dynamic vs static activation quantization, scale/zero-point plain explanation, weight quantization timing (recompute-every-load vs quantize-once). Ran the hands-on bitsandbytes notebook on Colab T4 (Phi-3-mini-4k-instruct, FP16/INT8/INT4 real VRAM measured and compared against the formula) — saved as `notebooks/04-quantization-bitsandbytes.ipynb`, walked through line by line, results logged into Module 04 notes. Set up an Azure Databricks A10 GPU cluster (`notebooks/04-quantization-bitsandbytes.ipynb`-equivalent cells) to re-run the same comparison, attached notebook `QuantizationDataBricksA10` — **not yet run, no results captured**, pick this up first next session. Module 02 (SLM families, reasoning vs non-reasoning SLMs) discussed conversationally but not yet logged to notes.
- **Paused 2026-09-24, mid-Module 04** — Manoj said quantization coverage is sufficient for now. **Resume point next session:** (1) get the Databricks A10 run results and compare against the Colab T4 numbers already in Module 04 notes, (2) then continue the "Visual Guide to Quantization" blog from wherever it left off (Part 2 / calibration section was the last confirmed read point), (3) Module 01 (Chapter 01 outline was covered, sections 2-4 — case studies, implementation guide, hardware platforms — still unread) is the other open thread, deferred in favor of continuing quantization by explicit choice.
