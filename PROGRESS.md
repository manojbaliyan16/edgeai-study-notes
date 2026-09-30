# Progress Tracker

Started: 2026-09-22

**Governing sequence:** see [STUDY_PLAN.md](STUDY_PLAN.md) — this repo (`edgeai-for-beginners`) is merged with a second repo (`100-days-of-inference`) into one 8-step sequence. This table tracks module-level status *within* that plan, not the full cross-repo picture.

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
- **2026-09-27** — Recovered and saved the merged 2-repo study plan (`STUDY_PLAN.md`, originally emailed 24-Sep, only lived in Gmail/Notion/artifact until now). **Real resume point for next session:** the plan says Step 1 (`100-days-of-inference` Phase 1) should finish before Step 2 (this repo's Module 04) — but Module 04 is already deep in progress here while Step 1's actual status is unverified. First thing to do next time: check real progress on `100-days-of-inference`, then decide whether to backfill Step 1 or keep pushing Module 04 since momentum is already there. Databricks A10 run (from 09-23) still not executed — still open.
- **2026-09-30** — Started Step 1 work: reading *Inference Engineering* (Philip Kiely) directly rather than following `100-days-of-inference`'s own notebooks, building own notes instead — new `book-notes/` folder created for this. Chapter 2 (Models) covered: layers/nodes independence + cross-layer connections (with diagram), text-vs-image embedding dimensionality + what "semantic"/"semantic search" actually means (with diagram), image-vs-text model types, autoregressive generation, neurons/hidden layers/encoder-decoder (with diagram) — all logged in `book-notes/ch02-models.md`. **Standing change from today:** every study session now gets logged into notes with diagrams and pushed to GitHub by default, not just when asked. **Resume point: §2.1.1 Linear Layers and Matmul.** Databricks A10 run still not executed — still open.
