# Edge AI Study Plan — Merged from 2 Repos

*Original plan sent 24-Sep-2026, saved here 27-Sep-2026 so it lives alongside the actual notes instead of only in email.*

## Goal

Two repos in progress — `100-days-of-inference` (single-GPU/edge inference internals) and this repo's source, Microsoft's `edgeai-for-beginners` (on-device SLM deployment + agents). Both touch quantization and edge inference; studied separately they'd duplicate 2–3 weeks of content. This plan merges them into one sequence, ordered to avoid re-learning the same concept twice, and drops the parts of each repo that don't serve the automotive/embedded target.

## What each repo contributes

**`100-days-of-inference`** (own fork, 25 relevant topics already extracted) — single-device inference mechanics: attention/KV-cache internals, CUDA kernel-level optimization, quantization number formats and algorithms, ONNX/TensorRT compilation, Nsight/torch.profiler benchmarking. Depth on *how a model runs fast on one chip*.

**`edgeai-for-beginners`** (Microsoft, own fork — this repo tracks it) — 8 modules, on-device SLM deployment: model families (Phi, Qwen, Gemma, BitNET), toolchain (llama.cpp, Olive, OpenVINO, MLX), SLMOps, agents/MCP, Windows tooling. Breadth on *which small models exist and which toolchain deploys them*.

**Overlap:** both cover quantization + ONNX/runtime export — don't study it twice. `100-days` gives TensorRT mechanics; `edgeai-for-beginners` Module 04 gives toolchain choice.

**Competes with existing plan:** Module 06 (Agents/MCP) duplicates the already-committed LLM/RAG/Agent track — don't do both in full.

## Merged sequence

| # | Source | Topic | Why here |
|---|---|---|---|
| 1 | 100-days, Phase 1 | 10 core topics: tokenization → embeddings → attention → KV cache → ops:byte ratio → CUDA kernels → ONNX/TensorRT → TensorRT-LLM → quantization formats → quantization algorithms | Marked "in progress" (24-Sep). Finish first — everything else builds on it. |
| 2 | edgeai, Module 04 | Model Optimization Toolkit: llama.cpp, Olive, OpenVINO, MLX | Same quantization concepts, across 4 toolchains — shows when TensorRT is the wrong tool. |
| 3 | 100-days, Phase 4 (11 items) | SDPA/Flash Attention, INT8 pipeline, GPTQ, CUDA/Triton kernels, TensorRT-LLM compile-and-compare | Hands-on lab applying #1 and #2 while fresh. |
| 4 | edgeai, Module 02 | SLM Model Foundations: Phi, Qwen, Gemma, BitNET, μModel | Actual small models deployed at the edge — needed before on-device agent work. |
| 5 | edgeai, Module 03 | SLM Deployment Practice | Direct continuation of #4 — deploy what was studied. |
| 6 | Existing LLM/RAG/Agent track | DeepLearning.AI RAG → RAG++ → LangGraph → MCP (already committed) | Do NOT also run edgeai Module 06 in full. Use only its hands-on lab. |
| 7 | 100-days, Phase 5 (4 items) | Nsight profiling, GPU memory profiling, FP16 vs INT8 vs INT4 benchmarks, reusable harness | Production validation — proving #1–#3 hold up. |
| 8 (optional) | edgeai, Module 08 | 10 Foundry Local sample apps | Capstone extras if time remains. Lowest priority. |

## What to skip, and why

- **100-days Phases 2, 3, 6, 7, 8** (~70 topics): multi-GPU cluster serving, MoE, multimodal — doesn't apply to a single embedded automotive SoC.
- **edgeai Module 00–01**: cloud-vs-edge intro — redundant with 14 years of experience.
- **edgeai Module 05** (SLMOps): premature — assumes production SLMs already running. Revisit after Module 03.
- **edgeai Module 06 full study**: duplicates the committed LLM/RAG/Agent track.
- **edgeai Module 07**: Windows-specific tooling — doesn't transfer to the Mac/Ubuntu → Linux automotive target.

## How this maps to the roadmap

Steps 1–3 are the substance behind **P1.5 (Advanced TensorRT — Production Edge AI Deployment)** already logged in the Master Roadmap Excel — this plan is how that row gets executed.

Steps 4–5 sit ahead of **Track 2: LLM/RAG/Agent Engineering**. Step 6 IS Track 2 — don't run a parallel version through this repo.

Finish steps 1–7 before starting **Track 4** (automotive capstone — RAG over DTC/ISO 14229 + MCP-exposed OTA/telemetry, extending `sdv-edge-gateway`). Step 8 is optional and shouldn't delay Track 4.

**P2.3** (SDV telemetry/OTA pipeline) is separate and already-committed — this plan doesn't substitute for it.

## Where things actually stand (updated 27-Sep-2026)

- **Step 1** (`100-days-of-inference` Phase 1) — marked "in progress" as of 24-Sep, but not touched inside *this* repo/session. Status unconfirmed — needs checking against the `100-days-of-inference` repo directly next session.
- **Step 2** (edgeai Module 04) — substantial work already done in *this* repo, out of the plan's intended order: full quantization theory (INT4 ruler, VRAM formula, symmetric/asymmetric, calibration, PTQ/QAT, dynamic/static), hands-on bitsandbytes notebook with real Colab T4 results, Databricks A10 cluster set up (not yet run).
- **Open sequencing question for next session:** the plan says finish Step 1 before Step 2 — but Step 2 is already well underway here while Step 1's actual status is unverified. First thing to resolve next time: check real progress on `100-days-of-inference`, then decide whether to backfill Step 1 or keep pushing forward on Step 2 since momentum is already there.

Live editable version (source of truth for edits): https://claude.ai/artifact/374728af-b0a8-46a1-9d49-0ee14c7b1b48 — also mirrored to Notion as "studyProgress".
