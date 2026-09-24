# Module 00 — Introduction to EdgeAI

**Status:** In Progress
**Source:** https://github.com/manojbaliyan16/edgeai-for-beginners/blob/main/introduction.md

## Key Concepts

### The Edge AI Paradigm

Traditional (cloud) AI puts a network round-trip in the critical path of every inference. Edge AI removes it by running the model on-device.

```mermaid
flowchart LR
    subgraph Traditional["Traditional AI (Cloud)"]
        direction LR
        D1["📱 Device"] -->|"upload data"| C1["☁️ Cloud"]
        C1 -->|"inference"| P1["⚙️ Processing"]
        P1 -->|"result"| R1["📡 Response"]
        R1 -->|"download"| D2["📱 Device"]
    end

    subgraph Edge["Edge AI (On-Device)"]
        direction LR
        D3["📱 Device"] -->|"no network hop"| L1["⚡ Local Processing"]
        L1 --> R2["✅ Immediate Response"]
    end
```

**Why this shift matters — eliminating the round-trip enables:**

| Benefit | Why |
|---|---|
| Instantaneous responses | Sub-millisecond latency — no network hop, no cloud queue |
| Enhanced privacy | Data never leaves the device |
| Reliable operation | Works without internet connectivity |
| Reduced costs | Minimal bandwidth and cloud compute usage |

## Connects to what I already know
- TensorRT/YOLOv8 edge inference (P1.1-P2.2 work)
- Quantization work on sdv-edge-gateway (fbgemm/cuda)
- RTCU edge telemetry pipeline (automotive telematics)

## Open Questions

## Exercise Notes
