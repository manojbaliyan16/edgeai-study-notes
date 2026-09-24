# Edge AI for Beginners — Study Notes

Study log tracking my progress through [edgeai-for-beginners](https://github.com/manojbaliyan16/edgeai-for-beginners) (fork of Microsoft's Edge AI course).

Notes framed against my automotive/embedded background (RTCU telematics architecture, TensorRT/YOLOv8 edge inference, `sdv-edge-gateway`) — see [manojbaliyan16/sdv-edge-gateway](https://github.com/manojbaliyan16/sdv-edge-gateway) and [manojbaliyan16/ml-automotive-foundations](https://github.com/manojbaliyan16/ml-automotive-foundations) for the related project work.

## Progress

See [PROGRESS.md](PROGRESS.md) for the live module checklist.

## Structure

Each module gets a `modules/NN-module-name.md` file with:
- Key concepts (in my own words, not copied from the course)
- How it connects to what I already know (TensorRT, ONNX quantization, MQTT edge pipelines, RTCU telematics architecture)
- Open questions / gaps
- Hands-on exercise notes, if any

Runnable hands-on notebooks live separately in `notebooks/NN-topic.ipynb`, linked from the matching module's notes file.

## Curriculum Map

| # | Module | Phase | Focus |
|---|---|---|---|
| 00 | Introduction to EdgeAI | Foundation | Overview, learning objectives |
| 01 | EdgeAI Fundamentals | Foundation | Cloud vs Edge AI, case studies |
| 02 | SLM Model Foundations | Foundation | Model families (Phi, Qwen, Gemma, BitNET, μModel, Phi-Silica) |
| 03 | SLM Deployment Practice | Implementation | Local + cloud deployment scenarios |
| 04 | Model Optimization Toolkit | Implementation | Llama.cpp, Microsoft Olive, OpenVINO, Apple MLX |
| 05 | SLMOps Production | Production | Distillation, fine-tuning, deployment ops |
| 06 | AI Agents & Function Calling | Production | Agent frameworks, function calling, MCP |
| 07 | Platform Implementation | Specialization | AI Toolkit, Foundry Local, Windows dev |
| 08 | Foundry Local Toolkit | Specialization | 10 sample apps (REST chat, RAG, multi-agent, model routing) |
| W1 | Workshop | Supplementary | Hands-on Python/Jupyter samples |
| W2 | WorkshopForAgentic | Supplementary | "AI Podcast Studio" multi-agent project |
