# Inference Engineering — Chapter 2: Models

**Source:** *Inference Engineering* by Philip Kiely (`/Users/manojkumar/Documents/AI_Stuff/InferenceEngineering`)
**Related:** `100-days-of-inference` repo, Day 02 (`day02/` — tokenization, model internals) covers overlapping ground; reading the book directly per own choice, building own notes rather than following the repo's notebooks.

## Layers, nodes, and connections

> "A group of nodes forms a layer. Nodes within a layer are independent of each other — they do their own calculations. The connection between nodes, or the 'network' in a neural network, is between layers, where the nodes in a layer receive the output of the previous layer."

**Diagram:** https://claude.ai/code/artifact/bf0b8041-a458-44aa-bc2f-b87d512306a5

**Key points:**
- A **layer** is a *group* of nodes, not a stand-in for a single node — nodes compose layers.
- Nodes **within** a layer never connect to each other — each computes independently, in parallel.
- Connections only cross **between** layers: every node in one layer sends its output to every node in the next layer.
- "Connection is between layers" describes the *pattern* (cross-layer only) — the actual wires are still node-to-node, just zoomed out.
- Flow: input → Layer 1 nodes compute independently → **all** of Layer 1's outputs handed to **every** Layer 2 node → Layer 2 computes → repeats to output. Each node combines *all* of the previous layer's outputs using its own weights (`weight × input + bias`, from the Module 04 activations work) to produce one output number.

## Dimensionality: text expands, images shrink

> "Internal representations for text input increase the dimensionality, encoding text chunks into vectors of hundreds or thousands of numbers to capture semantic meaning. But internal representations for image models reduce the dimensionality from millions of pixels down to a manageable size."

**Diagram:** https://claude.ai/code/artifact/bb7e55a3-4e6f-40d3-b5be-924588b0ff32

**Key points:**
- **Text:** starts small (a handful of tokens) → **expands** to a vector of hundreds/thousands of numbers (an embedding) to capture nuance/meaning that a few raw tokens can't hold.
- **Images:** start huge (millions of raw pixels, mostly redundant/correlated) → **shrink** down to the same kind of fixed-size embedding vector, discarding redundancy while keeping the meaningful content.
- Both processes converge on the *same kind of object*: a fixed-size embedding vector — text gets there by expanding, images get there by compressing.
- **"Semantic" = about meaning, not spelling/pixels.** In the embedding space, points get placed by what they *mean* — "dog," "puppy," "golden retriever" cluster close together; "car," "engine" sit far away, even though "dog" and "car" share zero letters.
- **Semantic search** = finding the nearest points in that space to a query's embedding. Searching "puppy" can surface a document that only ever said "golden retriever" and never used the word "puppy" — because they're close together in meaning-space, not in spelling.

## Open Questions
- How exactly does an image model's pixel→embedding compression decide what's "redundant" vs "meaningful"? (Revisit when hands-on with a vision encoder.)
