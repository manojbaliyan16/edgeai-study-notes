# Inference Engineering - Chapter 2: Models

**Source:** *Inference Engineering* by Philip Kiely
**Related:** `100-days-of-inference` repo, Day 02 (`day02/` - tokenization, model internals) covers overlapping ground. Reading the book directly instead, building my own notes rather than following the repo's notebooks.

## Layers, nodes, and connections

> "A group of nodes forms a layer. Nodes within a layer are independent of each other - they do their own calculations. The connection between nodes, or the 'network' in a neural network, is between layers, where the nodes in a layer receive the output of the previous layer."

```mermaid
flowchart LR
    subgraph L1["Layer 1"]
        direction TB
        A1((n1))
        A2((n2))
        A3((n3))
    end
    subgraph L2["Layer 2"]
        direction TB
        B1((n1))
        B2((n2))
        B3((n3))
        B4((n4))
    end
    subgraph L3["Layer 3"]
        direction TB
        C1((n1))
        C2((n2))
    end

    A1 --> B1
    A1 --> B2
    A1 --> B3
    A1 --> B4
    A2 --> B1
    A2 --> B2
    A2 --> B3
    A2 --> B4
    A3 --> B1
    A3 --> B2
    A3 --> B3
    A3 --> B4

    B1 -.->|"every node in L2<br/>feeds every node in L3"| C1
    B1 -.-> C2
    B2 -.-> C1
    B2 -.-> C2
    B3 -.-> C1
    B3 -.-> C2
    B4 -.-> C1
    B4 -.-> C2
```

No line ever connects two nodes inside the same layer box. Each one computes independently, in parallel - no idea what its neighbors are doing. Every connection crosses between layers instead: a node sends its output to *every* node in the next layer, not just one.

A layer, really, is just a group of nodes - not a stand-in for a single node the way I kept half-assuming. So when the book says "the connection is between layers," it's describing the overall pattern (wires only ever cross layer boundaries), not claiming connections skip individual nodes. Zoom in and it's still node-to-node wiring; zoom out and the pattern reads as layer-to-layer.

Flow, start to finish: input arrives, Layer 1's nodes each compute on their own, all of those outputs get handed to every node in Layer 2, Layer 2 computes, and so on until the output layer. Every node is doing the same basic calculation: take everything it receives, multiply each piece by its own learned weight, add a bias term, and combine it all into one output number (`weight x input + bias`). That single number is what gets passed forward.

## Dimensionality: text expands, images shrink

> "Internal representations for text input increase the dimensionality, encoding text chunks into vectors of hundreds or thousands of numbers to capture semantic meaning. But internal representations for image models reduce the dimensionality from millions of pixels down to a manageable size."

```mermaid
flowchart LR
    T["TEXT<br/>'dog' (a few tokens)"] -->|expands| E["embedding vector<br/>[0.14, -0.82, 0.31, ...]<br/>hundreds of numbers"]
    I["IMAGE<br/>millions of pixels"] -->|shrinks| E
```

Text starts small - a handful of tokens - and expands outward into a vector of hundreds or thousands of numbers, because a few raw tokens can't hold enough nuance on their own. Images run the opposite direction: millions of raw pixels, most of it redundant, get compressed down into that same kind of fixed-size vector. Different starting points, same destination - one fixed-size embedding that's supposed to capture "meaning."

What "semantic" actually means - a map where distance stands for meaning, not spelling:

```mermaid
graph LR
    dog((dog)) ---|close| puppy((puppy))
    dog ---|close| retriever((golden<br/>retriever))
    dog -.->|cat is nearby too| cat((cat))
    car((car)) ---|close| engine((engine))
    car ---|close| cel((check engine<br/>light))

    dog -.->|far from| car
    linkStyle 5 stroke:#999,stroke-dasharray: 4 4
```

That's the part I kept getting hung up on with "semantic search." In this embedding space, points land near each other based on what they mean, not how they're spelled - "dog," "puppy," and "golden retriever" cluster together; "car" and "engine" sit way off to the side, even though "dog" and "car" don't share a single letter. So a semantic search for "puppy" can surface a document that only ever said "golden retriever" - the model finds the nearest point in meaning-space, not the nearest matching string.

## Model types: image vs. text, autoregressive generation

Image and text models end up architected differently for the same reason as the dimensionality split above - one compresses, one expands - but underneath, both are still built from the same neuron/layer structure.

Worth locking in the right word here: **autoregressive**, not "regressive." Regression means predicting a continuous number, a different thing entirely. Autoregressive generation means producing output one token at a time - predict a token, append it to the input, feed the whole thing back in, predict the next one, repeat until done:

```mermaid
flowchart LR
    IN["'The cat'"] -->|model| OUT["'sat'"]
    OUT -.->|appended, fed back in as new input| IN
```

## Neurons, layers, hidden layers, encoder/decoder

```mermaid
flowchart LR
    subgraph ENC["ENCODER - builds the internal representation"]
        direction LR
        I1((n1)) --> H1((n1))
        I1 --> H2((n2))
        I1 --> H3((n3))
        I2((n2)) --> H1
        I2 --> H2
        I2 --> H3
        I3((n3)) --> H1
        I3 --> H2
        I3 --> H3
    end

    H1 --> R((internal<br/>representation))
    H2 --> R
    H3 --> R

    subgraph DEC["DECODER - produces the output"]
        direction LR
        R --> H4((n1))
        R --> H5((n2))
        H4 --> O((output))
        H5 --> O
    end
```

A neuron is the smallest unit - same thing as "node" above, just the book's preferred word. A group of neurons makes a layer, and every layer except the first (input) and last (output) counts as a hidden layer.

The encoder/decoder split follows naturally from that: the input-side layers are the encoder, squeezing the raw input down into one internal representation. The output-side layers are the decoder, taking that representation and expanding it back out into the final answer. I had this backwards the first time I tried to explain it out loud - easy mix-up, so worth writing down plainly: encoder builds the representation, decoder consumes it.

## §2.1.1 Linear Layers and Matmul

**Vector vs. matrix, quickly:** a vector is a single list of numbers (one row or one column). A matrix is a grid of numbers, rows stacked on columns - really just several vectors lined up together.
```
vector:        matrix:
[2]            [ 1   2 ]
[1]            [ 0   1 ]
               [ 3  -1 ]
```

A linear layer is the simplest form of matrix multiplication: an input vector goes in, gets multiplied by a weight matrix, a bias vector gets added, and an output vector comes out. First read that felt like a new concept on top of everything above, but it isn't - it's the same "each neuron computes `weight x input + bias`" idea, just written for a whole layer at once instead of one neuron at a time.

Take the 3-neuron layer from earlier and give it a 2-number input, `x = [2, 1]`:

```
weight matrix W (3 neurons x 2 inputs)      input x       bias b       output
[ 1   2 ]                                   [2]            [1]          [5]   <- neuron 1
[ 0   1 ]            x                      [1]      +     [0]     =    [1]   <- neuron 2
[ 3  -1 ]                                                  [2]          [7]   <- neuron 3
```

Each row of the matrix belongs entirely to one neuron. Row 1 (`1, 2`) is neuron 1's weights, nobody else's - multiply it against the input, add neuron 1's own bias (1), and that's neuron 1's output: `(1x2) + (2x1) + 1 = 5`. Same for row 2: `(0x2) + (1x1) + 0 = 1`. Row 3: `(3x2) + (-1x1) + 2 = 7`.

So `Wx + b` isn't a separate operation bolted onto the neuron picture - it's the exact same per-neuron calculation, packed row by row into one matrix so all three neurons get computed in a single step instead of a loop.

## Activation functions - why a network needs them at all

Stack linear layers with nothing between them and the math collapses on itself: `W2(W1x + b1) + b2` simplifies algebraically into one single `W'x + b'`. A 50-layer network with no activations is mathematically no more powerful than a 1-layer network, no matter how deep it looks on paper.

```
Stacked linear layers, no activation        Stacked layers WITH ReLU between them
(always collapses to one straight line)     (genuinely bends - new shapes possible)

  y                                           y
  |                    .                      |              .
  |                  .                        |         . .
  |                .                          |       .
  |              .                            |   . .
  |____________.______ x                      |_.__________ x
```

That collapse is exactly what an activation function exists to prevent - it sits between every pair of linear layers specifically to break the chain, so depth actually buys something.

**ReLU (Rectified Linear Unit)** is the simplest, most common one:

```
ReLU(x) = max(0, x)

  y
  |                /
  |              /
  |            /
  |__________/________ x
  |        0
   (x < 0 -> output 0, flat)   (x > 0 -> output = x, unchanged)
```

Every neuron produces a raw number first (`weight x input + bias`) - that raw number is the pre-activation. Run it through ReLU and the result is the neuron's **activation**: negative input becomes 0, positive input passes through unchanged. That's the whole rule. "Activation function" is just the name for whatever bends that raw number before it moves on to the next layer.

## Open Questions
- Why is prefill compute-bound while decode is memory-bound? Next thing to work out.
- How exactly does an image model's pixel-to-embedding compression decide what's "redundant" vs "meaningful"? Revisit once I'm hands-on with a vision encoder.

## §2.2 LLM Inference Mechanics

### What inference is

Training is the one-time job of adjusting a model's weights until it predicts text well. It runs for weeks on a GPU cluster and ends with a file of weights (billions of numbers). Inference is using that finished file to answer a request. The weights are frozen, nothing is learned while answering.

Think of training as studying for an exam and inference as sitting it. Your notes don't change mid-exam.

Inference is the part users actually touch. Every chat message, API call and autocomplete is one inference request, so the ongoing cost and the waiting time both live here. That is why inference engineering exists as its own field.

```mermaid
flowchart LR
  T["TRAINING<br/>learn from huge data<br/>weights change<br/>done once on a GPU cluster"] -->|"saves weights file"| I["INFERENCE<br/>answer a prompt<br/>weights frozen<br/>runs on every request"]
```

### Tokens and vocabulary

An LLM generates one token at a time, and each new token depends on every token before it. A token is a number standing for a chunk of text: a whole common word, or a fragment of a rarer one. Turning text into tokens and back needs no neural network. The tokenizer is a plain lookup table, and the full table is the model's vocabulary (usually over 100,000 entries). A more efficient tokenizer means fewer tokens for the same text, so fewer forward passes and faster inference.

```
"Explain TLS"  ->  [ "Explain", " TLS" ]  ->  [ 1842, 9031 ]
common words are one token, rare words split into pieces
```

### Three sequences and the context window

A request can contain three token sequences:

- **Input**: prompt, chat history, tool definitions.
- **Reasoning**: optional, only for thinking models. Tokens the model writes to itself before answering.
- **Output**: the answer.

The **context window** is the total number of tokens the model can process and generate per request. All three sequences have to fit inside it together. `max_tokens` caps the output part only.

```
|<--------------------- context window --------------------->|
|  INPUT                 |  REASONING        |  OUTPUT       |
|  prompt, history,      |  optional,        |  the answer   |
|  tool definitions      |  thinking models  |  (max_tokens  |
|                        |  only             |  caps this)   |
```

The window is per request, not per conversation. The model remembers nothing between requests, so a chat app re-sends the earlier turns as part of the input each time. A long chat eats the window.

### Who builds the input

The model does not choose its input. The application collects the pieces, and the chat template (a model-specific format) flattens them into one token sequence. Tokenizing that flattened text is step zero of inference.

```mermaid
flowchart LR
  S["system prompt<br/>developer rules"] --> CT
  H["chat history<br/>earlier turns"] --> CT
  U["your new message"] --> CT
  TD["tool definitions"] --> CT
  CT["chat template<br/>model-specific format"] --> SEQ["one flat token sequence"]
  SEQ --> PF["goes into PREFILL"]
```

### The two phases

Every request runs in two phases:

- **Prefill**: read the whole input in one go. All input tokens are processed together, and the results are saved in the KV cache.
- **Decode**: write the output one token per forward pass. Each token needs the ones before it, so passes cannot be skipped or run in parallel for a single request.

```mermaid
flowchart TD
  A["text prompt"] --> B["chat template + tokenizer<br/>text to token numbers"]
  B --> C["PREFILL<br/>read all input tokens at once<br/>fill the KV cache"]
  C --> D["DECODE<br/>one forward pass"]
  D --> E["logits<br/>one score per vocabulary word"]
  E --> F["normalize to probabilities<br/>weighted random pick"]
  F --> G{"stop token or<br/>limit reached?"}
  G -->|"no: append token, go again"| D
  G -->|yes| H["done"]
```

Prefill and decode account for nearly all inference time, because both run the full network.

### KV cache

Attention is how each token looks back at the earlier tokens to decide which ones matter for it. (The mechanics are not covered in these notes yet.) For attention, every token produces two sets of numbers, called K and V, that later tokens read.

A token's K and V never change once computed, because earlier tokens never depend on later ones. So they are stored in a table with one row per token. That table is the KV cache.

```
            token     K          V
 prefill    Explain   numbers    numbers
 (filled    TLS       numbers    numbers
  at once)
 decode     is        numbers    numbers    <- added on pass 1
 (one row   a         numbers    numbers    <- added on pass 2
  per pass) (next)    ......     ......     <- next pass adds its row here
```

- Without the cache, every decode pass would recompute K and V for all earlier tokens.
- With the cache, a pass computes only the new token's row and reads the rest from the table.
- The price is memory. The table sits in GPU memory and grows with every token, so longer conversations use more of it.

### How the output token is chosen

The last layer of the network produces a vector of logits, one score for every word in the vocabulary, so its length equals the vocabulary size. After normalizing, the scores become probabilities. The model does not decide the way a person does. The pick is a dice roll weighted by those probabilities. Training is what makes the right word score highest.

```
Prompt: "The capital of France is"        (example numbers, only the shape matters)

 Paris         ########################################  92%
 the           ##                                          4%
 a             #                                           2%
 Lyon          #                                           1%
 99,996 others #                                           1% combined

 weighted dice roll: Paris comes out about 92 times in 100
```

Three settings reshape that pick:

- **Temperature**: adjusts the logits before normalizing. Lower is more predictable, higher flattens the bars so other words get a chance.
- **Top-k**: keep only the k most likely tokens, re-normalize among them.
- **Top-p**: keep the smallest set of tokens whose probabilities add up to p.

Temperature 0 or top-k 1 makes the pick deterministic: always the highest score, so the same input gives the same output. For structured output such as JSON, logit biasing pushes invalid tokens out after each pass.

The loop repeats until the model produces a stop token (a special value meaning the output is finished), unless the context window or `max_tokens` is hit first.

## Next session
Continue §2.2: why prefill and decode behave differently on a GPU (compute-bound vs memory-bound), then §2.2.1 LLM Architecture (`config.json`, how to read a name like `Qwen3MoeForCausalLM`).
