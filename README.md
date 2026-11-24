# ATML PA4: Efficient Inference for Language Models

Code, configurations and machine readable results for three studies of inference cost in language models. Task 1 covers pruning and weight quantization of a pretrained model. Task 2 and Task 3 are added in their own folders as they are completed.

## Setup

```bash
git clone https://github.com/Mohidali2005/ATML_PA4.git
cd ATML_PA4
pip install -r requirements.txt
bash scripts/get_data.sh
```

`scripts/get_data.sh` clones the course data repository at a fixed commit and runs its `setup_data.py --public` script. That script downloads the fixed public rows and builds the tokenized windows. It reuses the source revisions pinned in `configs/data_lock/local_source_revisions.json`, then checks the records of every built file against the hashes in `configs/data_lock/sha256.json`. The hashes are taken over the decompressed records, so gzip headers and line endings do not affect them. A run that does not reproduce the same files stops at this point. Raw datasets and model weights are never committed.

## Task 1: Pruning and quantization of Qwen2.5-1.5B

All code is in `task1_compress/` and every setting is in `configs/task1.yaml`. The model is `Qwen/Qwen2.5-1.5B` (base) at the commit stored in the data lock. It is loaded in FP16 with SDPA attention. Every condition starts from a fresh load of the unchanged weights, and the model is never trained.

| Group | Method | Settings |
|---|---|---|
| P1 | magnitude pruning, per output row | sparsity 0.25, 0.50, 0.75 |
| P2 | Wanda pruning, per output row, block by block on the pruned prefix | sparsity 0.25, 0.50, 0.75 |
| 2:4 | Wanda with two of every four input weights kept | 50% |
| P3 | Wanda with a sparsity per block | 50% and 75% at the same number of zeroed weights as P2 |
| Q1 | asymmetric round to nearest, group 128 | 8, 4, 3 bits |
| Q2 | activation aware channel scaling, then the Q1 quantizer | 8, 4, 3 bits |
| Q3 | Q2 quantizer with a bit width per projection from {2, 3, 4, 8} | packed bytes at or below Q2 INT4 and Q2 INT3 |

### Fixed conventions

- **Eligible weights:** q, k, v, o, gate, up and down projections of all 28 blocks, 1,310,195,712 weights in total. Embeddings, norms, biases and the tied output head stay FP16 and are counted once.
- **Calibration and development data:** 128 C4 training sequences of 512 tokens feed Wanda, the activation aware search and every sensitivity probe. A separate set of 32 sequences is used only to choose between the P3 and Q3 allocation candidates.
- **Perplexity:** the fixed WikiText-2 test and C4 validation windows, each scoring exactly 50,000 target tokens. The first token of every window is not scored. Results are saved per window.
- **HellaSwag:** the 2,000 fixed validation rows, scored zero shot by the summed log likelihood of each ending. The prompt formatting is the same as lm-evaluation-harness hellaswag version 1.0. `acc_norm` divides by the character length of the ending. FP16, the three 50% pruning methods and the six INT4 and INT3 models use all 2,000 rows. Every other condition uses the first 512.
- **Quantizer:** min max asymmetric quantization over groups of 128 input columns, with round half to even.
  - The scale is `(max-min)/(2^b-1)`, stored as FP16 and used in its FP16 form.
  - The zero point is `clamp(-round(min/scale),0,2^b-1)`, stored as one byte.
  - Codes are packed lowest bit first, so every eight codes fill exactly b bytes.
  - Every group carries 3 bytes of metadata.
  - Evaluation uses dequantized FP16 weights. The real packed layers are measured separately.
- **Activation aware scaling:** follows the published method with a grid of 20 values for alpha (0 to 0.95).
  - The scales for q, k and v and for gate and up are folded into the preceding RMSNorm gain.
  - The scale for down is folded into the rows of up.
  - The o projection keeps a stored FP16 input scale, because grouped query attention shares each value head across six query heads, so folding into v is not valid. Its 86,016 bytes are counted in every budget.
  - The weight clipping search and the fused low bit kernels of the published method are not reproduced.
- **Storage:**
  - The sparse format stores one mask bit per eligible weight plus 16 bits per nonzero.
  - The quantized format stores packed codes plus metadata.
  - Both include the uncompressed FP16 parameters.

### Adaptive methods

- **P3:**
  - *Measured design:* each block is pruned on its own at seven sparsities, and the rise in calibration NLL is measured. An exact dynamic program then picks one sparsity per block, between 0.10 and 0.85, so that the mean equals the target.
  - *Proxy design:* the outlier ratio of Wanda scores (OWL), mapped linearly with lambda in {0.04, 0.08, 0.12}.
- **Q3:**
  - *Measured design:* each projection is quantized on its own at 2, 3, 4 and 8 bits, and the rise in calibration NLL is measured. An exact knapsack then picks bit widths under the packed byte budget.
  - *Proxy design:* the local output error of each projection, using the same knapsack.

For both methods, the candidate with the lowest development NLL is selected. The choice is written to `results/task1/selection/` before any held out score of P3 or Q3 exists.

### Running

The study ran as one offline Kaggle commit on two T4 GPUs:

```bash
bash scripts/run_task1_kaggle.sh /kaggle/temp/artifacts
python -m task1_compress.analyze
```

The run script works in this order:
1. A short smoke pass of every stage on the real model.
2. `track_a` on the first GPU: baseline and checks, P1, P2, 2:4, P3 and Q1.
3. `track_b` on the second GPU at the same time: Q2 and Q3.
4. `final` alone on one GPU. It rebuilds every condition from saved masks or scales and times it, measures the packed execution path and the available kernels, and writes the greedy continuations.

A stage can also be run on its own with `python -m task1_compress.run_task1 --stage track_a`. Adding `--smoke --tiny` runs everything on a small random model on the CPU.

The masks and scales are written outside the repository. A second commit on one T4 rebuilds them from the committed results and repeats the timing stage:

```bash
bash scripts/run_task1_rebuild_kaggle.sh /kaggle/temp/artifacts
```

- `--stage rebuild` recreates every evaluated condition. Uniform conditions are rebuilt from their rule, and P3 and Q3 from the allocation saved in `results/task1/conditions/<name>/allocation.json`. The Q2 scale search is repeated without the sensitivity probes.
- Each rebuilt model is walked over the calibration set again. `checks/rebuild.csv` compares its calibration NLL with the one it was evaluated with, and its calibration output errors are written again.
- `final` then times every condition. An untimed pass of the FP16 model runs first so the GPU clocks settle, and FP16 is timed again at the end (`fp16_end`) to show how far the clocks drifted over the stage.

The committed calibration output errors, timing, packed execution, kernel and continuation files come from this second commit.

### Results layout

`results/task1/` holds:
- `conditions/<name>/`: scores, per window perplexity, per row HellaSwag scores, calibration errors, and the allocations for P3 and Q3
- `sensitivity/`: probes, OWL ratios, activation aware search grids and Wanda statistics
- `selection/`: development scores of every candidate, the frozen choice, and the later held out audit of the candidates that were not chosen
- `resources/`: timing, packed execution and kernel measurements
- `checks/`: environment, data, structure, numerical and harness checks
- `tables/`, `figures/` and `qualitative/`: built by `analyze.py`

## Task 2: Attention and context memory

All code is in `task2_attention/` and every setting is in `configs/task2.yaml`. The frozen MQAR development and test files and the 180 retrieval prompts come from the course data repository at the same fixed commit. `python -m task2_attention.mqar --verify` checks their records against `configs/data_lock/task2_sha256.json`.

### Part A: exact attention

Random queries, keys and values are drawn with B = 2, H = 4 and head dimension 64, at lengths 256 to 8192, in FP16 and FP32.
- **Dense** builds the full score table, applies the causal mask and uses a max-subtracted softmax.
- **Tiled** visits blocks of 128 queries and 128 keys. It skips blocks that lie entirely in the future and keeps a running max, sum and output in FP32. Its backward pass is plain autograd through the loop, which stores every visited tile. It is a reference implementation, not a fused kernel.
- **SDPA** is measured with the default dispatch and with each backend forced. Support is queried with `torch.backends.cuda.can_use_*`, and the kernels launched by the default dispatch are recorded with the profiler.

Every time and memory measurement runs in its own process, so an out of memory error is recorded as a result instead of stopping the run. Errors are relative to dense FP32. Outputs are compared at every length; gradients are compared up to length 4096.

### Part B: multi query associative recall

- **Model:** two pre-norm blocks, hidden size 128, four heads of 32, a 4x MLP, learned token embeddings plus sinusoidal positions, and a 64-way value classifier read at the query positions. Only the attention operator differs between runs, and every model has 437,824 parameters.
- **Training:** AdamW with peak learning rate 8e-4, weight decay 0.01 on matrices, 200 warmup updates and a linear decay to 15 percent, gradient clipping at 1.0, batch 32 and 4,500 updates, for seeds 6304, 6305 and 6306.
  - Each (K, regime) training stream holds 144,000 examples made by the course generator from its `train` seeds, and is shared by every run that trains on it. The seed sets the initial weights and the order in which the stream is read.
  - `results/task2/checks/mqar_checks.json` records the following: every frozen record is rebuilt exactly by the generator; the regime distance limits; and no training sequence equals a development or test sequence.
- **Mechanisms:**
  - dense causal attention
  - a causal window of 64 positions
  - a local and dilated mix: two heads see the last 64 positions, and two heads see offsets 0, 4, ..., 252
  - causal ELU+1 linear attention with the running state materialized at every position

  The window and dilated masks are dense boolean masks applied to dense scores. They are correctness implementations, not sparse kernels.
- **Evaluation:**
  - Development accuracy is measured at updates 500, 1,000, 1,500, 2,000, 3,000 and 4,500.
  - At the end, every model is tested on all 9,216 frozen test sequences (every K and regime), with accuracy also split into distance bins.
  - The dense models are evaluated a second time with the 64-position window imposed at test time only.
- **Resource costs:** a forward-only sweep on random input embeddings measures time and memory at lengths 256 to 8192.

### Part C: KV cache memory of Qwen2.5-1.5B-Instruct

**Prompt split.** Each prompt is the chat template applied to the case text. It is split on tokenizer offsets into the context (everything before `QUESTION_BEGIN`) and the question suffix.

**Prefill and scoring.**
1. The context is prefilled once in FP16 with SDPA.
2. For question-aware scoring, the suffix is run once on the full cache. Pre-hooks on each attention layer rebuild the rotary queries and keys, and compute FP32 causal attention from every suffix token to every context position. The mass per position is summed over suffix tokens and over the six query heads of each key value head, then max-pooled over 7 positions.
3. That pass's suffix keys and values are discarded.

**Answering.** Each cache variant is then built from the context cache. The suffix is prefilled again through it at its original absolute positions, so every answer depends only on the compressed cache. The gold code is teacher-forced on a copy of the cache, and the answer is decoded greedily for up to 16 tokens.

**Budget.** Every prompt has one budget: the serialized bytes of uniform INT4 with channel-wise keys and token-wise values on its context. Groups are 64 elements with an FP16 scale and an FP16 offset, and the trailing `L mod 64` tokens stay FP16.

| Variant | Storage |
|---|---|
| sink and recent | 8 sink and the most recent positions in FP16, as many as fit |
| question aware | 8 sink, 64 recent, then the highest-scoring positions, chosen per layer and key value head; a 2 byte index per selected position is counted |
| INT4 token wise | keys and values grouped along features |
| INT4 channel K | keys grouped along tokens per channel, values along features |
| hybrid | 8 sink, 64 recent and the highest-scoring positions in FP16; the rest in INT2 complete groups (keys channel-wise, values token-wise); an incomplete INT2 group stays FP16 |

During decoding the INT4 caches quantize every newly completed 64-token group.

**Checks.** `kv/validation.json` covers:
- split prefill against one full forward
- an untouched edited cache against `generate`
- a shared random eviction against a full forward with an equivalent 4D mask

The packed byte counts are checked against the formula in the manual.

### Running

The study runs as one offline Kaggle commit on two T4 GPUs:

```bash
bash scripts/run_task2_kaggle.sh
```

The script runs a smoke pass of every part first. Then Part A followed by Part B runs on the first GPU while Part C runs on the second, and `analyze.py` builds `results/task2/tables` and `results/task2/figures`. Each part can be run on its own:
- `python -m task2_attention.exact`
- `python -m task2_attention.train_recall --stage data|train|sweep`
- `python -m task2_attention.kv_eval`

Adding `--smoke` runs small sizes, and `--tiny` gives `kv_eval` a small random model with the real tokenizer. The final MQAR checkpoints are written to `outputs/task2/checkpoints/` and are not committed.

## Attribution

- Dataset preparation uses `setup_data.py` from the course data repository (`AbDu11aHHH/atml-pa4-datasets`), which is cloned at a fixed commit and not modified. Task 2 imports the course MQAR generator from the same clone.
- The HellaSwag text cleaning follows the hellaswag task of EleutherAI lm-evaluation-harness, which is also used to cross check the scoring.
