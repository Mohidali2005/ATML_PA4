# ATML PA4: Efficient Inference for Language Models

Code, configurations and machine readable results for three studies of inference cost in language models. Task 1 covers pruning and weight quantization of a pretrained model. Task 2 and Task 3 are added in their own folders as they are completed.

## Setup

```bash
git clone https://github.com/Mohidali2005/ATML_PA4.git
cd ATML_PA4
pip install -r requirements.txt
bash scripts/get_data.sh
```

`scripts/get_data.sh` clones the course data repository at a fixed commit and runs its `setup_data.py --public` script. That script downloads the fixed public rows and builds the tokenized windows. It reuses the source revisions pinned in `configs/data_lock/local_source_revisions.json`, then checks every built file against `configs/data_lock/sha256.json`. A run that does not reproduce the same files stops at this point. Raw datasets and model weights are never committed.

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

### Results layout

`results/task1/` holds:
- `conditions/<name>/`: scores, per window perplexity, per row HellaSwag scores, calibration errors, and the allocations for P3 and Q3
- `sensitivity/`: probes, OWL ratios, activation aware search grids and Wanda statistics
- `selection/`: development scores of every candidate, the frozen choice, and the later held out audit of the candidates that were not chosen
- `resources/`: timing, packed execution and kernel measurements
- `checks/`: environment, data, structure, numerical and harness checks
- `tables/`, `figures/` and `qualitative/`: built by `analyze.py`

## Attribution

- Dataset preparation uses `setup_data.py` from the course data repository (`AbDu11aHHH/atml-pa4-datasets`), which is cloned at a fixed commit and not modified.
- The HellaSwag text cleaning follows the hellaswag task of EleutherAI lm-evaluation-harness, which is also used to cross check the scoring.
