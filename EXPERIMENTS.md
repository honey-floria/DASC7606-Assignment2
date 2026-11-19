# Controlled experiments

All core comparisons use the fixed supplied train/validation data and tokenizer,
random initialization, seed 1337, context length 256, vocabulary size 8192,
batch size 16, gradient accumulation 1, BF16 training, and exactly 2,500
optimizer updates (10,240,000 processed target tokens). Evaluation uses the same
fixed prompts and the complete validation split in float32.

## Baseline

- Config: `configs/baseline.yaml`
- Model: 6 layers, 6 heads, width 384 (13,891,584 parameters)
- Peak learning rate: 6e-4
- Output: `outputs/baseline`

## Model-side experiment: increased depth

- Research question: With all other model and training settings fixed, does
  increasing decoder depth from 6 to 8 layers improve validation performance and
  short-story continuity within the same processed-token budget?
- Hypothesis: The additional depth and capacity may improve validation loss and
  entity/plot consistency, but it will increase parameter count, GPU memory, and
  runtime. The benefit may be limited because both runs see the same number of
  tokens.
- Single changed factor: `n_layer: 6 -> 8`.
- Config: `configs/model_depth8.yaml`
- Model size: 17,440,512 parameters (below the 20M limit).
- Output: `outputs/model_depth8`

## Optimization-side experiment: lower peak learning rate

- Research question: With the baseline model and all other training settings
  fixed, how does lowering the peak learning rate from 6e-4 to 4e-4 affect
  convergence and final validation performance?
- Hypothesis: The lower rate may produce smoother updates, but under the fixed
  token budget it may converge more slowly and finish slightly behind the
  baseline.
- Single changed factor: `learning_rate: 6e-4 -> 4e-4`.
- Config: `configs/opt_lr4e4.yaml`
- Model size: 13,891,584 parameters.
- Output: `outputs/opt_lr4e4`

## Commands

```sh
python -m src.train --config configs/model_depth8.yaml --out_dir outputs/model_depth8 --device cuda
python -m src.train --config configs/opt_lr4e4.yaml --out_dir outputs/opt_lr4e4 --device cuda
```

Each completed run is followed by `scripts/plot_run.py`, fixed-prompt sampling,
and complete-validation evaluation using the same commands and protocol as the
baseline.

## Results

All three runs completed the same 10,240,000-token budget. Trainer validation
loss is sampled under the configured mixed-precision training workflow. The
independent result is evaluated over all 389,048 validation tokens in float32;
the two values are therefore related but should not be treated as interchangeable.

| Run | Single change | Params | Trainer val loss / PPL | Full float32 CE / PPL | Runtime | Peak GPU memory |
|---|---|---:|---:|---:|---:|---:|
| Baseline | None | 13,891,584 | 2.23378 / 9.3351 | 2.29937 / 9.9679 | 49.49 s | 1,267.55 MiB |
| Model depth | `n_layer` 6 -> 8 | 17,440,512 | 2.21541 / 9.1652 | **2.28101 / 9.7865** | 72.23 s | 1,517.74 MiB |
| Lower LR | peak LR 6e-4 -> 4e-4 | 13,891,584 | 2.31234 / 10.0980 | 2.37793 / 10.7825 | 58.34 s | 1,267.55 MiB |

### Model-side interpretation

Increasing depth reduced full-validation cross-entropy by about 0.01837 and PPL
by about 0.1814 relative to baseline. The improvement is small but agrees with
the sampled trainer validation result. It cost roughly 25.5% more parameters,
46% more wall-clock runtime in these runs, and 19.7% more peak allocated GPU
memory. The hypothesis is therefore only partly supported: extra depth improves
the quantitative result at this budget, but the gain is modest relative to its
compute cost.

The fixed-prompt samples remain grammatical and story-like, but qualitative
consistency does not improve uniformly. The dragon stays central longer, yet the
lady/bear/man identities become confused; the Tom continuation abandons the
original helping problem; the ball story mixes dad and mom and repeats its ending.
Thus the lower validation loss does not establish a general improvement in
narrative consistency from these three samples.

### Optimization-side interpretation

Lowering the peak learning rate worsened full-validation cross-entropy by about
0.07855 and PPL by about 0.8146 relative to baseline. Its curve is smooth, but it
remains above the baseline throughout the matched token budget. This supports
the hypothesis that 4e-4 converges too slowly for the fixed budget rather than
revealing an instability benefit.

Its generations expose stronger repetition and local incoherence: the dragon
sample repeats "dragon" and "had fun", while the Tom sample repeatedly uses
"hole" and switches between Tom and Tim. The ball sample is shorter and locally
readable but still changes a rock into a ball.

Runtime comparisons should be treated cautiously because the jobs were executed
sequentially on a shared cluster. Parameter count, processed tokens, precision,
evaluation protocol, and recorded GPU model are the stronger controlled evidence.

## Final core checkpoint selection

Using only the public validation evidence and the common full-split float32
protocol, `outputs/model_depth8/best.pt` is selected as the final core checkpoint.
It has the lowest full-validation cross-entropy (2.28101) and perplexity (9.7865)
of the three required runs. This selection does not imply a hidden-test guarantee
or a uniformly better qualitative sample.

Checkpoint SHA-256 values:

- Baseline: `a90ff5000932d8674dd0758e816d8334c1aead2630e4b7471d0827e4bb09be86`
- Model depth: `2858f65f13359ac4bea12c6672c2dd56d90a57971ce37204c91df1f903496203`
- Lower LR: `a02bb40ede42b672cbe4afacf75194c12ef21e25651839135a256f4db7aa3f76`
