# What does this PR do ?

Adds `policy.megatron_cfg.dynamic_context_parallel`: Megatron-Core's dynamic context parallelism for the Megatron policy backend. Its scheduler sizes the context-parallel (CP) group of every packed microbatch from its sequence lengths, so short sequences run on one GPU as under plain data parallelism and a longer sequence borrows as many adjacent data-parallel ranks as it needs. A static `context_parallel_size` instead splits every microbatch over the group the longest sequence requires.

# Issues

None.

# Usage

```yaml
policy:
  megatron_cfg:
    enabled: true
    context_parallel_size: 1        # stays 1; the group is sized per microbatch
    dynamic_context_parallel: true
  sequence_packing:
    enabled: true
    train_mb_tokens: 8192           # token capacity of ONE rank per microbatch
    logprob_mb_tokens: 8192
```

A sequence of up to `k` times that capacity runs on a CP group of `k` ranks (`k` a power of two, up to all data-parallel ranks). `make_sequence_length_divisible_by` needs no CP factor.

# Design

Dynamic CP is treated as a packing schedule; nothing else in the training loop changes shape.

- The driver asks one policy worker for Megatron-Core's schedule of the global batch (`DefaultDynamicCPScheduler`; Megatron-Core is not installed in the driver environment). The schedule is a list of rounds, each holding one pack per data-parallel rank; ranks that share a pack form its CP group.
- `BatchedDataDict.shard_by_batch_size` flattens the rounds into the existing sequence-packing bin list, so the existing round-robin shard split reproduces the schedule. A pack's CP size is its multiplicity in its round, carried on the shard as `micro_batch_cp_sizes`.
- The worker shards a pack exactly like a static CP group of that size and hands the runtime group to Megatron-Core in `PackedSeqParams(local_cp_size, cp_group)`.
- Every rank of a group computes the pack's loss; the group's first rank accounts for it in the global token / sequence counts and the reported metrics, and the driver keeps one copy of the logprobs.
- The driver reuses the schedule of an unchanged batch (`lru_cache(maxsize=1)`): a GRPO step schedules the same batch for its logprob, reference-logprob and training calls.
- One change also affects static sequence packing: `shard_by_batch_size` now assembles each shard with one gather instead of one `select_indices` per pack followed by `from_batches`. A CP group's pack is shipped to every rank of the group, and the per-pack assembly was the whole cost of that (1.40 s per step for 224 rows of 60000 tokens on 16 ranks; 0.04 s with one gather, and static CP4 goes from 0.41 s to 0.01 s).

No submodule or lock-file change: it runs on the Megatron-Bridge / Megatron-LM commits `main` pins.

Rejected with an error rather than silently mis-run: pipeline parallelism, `context_parallel_size > 1`, the data plane (`data_plane.enabled: true`), models that pack or CP-slice their own inputs, prepacked inputs, `moe_hybridep_prepad_packed_inputs`, `use_fused_linear_logprobs`, draft training, the `pair_grouping_key` / `max_sequences_per_bin` / `microbatch_order: largest_first` packing options, top-k logits and full-payload teacher scoring.

# Validation

On `main`'s pins (Megatron-Bridge `ae0a1b02`, Megatron-LM `e0a4bfed`), GB300 nodes, NeMo-RL v0.8.0 image with this tree on `PYTHONPATH`:

**Same numbers as without CP.** Identical batches (32 sequences of 200-15000 tokens, Qwen2.5-1.5B, 4 GPUs), step 0; the last column is the largest gradient-norm deviation from the no-CP run over 4 steps (dynamic CP / static CP4):

| Case | Loss | Gradient norm | Worst deviation |
|---|---|---|---|
| SFT, no CP / dynamic CP / static CP4 | 13.297701 / 13.297647 / 13.297762 | 21.4918 / 21.4911 / 21.4937 | 0.03% / 0.04% |
| GRPO, no CP / dynamic CP / static CP4 | 0.202754 / 0.202754 / 0.202754 | 4.4390 / 4.4380 / 4.4393 | 0.05% / 0.12% |
| SFT, TP 2, no CP / dynamic CP | 13.297774 / 13.297812 | 21.4880 / 21.4901 | 0.01% |
| SFT, TP 2 + sequence parallel, no CP / dynamic CP | 13.297774 / 13.297812 | 21.4876 / 21.4902 | 0.03% |

Over three runs of this check the worst deviation was 0.04% (SFT) and 0.17% (GRPO). GRPO logprobs against the no-CP run: mean absolute difference 0.0110 with dynamic CP, 0.0178 with static CP4 (bf16). With four global batches per `train` call (four optimizer steps per call) dynamic CP stays within the spread of the no-CP run itself, which is up to 1.9% in gradient norm between two identical no-CP runs.

**Checkpointing.** `examples/run_sft.py` with dynamic CP saves at steps 3, 6 and 9; a second run resumes from step 6 and reproduces the losses of steps 7-9 within 0.0002.

**Unit tests.** Each suite ran on this branch and on `main` (`89329fc62`) in the same image, compared by failing test id:

| Suite | This branch | `main` |
|---|---|---|
| `tests/unit/models/megatron` + `tests/unit/data/test_megatron_sft_packed.py`, `--mcore-only` | 865 passed | 851 passed |
| `tests/unit/models/value`, `--mcore-only` | 19 passed | 19 passed |
| `tests/unit/models/policy`, `--mcore-only` | 276 passed, 2 errors | 276 passed, 2 errors |
| `tests/unit/algorithms`, `--mcore-only` | 33 passed | 32 passed, 1 failed |
| `tests/unit/models/policy` | 256 passed, 21 errors | 251 passed, 21 errors |
| `tests/unit/algorithms` | 2304 passed, 1 failed | 2304 passed, 1 failed |
| `tests/unit/distributed` | 577 passed, 3 failed | 577 passed, 1 failed |
| `tests/unit/data` | 834 passed, 1 failed | 834 passed, 1 failed |
| `tests/unit/test_config_validation.py` | 443 passed | 443 passed |

The failures are the environment, not the change. The same ids fail on both sides: 23 tests need the gated `meta-llama/Llama-3.2-1B` repository, one needs `torchaudio`, and `test_drain_payload_metrics_returns_collector_interval` fails on `main` too. The ones that differ are port collisions in tests that pick a random port (`EADDRINUSE` in `dist.init_process_group`), on different tests in every run and on either side; the three that hit it on the branch pass when rerun alone.

**Not run:**
- `tests/unit/data_plane` beyond the files that do not start the transfer queue (50 passed here, 49 on `main`): the image ships `transfer_queue` 0.1.9 and `main` needs 0.1.11. This includes `test_seqpack_equivalence.py`, which covers the shard assembly this PR changes.
- `tests/unit/experience` and a few generation / environment test modules that do not import in that image.
- The functional test suite, and the repo's CI images (`uv sync` environment).

# Benchmarks

GB300 nodes (4 GPUs of 284 GB each), bf16, identical batches for both sides, static CP at the same per-GPU token capacity. Measured on Megatron-Bridge `c77206a` (newer than `main`'s pin); the first row was repeated on `main`'s pin and gave 4.82 s against 2.82 s per step (1.66x), with loss curves within 0.0003 of each other. Qwen3-32B has its own section below.

| Model | Workload | GPUs | Static CP | Dynamic CP | Speed-up |
|---|---|---|---|---|---|
| Qwen2.5-7B | `examples/run_sft.py`, Tulu-3 mixture, 65536-token cap | 4 | CP4: 4.67 s/step | 2.83 s/step | 1.62x |
| Qwen2.5-7B | SFT policy step, Tulu-3 lengths | 8 / 16 / 32 / 64 | CP4: 6.08 / 6.78 / 8.46 / 10.08 s | 3.25 / 3.70 / 5.09 / 6.32 s | 1.87x / 1.83x / 1.66x / 1.60x |
| Qwen2.5-7B | SFT policy step, Tulu-3 lengths, 131072-token cap | 8 | CP8: 8.85 s | 3.26 s | 2.72x |
| Qwen2.5-7B | GRPO policy step (logprobs + reference logprobs + train), Tulu-3 lengths | 4 / 8 / 16 | CP4: 12.37 / 13.84 / 19.27 s | 7.46 / 9.89 / 13.98 s | 1.66x / 1.40x / 1.38x |
| Qwen2.5-7B | SFT policy step, every sequence needs CP (CoderForge lengths) | 4 / 8 / 16 | CP4: 6.68 / 7.18 / 7.15 s | 6.73 / 7.20 / 7.64 s | 0.99x / 1.00x / 0.94x |

The gain is the CP overhead short sequences no longer pay, so it grows with the share of tokens in sequences that fit on one GPU and with the static CP size being replaced. Two things reviewers should know:

- When nearly every sequence needs the whole CP group there is nothing to remove: it matches static CP on 4-8 GPUs and is up to 9% behind on 16 GPUs, in steps where Megatron-Core's scheduler widens a leftover pack to a group larger than any sequence needs.
- Megatron-Core's scheduler runs on the driver's critical path and its time grows about 4-5x per doubling of GPUs (0.09 s at 16 GPUs, 1.5-1.7 s at 64 GPUs with ~4k sequences). The schedule reuse here removes the repeats within a GRPO step; a fix of the scheduler itself belongs in Megatron-LM.

# Qwen3-32B results

A real post-training run with dynamic CP, a static CP4 run of its first 10 steps, and replays of the run's batches on every static setup that fits. GB300 nodes (4 GPUs of 284 GB each), bf16, Megatron-Bridge `c77206a` (newer than `main`'s pin) unless a table says otherwise. In the tables, "tokens per GPU" is `sequence_packing.train_mb_tokens` / `logprob_mb_tokens` and "recomputation" is `megatron_cfg.activation_checkpointing`.

In short:

- 40 GRPO steps took AIME 2024 accuracy from 13.4% to 61.2%.
- Against static CP4 at the same per-GPU capacity, logprobs + training took 32.3 s instead of 59.6 s per step (1.85x). The whole step is mostly rollout generation and went from 196.8 s to 172.2 s (1.14x).
- Static CP4 is not the strongest static baseline on these GPUs. Static CP2 and no CP at larger per-GPU capacities also fit a 16384-token cap, and dynamic CP is level with them, not faster. It then saves memory: 188 GiB peak against 276 GiB without CP.
- Where CP is required (65536-token cap, 64 GPUs), dynamic CP is 1.62x faster than static CP8 and 1.20x faster than the best static setup that fits.
- One open question about gradient norms under CP at 8192 tokens per GPU. Static CP2 on `main` shows it too; see the last subsection.

## Setup of the run

- `Qwen/Qwen3-32B`, thinking disabled. GRPO in the DAPO style: no KL penalty, ratio clip 0.2 / 0.28, truncated importance sampling, rewards scaled to [-1, 1], overlong penalty.
- Training prompts: DAPO-Math-17K. Validation: AIME 2024, 960 samples per evaluation, every 10 steps.
- 64 prompts x 16 generations = 1024 rollouts per step, learning rate 2e-6, 40 steps, one seed.
- 4 nodes (16 GPUs), TP 1, activation recomputation, 4096 tokens per GPU, sequences up to 16384 tokens, `dynamic_context_parallel: true`. Rollouts are generated by the Megatron policy workers.
- Static baseline: the same run with `context_parallel_size: 4` at 4096 tokens per GPU, stopped after 10 steps.

## Reward gain

| Steps | Share of rollouts correct, training prompts | Mean response tokens | AIME 2024 accuracy after the last step |
|---|---|---|---|
| before training | - | - | 13.4% |
| 1-5 | 32.4% | 1623 | - |
| 6-10 | 47.6% | 1696 | 28.6% |
| 11-15 | 51.6% | 1884 | - |
| 16-20 | 60.0% | 2507 | 40.6% |
| 21-25 | 67.0% | 3411 | - |
| 26-30 | 72.2% | 4648 | 60.4% |
| 31-35 | 71.9% | 5402 | - |
| 36-40 | 72.2% | 6141 | 61.2% |

Both curves level off after step 30. The model also learned to write much longer answers, which is what makes later steps slow: rollouts + logprobs + training took 172 s per step over steps 2-10 and 534 s over steps 31-40.

Rollout lengths over the whole run (40960 rollouts): mean 3580 tokens, median 2269, 90th percentile 8457, 99th percentile 15545, maximum 15789. Sequences longer than 4096 tokens hold 9% of the tokens in the first step, 62% in step 23 and 85% in step 40.

The static CP4 run gives the same rewards, within 2 points at every step:

<details>
<summary>Share of rollouts correct per step, static CP4 against dynamic CP</summary>

| Step | Static CP4 | Dynamic CP |
|---|---|---|
| 1 | 31.1% | 29.9% |
| 2 | 29.2% | 29.7% |
| 3 | 35.3% | 33.5% |
| 4 | 28.1% | 29.5% |
| 5 | 39.8% | 39.3% |
| 6 | 41.1% | 40.2% |
| 7 | 51.0% | 49.1% |
| 8 | 48.1% | 49.6% |
| 9 | 49.0% | 48.8% |
| 10 | 50.3% | 50.1% |

</details>

## Speed in the run

Steps 2-10 of both runs, seconds per step:

| Phase | Static CP4 | Dynamic CP | Speed-up |
|---|---|---|---|
| Logprobs | 15.6 | 7.1 | 2.20x |
| Training | 44.0 | 25.2 | 1.75x |
| Logprobs + training | 59.6 | 32.3 | 1.85x |
| Rollout generation (not affected by CP) | 129.4 | 133.9 | - |
| Preparation for generation and logprobs | 7.8 | 6.0 | - |
| Whole step, without validation and checkpoint saves | 196.8 | 172.2 | 1.14x |

Rollout generation on the Megatron backend is 66% of the static step and 78% of the dynamic one, so the whole step gains much less than the phases CP touches. Over steps 31-40 of the dynamic-CP run, generation is 357.6 s, logprobs 36.5 s and training 130.7 s per step; no static run reached those steps, so the step-23 replay below is the comparison for late batches.

## Every setup on two real batches of the run

Static CP4 at 4096 tokens per GPU is the comparison at equal per-GPU capacity, but with recomputation one of these GPUs holds 8192 or even 16384 tokens of Qwen3-32B, so static CP2 and no CP also fit sequences of 16384 tokens. Each setup below ran the recorded batch of the run's first step and of its step 23 (the 1024 rollouts with their advantages, logprobs + training with the run's loss, pretrained weights, learning rate frozen), 16 GPUs. Cells are logprobs + training = total, seconds per step.

| Setup | Tokens per GPU | Recomputation | First-step batch (1.88M tokens) | Peak GPU memory | Step-23 batch (3.94M tokens) | Peak GPU memory | Gradient norm, first-step / step-23 batch |
|---|---|---|---|---|---|---|---|
| Static CP4 | 4096 | yes | 13.2 + 39.1 = **52.3** | 186 GiB | 27.2 + 90.9 = **118.1** | 186 GiB | 0.1026 / 0.0805 |
| Static CP4 | 4096 | no | 12.8 + 31.2 = **43.9** | 242 GiB | 21.4 + 61.4 = **82.9** | 225 GiB | 0.1026 / 0.0805 |
| Static CP2 | 8192 | yes | 7.8 + 26.0 = **33.8** | 231 GiB | 14.8 + 55.3 = **70.1** | 186 GiB | 0.1037 / 0.2294 |
| No CP | 16384 | yes | 7.8 + 25.4 = **33.2** | 276 GiB | 14.4 + 52.6 = **67.0** | 220 GiB | 0.1022 / 0.0810 |
| Dynamic CP | 4096 | yes | 7.4 + 25.9 = **33.3** | 188 GiB | 19.2 + 65.1 = **84.3** | 188 GiB | 0.1025 / 0.0809 |
| Dynamic CP | 8192 | yes | 7.0 + 23.4 = **30.4** | 188 GiB | 14.2 + 52.8 = **66.9** | 188 GiB | 0.1022 / 0.2289 |
| Dynamic CP | 4096 | no | 7.6 + 20.2 = **27.7** | 236 GiB | 25.0 + 53.4 = **78.4** | 236 GiB | 0.1025 / 0.0809 |

- Against static CP4 with the same settings, dynamic CP is 1.57x faster on the first-step batch and 1.40x on the step-23 batch. With its best setting it is 1.89x and 1.77x faster.
- Against the best static setup that fits (no CP at 16384 tokens per GPU, or static CP2), dynamic CP is level: 30-33 s against 33-34 s on the early batch, 67 s against 67-70 s on the late one. On this hardware and this length cap CP is not needed, and dynamic CP then costs nothing.
- Where dynamic CP is ahead outright: it peaks at 188 GiB where no CP peaks at 276 GiB, close to the limit of the GPU. That leaves room to switch recomputation off, which is 1.20x faster than the best static setup on the early batch (27.7 s against 33.2 s).
- The gradient norms agree within 1.5% across setups, except in the two that put 8192 tokens on a GPU under CP on the step-23 batch (0.229 against 0.081); that is the open question below.
- The per-GPU capacity has to follow the data. At 4096 tokens per GPU the late batch is slower with dynamic CP (84.3 s) than without CP (67.0 s); at 8192 it matches. The run above used 4096 throughout, which was right for its first steps and too small for its last ones.

## Node count

Replays of the run's early rollout lengths (random tokens with those lengths, the same batches for every setup; the number of sequences per step grows with the node count).

Static CP4 against dynamic CP, both at 4096 tokens per GPU with recomputation:

| Nodes (GPUs) | Sequences per step | SFT step | GRPO step without reference pass | GRPO step with reference pass |
|---|---|---|---|---|
| 4 (16) | 1024 | 40.1 -> 27.4 s (1.46x) | 68.7 -> 34.5 s (1.99x) | 91.1 -> 52.3 s (1.74x) |
| 8 (32) | 2048 | 54.9 -> 29.0 s (1.90x) | 58.4 -> 37.6 s (1.55x) | - |
| 16 (64) | 4096 | 45.7 -> 31.6 s (1.45x) | 72.2 -> 41.9 s (1.72x) | - |

The two missing cells ran the host out of memory: the reference model is kept in host RAM.

Every setup, GRPO step without the reference pass, seconds:

| Setup | Tokens per GPU | Recomputation | 4 nodes (16 GPUs) | 8 nodes (32 GPUs) | 16 nodes (64 GPUs) |
|---|---|---|---|---|---|
| Static CP4 | 4096 | yes | 68.7 | 58.4 | 72.2 |
| Static CP2 | 8192 | yes | 36.4 | 36.1 | 37.4 |
| No CP | 16384 | yes | 34.3 | 34.5 | 35.9 |
| Dynamic CP | 4096 | yes | 34.5 | 37.6 | 41.9 |
| Dynamic CP | 8192 | yes | 33.0 | 33.0 | 37.0 |
| Dynamic CP | 4096 | no | 35.4 | 36.5 | 38.6 |

The picture is the same at 8 and 16 nodes: well ahead of static CP4, level with static CP2 and no CP. At 64 GPUs dynamic CP at 4096 tokens per GPU is behind no CP (41.9 s against 35.9 s); part of its step there is Megatron-Core's scheduler on the driver, 1.5-1.7 s per call.

## Where CP is required

Tulu-3 lengths with a 65536-token cap do not fit on one GPU, so some CP is needed. SFT step, 16 nodes (64 GPUs):

| Setup | Tokens per GPU | Seconds per step | Of which on the workers | Step-0 loss | Step-0 gradient norm |
|---|---|---|---|---|---|
| Static CP8 | 8192 | 29.9 | 28.3 | 13.501937 | 42.906 |
| Static CP4 | 16384 (at the memory limit) | 22.1 | 20.8 | 13.501924 | 43.037 |
| Dynamic CP | 8192 | 18.4 | 15.4 | 13.501875 | 42.949 |

Dynamic CP is 1.62x faster than static CP8 at the same per-GPU capacity and 1.20x faster than the best static setup that fits. About 1.8 s of its step is Megatron-Core's scheduler on the driver.

## Same numbers on real batches

The first-step batch of the run at identical weights, 16 GPUs:

| | No CP | Dynamic CP, 4096 per GPU | Static CP4 |
|---|---|---|---|
| Loss | -0.0018244 | -0.0018225 | -0.0018414 |
| Gradient norm | 0.10217 | 0.10249 | 0.10256 |
| Logprobs, mean absolute difference to no CP | - | 0.0013 | 0.0093 |

## Open question: gradient norm under CP at 8192 tokens per GPU

Not addressed in this PR, because static CP on `main` shows it too. On the step-23 batch, the two setups that put 8192 tokens on a GPU under CP report a policy-gradient norm of 0.23 where the others report 0.081 (16 GPUs, every setup run twice):

| Setup | Policy-gradient loss | Gradient norm | Next-token loss on the same tokens | Its gradient norm |
|---|---|---|---|---|
| No CP, 16384 per GPU | 0.093271 | 0.08095 / 0.08095 | 0.497163 | 1.57617 / 1.57619 |
| Static CP4, 4096 per GPU | 0.093253 | 0.08050 / 0.08050 | 0.497165 | 1.57609 / 1.57609 |
| Dynamic CP, 4096 per GPU | 0.093247 | 0.08089 / 0.08089 | 0.497175 | 1.57580 / 1.57580 |
| Static CP2, 8192 per GPU | 0.093251 | 0.23038 / 0.23229 | 0.497173 | 1.57955 / 1.57952 |
| Dynamic CP, 8192 per GPU | 0.093255 | 0.22935 / 0.23295 | 0.497174 | 1.58016 / 1.58013 |

- The losses agree everywhere and the logprob pass is bit-for-bit repeatable in all five setups, so the forward pass is not the cause.
- In the two 8192-per-GPU setups the training step is not exactly repeatable. Their gradient differs from the others by a vector of norm about 0.1-0.2: 0.2% of a next-token gradient of 1.58, but the larger part of a policy gradient of 0.08.
- It reproduces on the Megatron commits `main` pins, with static CP2 as well as dynamic CP, so it is neither this change nor the Megatron version.
- It does not reproduce on Qwen2.5-7B with 4 GPUs, where all setups agree within 0.6%. Whether it follows Qwen3-32B or the 16-GPU layout is open, and so is the cause. It needs its own issue.
- The run above used 4096 tokens per GPU, where the gradients match no CP on both batches.

<details>
<summary>The same batch on <code>main</code>'s pins (Qwen3-32B, 16 GPUs) and on Qwen2.5-7B (4 GPUs)</summary>

Qwen3-32B, 16 GPUs, this branch on `main`'s Megatron-Bridge `ae0a1b02` / Megatron-LM `e0a4bfed`:

| Setup | Policy-gradient loss | Gradient norm (run twice) |
|---|---|---|
| No CP, 16384 per GPU | 0.093271 | 0.08096 / 0.08096 |
| Static CP2, 8192 per GPU | 0.093251 | 0.23118 / 0.23135 |
| Dynamic CP, 8192 per GPU | 0.093255 | 0.23326 / 0.23027 |

Qwen2.5-7B, 4 GPUs, `main`'s pins:

| Setup | Policy-gradient loss | Gradient norm (run twice) |
|---|---|---|
| No CP, 16384 per GPU | 0.087903 | 0.14470 / 0.14473 |
| No CP, no recomputation | 0.087903 | 0.14470 / 0.14473 |
| Static CP4, 4096 per GPU | 0.087915 | 0.14510 / 0.14510 |
| Dynamic CP, 4096 per GPU | 0.087914 | 0.14549 / 0.14553 |
| Static CP2, 8192 per GPU | 0.087908 | 0.14472 / 0.14473 |
| Static CP2, no recomputation | 0.087908 | 0.14472 / 0.14473 |
| Dynamic CP, 8192 per GPU | 0.087901 | 0.14495 / 0.14494 |

</details>

## Limits

- One run, one seed, 40 steps. The reward curve is a demonstration, not a tuned recipe.
- The static CP4 run covers steps 1-10 only; later batches are compared by replay.
- All per-setup timings are on two batches of this run. Other length distributions shift the balance between the setups.

# Before your PR is "Ready for review"
**Pre checks**:
- [x] Make sure you read and followed [Contributor guidelines](/NVIDIA-NeMo/RL/blob/main/CONTRIBUTING.md)
- [x] Did you write any new necessary tests?
- [ ] Did you run the unit tests and functional tests locally? Visit our [Testing Guide](/NVIDIA-NeMo/RL/blob/main/docs/testing.md) for how to run tests
- [x] Did you add or update any necessary documentation? Visit our [Document Development Guide](/NVIDIA-NeMo/RL/blob/main/docs/documentation.md) for how to write, build and test the docs.

# Additional Information

- Not exercised on GPUs: MoE models, multi-token prediction, LoRA, vision inputs, a data-parallel rank count that is not a power of two.
- Known Megatron-Core behaviour worked around here: for packs of very small sequences the scheduler can emit a group that straddles two of the aligned rank blocks the runtime groups are built from; `schedule_dynamic_cp` orders each round largest group first, which avoids it.
- Possible follow-ups: return the logprobs of a replicated pack from its first rank only; data-plane support.
- The gradient-norm question in the Qwen3-32B section is not caused by this PR and is not addressed by it.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
