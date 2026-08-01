# Validation Status — qwen25-7b-fp8-quant-pipeline

**Model/Task:** FP8 post-training quantization of Qwen2.5-7B-Instruct via nvidia-modelopt on DGX Spark
**Platform:** Kubeflow Pipelines on NVIDIA DGX Spark (GB10, 128 GB unified memory)
**Last updated:** {{DATE}}

---

## Current Status

| Component             | Status                                   |
| ---------------------- | ---------------------------------------- |
| `download_model`       | ✅ Runs and completes                     |
| `baseline_eval`        | ✅ Runs and completes                     |
| `quantize`             | ✅ Runs and completes                     |
| `post_quant_eval`      | ✅ Runs and completes                     |
| `quantization_gate`    | ✅ Passed on run-002 (accuracy_delta -0.000586, threshold -0.02)          |

**Project completed a full, genuinely fresh end-to-end run (run-002) with the accuracy gate
passing.** The KFP execution-caching bug (see Fixed Issues) was the blocker on run-001; once
fixed, run-002 quantized cleanly with `kv_cache_qformat: none` and passed the gate with
essentially free accuracy. Independently smoke-tested via plain `vllm serve` (no
`--quantization=fp8` flag needed — vLLM auto-detects FP8 from the modelopt-exported config).

---

## Run Table

| Run | Purpose | Result | Key Finding |
| --- | ------- | ------ | ----------- |
| run-001 (attempt 1) | First end-to-end run | FAIL (infra) | `post_quant_eval` OOMKilled — pod memory limit `64G` too low for vLLM torch.compile/cudagraph capture + lm_eval in one container. Fixed: bumped to `96G`. |
| run-001 (attempt 2) | Retry after memory fix | FAIL (infra) | `post_quant_eval` `ModuleNotFoundError: tenacity` — `lm_eval`'s `local-completions` API model needs the `[api]` extra. Fixed: `lm_eval` → `lm_eval[api]`. |
| run-001 (attempt 3) | Retry after packages fix | FAIL (gate) | Accuracy delta -0.1831 (baseline 0.8092 HFLM vs quantized 0.6260 local-completions) — traced to an eval-backend mismatch: baseline used in-process `HFLM`, post-quant used vLLM-served `local-completions`. Fixed: rewrote `baseline_eval` to use the same vLLM-serve + `local-completions` path as `post_quant_eval`. |
| run-001 (attempt 4, purged) | Retry with matched eval backends | FAIL (gate) | Accuracy delta -0.1793 (baseline 0.8053, quantized 0.6260 — **identical quantized score to attempt 3**). Workflow node timeline showed `quantize` and `post_quant_eval` executors as `Skipped` (cache hits, 0s duration) while only `baseline_eval` (which had a real code change) actually ran a container. KFP's execution cache — separate from `purge_kfp_mlflow.py`'s KFP-run/MLflow purge — silently reused stale `quantize`/`post_quant_eval` outputs from before the memory-limit and `lm_eval[api]` fixes. **We have never actually observed a fresh `quantize`+`post_quant_eval` pair in the same run.** Fixed: `.set_caching_options(enable_caching=False)` added to `baseline_eval`, `quantize`, `post_quant_eval` in the pipeline assembly cell — verified in the compiled IR (`cachingOptions: {}` for all three, vs. `{enableCache: true}` for `download_model`/`quantization_gate`, which are fine to keep cached). |
| run-002 | First genuinely fresh run with caching disabled | FAIL (gate) | Accuracy delta -0.1865 with `kv_cache_qformat: fp8` — unstable FP8 KV-cache quantization on Qwen2.5-7B caused a real (not infra-caused) accuracy collapse. Fixed: `kv_cache_qformat: none` (quantize weights/activations to FP8, leave KV cache at native precision). |
| run-002 (rerun) | Retry with `kv_cache_qformat: none` | **PASS** | Accuracy delta -0.000586 (baseline 0.805299, quantized 0.804714) — well within the -0.02 threshold. All 5 DAG nodes succeeded. Checkpoint verified at contract path; independently smoke-tested with plain `vllm serve` (health check, `/v1/models`, and a chat completion all succeeded). |

> Update this table after each run.

---

## What Is Implemented

### Infrastructure (inherited from platform template)
- KFP v2 pipeline with GPU component using NGC PyTorch base image
- `purge_kfp_mlflow.py`
- Nsight Operator integration — add `kubernetes.add_pod_label(task, "nvidia-nsight-profile", "enabled")` to profile any stage

### Project-specific
- `download_model` → `baseline_eval` → `quantize` → `post_quant_eval` → `quantization_gate` DAG
- `vendor/hf_ptq.py`, `vendor/example_utils.py`, `vendor/cast_mxfp4_to_nvfp4.py` — vendored
  NVIDIA Model-Optimizer PTQ scripts, injected into the generated `pipeline.py` by
  `scripts/build_pipeline.py`
- `notebook.ipynb` — all 5 components + pipeline assembly cell

---

## What Is Still Pending

- Nothing blocking — pipeline validated end-to-end on run-002. Scaffolding a `serving-trt-fp8`
  sibling project to serve this checkpoint is a separate, not-yet-approved task.

---

## Known Issues

- `purge_kfp_mlflow.py` clears KFP runs, pipeline versions, the pipeline registration, orphaned
  Argo workflows, and MLflow runs/experiment — but **does not** clear KFP v2's execution cache
  (a separate MLMD/cache-server-backed store keyed by component+input fingerprint, independent
  of run/pipeline IDs). A component whose logic/resources changed in a way that doesn't affect
  its cache key (e.g. `set_memory_limit`, some `packages_to_install` edits) can silently reuse a
  stale execution's outputs even immediately after a purge. Mitigated for the three GPU stages by
  explicitly disabling caching on them (see Fixed Issues), but worth remembering if new cacheable
  stages are added later.

---

## Fixed Issues

- **`post_quant_eval` OOMKilled (exit 137)** — pod memory limit `64G` too low for vLLM's
  torch.compile + cudagraph capture + lm_eval harness in one container (separate from GPU-memory
  accounting via `--gpu-memory-utilization`). Fixed: `set_memory_limit("64G")` → `"96G"` on
  `baseline_eval`, `quantize`, `post_quant_eval`.
- **`post_quant_eval` `ModuleNotFoundError: tenacity`** — `lm_eval`'s `local-completions` model
  type requires the `[api]` extra, not pulled in by plain `lm_eval`. Fixed:
  `packages_to_install=["lm_eval[api]", ...]`.
- **Accuracy gate FAIL from an eval-backend mismatch** — `baseline_eval` originally used
  in-process `HFLM`; `post_quant_eval` used vLLM-served `local-completions`. Different backends
  can diverge on loglikelihood-based multiple-choice tasks (tokenization/prompt-formatting
  differences), producing an apparent accuracy drop unrelated to quantization quality. Fixed:
  rewrote `baseline_eval` to use the identical vLLM-serve + `local-completions` path as
  `post_quant_eval`.
- **Accuracy gate FAIL from stale KFP-cached artifacts** — after matching eval backends, the
  gate still failed with an *identical* quantized score to the prior (unmatched-backend) failure.
  Workflow node timeline confirmed `quantize` and `post_quant_eval` were cache-hit-skipped
  (0-second `Skipped` executor nodes) rather than actually re-run, even after an explicit purge —
  meaning no fix applied to those two components had ever actually been exercised. Fixed:
  `.set_caching_options(enable_caching=False)` on `baseline_eval`, `quantize`, and
  `post_quant_eval` in the pipeline assembly cell (`notebook.ipynb` cell `3ea679a5`).

---

## Latest Nsight Finding

No profiling runs yet.
