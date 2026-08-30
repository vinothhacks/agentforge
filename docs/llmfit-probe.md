# llmfit probe (M0)

Binary: `C:\Users\sm2063\.local\bin\llmfit.exe` (PATH).  
Not present: `%USERPROFILE%\.cargo\bin\llmfit.exe`, repo `target/release/llmfit.exe`.  
Version: **llmfit 1.1.10**

TUI was not scraped. JSON / `--help` only.

## GATE: `ollama_name`

| Command | Field `ollama_name` |
| --- | --- |
| `llmfit --json fit -n 5` (and `-n 80`) | **Key present on every row. Value always `null`.** 0/80 non-null. |
| `llmfit --json list` (11271 models) | **Key absent.** 0/11271 have `ollama_name`. |

**Decision:** Do not invent Ollama tags from HuggingFace names. Catalog source (b) `PULLABLE` is only rows whose `ollama_name` is a non-empty string. Source (a) `ollama list` remains `INSTALLED`. Source (c) llmfit rows without `ollama_name` are `NO_RUNTIME`.

## `llmfit download` output dir (this version)

**Yes.** `--output-dir <PATH>` overrides `~/.cache/llmfit/models/`.  
Help: "Pass --output-dir to write to a different location". No `--json` support for download progress (parse stdout). Download is GGUF for llama.cpp, not `ollama pull`.

## `llmfit --help` (raw)

```
Right-size LLM models to your system's hardware.
...
Commands: system, doctor, claim, list, fit, search, info, diff, plan,
recommend, download, hf-search, update, run, serve, bench, help
Global: --json, --memory, --ram, --cpu-cores, --max-context, --no-dashboard
```

## `llmfit download --help` (raw excerpt)

```
Usage: llmfit.exe download [OPTIONS] <MODEL>
  -q, --quant <QUANT>
      --budget <GB>
      --list
      --output-dir <PATH>
```

## `llmfit --json system` (raw)

```json
{
  "system": {
    "available_ram_gb": 5.03,
    "backend": "Vulkan",
    "cpu_cores": 12,
    "cpu_name": "12th Gen Intel(R) Core(TM) i7-1255U",
    "gpu_available_gb": null,
    "gpu_count": 1,
    "gpu_name": "Name",
    "gpu_vram_gb": null,
    "gpus": [
      {
        "backend": "Vulkan",
        "count": 1,
        "memory_bandwidth_gbps": null,
        "name": "Name",
        "unified_memory": false,
        "vram_gb": null
      }
    ],
    "has_gpu": true,
    "total_ram_gb": 31.69,
    "unified_memory": false
  }
}
```

**system field names:** `available_ram_gb`, `backend`, `cpu_cores`, `cpu_name`, `gpu_available_gb`, `gpu_count`, `gpu_name`, `gpu_vram_gb`, `gpus[]` (`backend`, `count`, `memory_bandwidth_gbps`, `name`, `unified_memory`, `vram_gb`), `has_gpu`, `total_ram_gb`, `unified_memory`.

## `llmfit --json fit -n 5` — first row field names

`best_quant`, `capabilities`, `capability_ids`, `category`, `context_length`, `disk_size_gb`, `effective_context_length`, `estimate_basis`, `estimated_tps`, `fit_label`, `fit_level`, `gguf_sources`, `installed`, `is_moe`, `license`, `measured_tps`, `memory_available_gb`, `memory_required_gb`, `moe_offloaded_gb`, `name`, `notes`, **`ollama_name`**, `parameter_count`, `params_b`, `provider`, `release_date`, `run_mode`, `run_mode_label`, `runtime`, `runtime_label`, `score`, `score_components`, `supports_tp`, `total_memory_gb`, `usable_context`, `use_case`, `utilization_pct`, `verify_command`.

Sample (truncated): `name=marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B`, `ollama_name=null`, `fit_level=Good`, `best_quant=Q8_0`, `estimated_tps=184.8`, `measured_tps=null`, `runtime=llama.cpp`.

## `llmfit list --json` head (schema, not 21MB dump)

Top-level **JSON array** (not `{models:[]}`). First object keys:

`active_experts`, `active_parameters`, `architecture`, `attention_layout`, `capabilities`, `context_length`, `format`, `gguf_sources`, `head_dim`, `hidden_size`, `is_moe`, `languages`, `license`, `min_ram_gb`, `min_vram_gb`, `moe_intermediate_size`, `name`, `num_attention_heads`, `num_experts`, `num_hidden_layers`, `num_key_value_heads`, `parameter_count`, `parameters_raw`, `provider`, `quantization`, `recommended_ram_gb`, `release_date`, `shared_expert_intermediate_size`, `use_case`, `vocab_size`.

No `ollama_name`. First name: `darkc0de/XORTRON-NXTXPRT10PRO-31B`.

## `llmfit bench`

`llmfit bench [MODEL] --provider ollama|auto --json --runs 3`

## Install if missing

```
uv tool install llmfit
```
or put `llmfit.exe` on PATH / `%USERPROFILE%\.cargo\bin`.
