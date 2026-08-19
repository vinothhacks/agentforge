# P-1 disagreement experiment

**D = P(probe_fail | catalog_yes)** = `None`

**Decision:** insufficient_n — no catalog_yes tag completed n=5. Do not treat this as D<0.10.

Repeats n=5, mode=fast, catalog_yes with n complete: 0. gate=`continue_p0_with_caveat`.

Catalog flags are read client-side from llmfit `capability_ids` (REST has no `use_case=tool_use`). Uninstalled tags = `not_installed`. OpenRouter without a key = `no_openrouter_key`.

## Findings

- Live-probed tags: 5. Passed as `agent`: gemma4:cloud.
- Live-probed `chat_only`: qwen3:4b, granite3.1-moe:latest, parable/granite4.1-fable:3b, richardyoung/qwythos-9b-abliterated:IQ3_M.
- `D` is undefined until at least one **catalog_yes** tag completes n=5. That is not D<0.10. llmfit did not flag the installed Ollama tags as `tool_use`, which is the catalog hole this experiment exists to measure.

| tag | provider | catalog | pass/n | lo95 | verdict | cause |
| --- | --- | --- | --- | --- | --- | --- |
| qwen3:4b | ollama | unknown | 0/5 | 0.0 | chat_only | no_tool_call |
| granite3.1-moe:latest | ollama | unknown | 0/5 | 0.0 | chat_only | bad_tool |
| parable/granite4.1-fable:3b | ollama | unknown | 0/5 | 0.0 | chat_only | bad_tool |
| richardyoung/qwythos-9b-abliterated:IQ3_M | ollama | unknown | 0/5 | 0.0 | chat_only | probe_timeout |
| llama3.1:8b | ollama | unknown | None | None | not_installed | not_installed |
| llama3.2:3b | ollama | unknown | None | None | not_installed | not_installed |
| mistral:7b | ollama | unknown | None | None | not_installed | not_installed |
| qwen2.5:7b | ollama | unknown | None | None | not_installed | not_installed |
| qwen2.5-coder:7b | ollama | unknown | None | None | not_installed | not_installed |
| phi3:mini | ollama | unknown | None | None | not_installed | not_installed |
| gemma2:9b | ollama | unknown | None | None | not_installed | not_installed |
| gemma3:4b | ollama | unknown | None | None | not_installed | not_installed |
| command-r:35b | ollama | unknown | None | None | not_installed | not_installed |
| deepseek-r1:7b | ollama | unknown | None | None | not_installed | not_installed |
| nomic-embed-text | ollama | unknown | None | None | not_installed | not_installed |
| tinyllama | ollama | no | None | None | not_installed | not_installed |
| codellama:7b | ollama | unknown | None | None | not_installed | not_installed |
| vicuna:7b | ollama | unknown | None | None | not_installed | not_installed |
| openai/gpt-4o-mini | openrouter | unknown | None | None | no_openrouter_key | no_openrouter_key |
| anthropic/claude-3.5-sonnet | openrouter | unknown | None | None | no_openrouter_key | no_openrouter_key |
| google/gemini-2.0-flash-001 | openrouter | unknown | None | None | no_openrouter_key | no_openrouter_key |
| meta-llama/llama-3.1-8b-instruct | openrouter | unknown | None | None | no_openrouter_key | no_openrouter_key |
| qwen/qwen-2.5-7b-instruct | openrouter | unknown | None | None | no_openrouter_key | no_openrouter_key |
| not-a-real-model-xyz | ollama | unknown | None | None | not_installed | not_installed |
| gemma4:cloud | ollama | unknown | 5/5 | 0.5655 | agent | None |

## Gate rule (from the plan)

- `D < 0.10` - feature; stop or shrink to the document template.
- `D >= 0.25` — catalog lies often; the probe is the product. Continue P0.
- `0.10 <= D < 0.25` — probe is a gate, not the headline. Continue P0; do not market certification.

User instruction for this implementation pass: complete P0–P3 regardless. This file is the interview artifact. Do not market certification unless gate=product.

## Design (25 tags)

Mix of installed Ollama instruct tags, catalog tool_use yes/no, unknown, OpenRouter allowlist ids, one abliterated/custom-template tag, one embedding tag, one fake name, one `:cloud` tag.
