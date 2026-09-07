"""Merge ollama list + llmfit fit rows into one catalog table."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import httpx

from gateway import llmfit_client

State = Literal["INSTALLED", "PULLABLE", "NO_RUNTIME", "UNRESOLVABLE"]

FITS_RANK = {
    "perfect": 4,
    "good": 3,
    "marginal": 2,
    "tight": 1,
    "wont_fit": 0,
    "unknown": -1,
}

SHARD_RE = re.compile(r"(\d{5}-of-\d{5})|(-of-\d+\.gguf)|(\.gguf\.part)", re.I)


def cache_dirs() -> dict[str, Path]:
    home = Path.home()
    hf = Path(os.environ.get("HF_HOME") or (home / ".cache" / "huggingface" / "hub"))
    ollama = Path(os.environ.get("OLLAMA_MODELS") or (home / ".ollama" / "models"))
    llama = Path(os.environ.get("LLAMA_CACHE") or (home / ".cache" / "llama.cpp"))
    llmfit = home / ".cache" / "llmfit" / "models"
    return {"llmfit": llmfit, "huggingface": hf, "ollama": ollama, "llama": llama}


def already_downloaded_names(dirs: dict[str, Path] | None = None) -> set[str]:
    found: set[str] = set()
    roots = dirs or cache_dirs()
    ollama_root = roots["ollama"]
    if ollama_root.is_dir():
        for p in ollama_root.rglob("*"):
            if p.is_file():
                found.add(p.name.lower())
                found.add(p.stem.lower())
    for key in ("llmfit", "huggingface", "llama"):
        root = roots[key]
        if not root.is_dir():
            continue
        for p in root.rglob("*"):
            if p.is_file():
                found.add(p.name.lower())
                found.add(p.stem.lower())
    return found


def is_sharded_gguf(sources: Any) -> bool:
    blob = " ".join(str(s) for s in sources) if isinstance(sources, list) else str(sources or "")
    return bool(SHARD_RE.search(blob))


def looks_like_gguf_repo(name: str, gguf_sources: Any = None) -> bool:
    """Heuristic: repo likely has GGUF files (llmfit download target)."""
    sources = gguf_sources if isinstance(gguf_sources, list) else []
    if any(str(s).strip().lower().endswith(".gguf") for s in sources):
        return True
    n = (name or "").strip().lower()
    if n.endswith("-gguf") or n.endswith("/gguf"):
        return True
    if "/gguf" in n or "-gguf/" in n:
        return True
    return False


def gguf_downloadable(name: str, gguf_sources: Any = None, *, sharded: bool = False) -> bool:
    if sharded:
        return False
    return looks_like_gguf_repo(name, gguf_sources)


def parse_memory_gb(memory: str | None) -> float | None:
    if memory is None or str(memory).strip() == "":
        return None
    raw = str(memory).strip().upper().replace("GIB", "").replace("GB", "").replace("GI", "").replace("G", "")
    try:
        return float(raw)
    except ValueError:
        return None


def _norm_cache_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-")


def in_download_cache(name: str, ollama_name: str | None, downloaded: set[str]) -> bool:
    found = {_norm_cache_key(x) for x in downloaded if x}
    needles: set[str] = {_norm_cache_key(name)}
    if "/" in name:
        needles.add(_norm_cache_key(name.rsplit("/", 1)[-1]))
    if ollama_name:
        needles.add(_norm_cache_key(ollama_name))
        needles.add(_norm_cache_key(ollama_name.split(":")[0]))
    needles = {n for n in needles if len(n) >= 4}
    for n in needles:
        if n in found:
            return True
        for f in found:
            if f == n or f.startswith(n + "-") or n.startswith(f + "-"):
                return True
    return False


def supports_tools(raw: dict[str, Any]) -> bool:
    ids = [str(c).lower() for c in (raw.get("capability_ids") or [])]
    caps = [str(c).lower() for c in (raw.get("capabilities") or [])]
    blob = ids + caps
    tokens = {"tool_use", "tools", "tool-use", "function_calling", "function-calling"}
    return any(t in tokens or "tool" in t for t in blob)


def fit_label(size_gb: float, available_gb: float) -> str:
    if size_gb <= 0 or available_gb <= 0:
        return "unknown"
    ratio = size_gb / available_gb
    if ratio <= 0.45:
        return "perfect"
    if ratio <= 0.75:
        return "good"
    if ratio <= 0.95:
        return "marginal"
    if ratio <= 1.15:
        return "tight"
    return "wont_fit"


def _ollama_tags(base: str = "http://127.0.0.1:11434") -> list[dict[str, Any]]:
    try:
        r = httpx.get(f"{base}/api/tags", timeout=5.0)
        r.raise_for_status()
        payload = r.json()
    except Exception:  # noqa: BLE001
        return []
    rows = []
    for m in payload.get("models") or []:
        size = float(m.get("size") or 0)
        rows.append(
            {
                "name": m.get("name") or m.get("model"),
                "size_gb": round(size / (1024**3), 2) if size else None,
            }
        )
    return [r for r in rows if r.get("name")]


def ollama_show(name: str, base: str = "http://127.0.0.1:11434") -> dict[str, Any]:
    try:
        r = httpx.post(f"{base}/api/show", json={"name": name}, timeout=10.0)
        r.raise_for_status()
        return r.json()
    except Exception:  # noqa: BLE001
        return {}


def chat_template_ok(show: dict[str, Any]) -> bool | None:
    if not show:
        return None
    tmpl = show.get("template") or (show.get("details") or {}).get("family")
    if tmpl is None:
        return False
    if isinstance(tmpl, str) and not tmpl.strip():
        return False
    return True


def _row(
    *,
    name: str,
    source: str,
    params: str | None,
    quant: str | None,
    size: float | None,
    est_vram: float | None,
    fits: str,
    ctx: int | None,
    supports_tools: bool,
    est_tok_s: float | None,
    measured_tok_s: float | None,
    state: State,
    ollama_name: str | None = None,
    reason: str | None = None,
    sharded: bool = False,
    chat_template_ok_flag: bool | None = None,
    gguf_sources: list | None = None,
    gguf_downloadable_flag: bool | None = None,
) -> dict[str, Any]:
    sources = gguf_sources or []
    dl_ok = gguf_downloadable_flag
    if dl_ok is None and state not in {"INSTALLED", "PULLABLE"}:
        dl_ok = gguf_downloadable(name, sources, sharded=sharded)
    return {
        "name": name,
        "source": source,
        "params": params,
        "quant": quant,
        "size": size,
        "est_vram": est_vram,
        "FITS": fits,
        "ctx": ctx,
        "supports_tools": supports_tools,
        "est_tok_s": est_tok_s,
        "measured_tok_s": measured_tok_s,
        "state": state,
        "ollama_name": ollama_name,
        "reason": reason,
        "sharded": sharded,
        "chat_template_ok": chat_template_ok_flag,
        "gguf_sources": sources,
        "gguf_downloadable": dl_ok,
    }


def merge_catalog(
    *,
    ollama_models: list[dict[str, Any]] | None = None,
    fit_payload: dict[str, Any] | None = None,
    system: dict[str, Any] | None = None,
    downloaded: set[str] | None = None,
    memory: str | None = None,
    max_context: int | None = None,
    ollama_base: str = "http://127.0.0.1:11434",
    live: bool = True,
    limit: int = 80,
) -> dict[str, Any]:
    if live:
        try:
            system = system or llmfit_client.system()
        except llmfit_client.LlmfitMissing:
            system = system or {}
        except (llmfit_client.LlmfitError, llmfit_client.LlmfitSchemaError):
            system = system or {}
        if fit_payload is None:
            try:
                fit_payload = llmfit_client.fit(limit=limit, memory=memory, max_context=max_context)
            except (llmfit_client.LlmfitMissing, llmfit_client.LlmfitError, llmfit_client.LlmfitSchemaError):
                fit_payload = {"models": []}
        if ollama_models is None:
            ollama_models = _ollama_tags(ollama_base)
        if downloaded is None:
            downloaded = already_downloaded_names()
    ollama_models = ollama_models or []
    fit_payload = fit_payload or {"models": []}
    downloaded = downloaded or set()
    ram = parse_memory_gb(memory)
    if ram is None:
        # Fit is a property of the machine, not of this instant. Keying it on
        # free RAM made a 1.9 GB model read "wont_fit" on a 31.7 GB box and
        # flip between page loads.
        ram = float((system or {}).get("total_ram_gb") or (system or {}).get("available_ram_gb") or 0.0)

    installed_names = {str(m["name"]) for m in ollama_models}
    rows: list[dict[str, Any]] = []

    for m in ollama_models:
        name = str(m["name"])
        size = m.get("size_gb")
        show = ollama_show(name, ollama_base) if live else {}
        tmpl = chat_template_ok(show) if live else True
        rows.append(
            _row(
                name=name,
                source="ollama",
                params=None,
                quant=None,
                size=size,
                est_vram=size,
                fits=fit_label(float(size or 0), ram),
                ctx=None,
                supports_tools=True,
                est_tok_s=None,
                measured_tok_s=None,
                state="INSTALLED",
                ollama_name=name,
                chat_template_ok_flag=tmpl,
            )
        )

    for raw in fit_payload.get("models") or []:
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name") or "")
        if not name:
            continue
        ollama_name = raw.get("ollama_name")
        if isinstance(ollama_name, str):
            ollama_name = ollama_name.strip() or None
        else:
            ollama_name = None
        sharded = is_sharded_gguf(raw.get("gguf_sources"))
        tools = supports_tools(raw)
        size = raw.get("disk_size_gb") or raw.get("memory_required_gb")
        fits = str(raw.get("fit_level") or raw.get("fit_label") or "unknown")
        cached = in_download_cache(name, ollama_name, downloaded)
        if sharded:
            state: State = "UNRESOLVABLE"
            reason = "sharded GGUF repo; pull refused"
        elif ollama_name and (ollama_name in installed_names or cached):
            state = "INSTALLED"
            reason = "already in cache" if cached and ollama_name not in installed_names else None
        elif ollama_name:
            state = "PULLABLE"
            reason = None
        else:
            state = "NO_RUNTIME"
            reason = "no ollama_name; llama.cpp is off"
        if ollama_name and ollama_name in installed_names:
            continue
        rows.append(
            _row(
                name=name,
                source="llmfit",
                params=str(raw.get("parameter_count") or "") or None,
                quant=raw.get("best_quant"),
                size=float(size) if size is not None else None,
                est_vram=raw.get("memory_required_gb"),
                fits=fits,
                ctx=raw.get("effective_context_length") or raw.get("context_length"),
                supports_tools=tools,
                est_tok_s=raw.get("estimated_tps"),
                measured_tok_s=raw.get("measured_tps"),
                state=state,
                ollama_name=ollama_name,
                reason=reason,
                sharded=sharded,
                gguf_sources=list(raw.get("gguf_sources") or []) if isinstance(raw.get("gguf_sources"), list) else [],
            )
        )

    def sort_key(r: dict[str, Any]) -> tuple:
        fits = FITS_RANK.get(str(r.get("FITS") or "").lower(), -1)
        return (-fits, 0 if r.get("supports_tools") else 1, -(float(r.get("est_tok_s") or 0)))

    rows.sort(key=sort_key)
    return {"system": system or {}, "rows": rows, "caches": {k: str(v) for k, v in cache_dirs().items()}}


def _params_b(text: Any) -> float:
    raw = str(text or "").strip().upper().replace(" ", "")
    m = re.match(r"([0-9]*\.?[0-9]+)([BMK])?", raw)
    if not m:
        return 0.0
    n = float(m.group(1) or 0)
    unit = m.group(2) or "B"
    if unit == "M":
        return n / 1000.0
    if unit == "K":
        return n / 1_000_000.0
    return n


def _as_strs(val: Any) -> list[str]:
    if val is None:
        return []
    if isinstance(val, list):
        return [str(x).strip() for x in val if str(x).strip()]
    s = str(val).strip()
    return [s] if s else []


def browse_llmfit(
    *,
    q: str = "",
    provider: str = "",
    use_case: str = "",
    capability: str = "",
    fit: str = "",
    sort: str = "score",
    tools: bool | None = None,
    downloadable: bool | None = None,
    offset: int = 0,
    limit: int = 80,
    memory: str | None = None,
    refresh: bool = False,
) -> dict[str, Any]:
    """Full llmfit list (JSON), filtered in-process. Never scrapes the TUI."""
    try:
        system = llmfit_client.system()
    except (llmfit_client.LlmfitMissing, llmfit_client.LlmfitError, llmfit_client.LlmfitSchemaError):
        system = {}
    try:
        raw_rows = llmfit_client.list_models(refresh=refresh)
    except (llmfit_client.LlmfitMissing, llmfit_client.LlmfitError, llmfit_client.LlmfitSchemaError) as exc:
        return {"system": system, "rows": [], "total": 0, "shown": 0, "filters": {}, "error": str(exc)}

    ram = parse_memory_gb(memory)
    if ram is None:
        ram = float(system.get("total_ram_gb") or system.get("available_ram_gb") or 0.0)
    downloaded = already_downloaded_names()
    qn = (q or "").strip().lower()
    prov = (provider or "").strip().lower()
    uc = (use_case or "").strip().lower()
    cap = (capability or "").strip().lower()
    fit_want = (fit or "").strip().lower()

    providers: set[str] = set()
    use_cases: set[str] = set()
    caps: set[str] = set()
    built: list[dict[str, Any]] = []

    for raw in raw_rows:
        name = str(raw.get("name") or "")
        if not name:
            continue
        provider_s = str(raw.get("provider") or "")
        use_vals = _as_strs(raw.get("use_case"))
        cap_vals = _as_strs(raw.get("capabilities")) or _as_strs(raw.get("capability_ids"))
        if provider_s:
            providers.add(provider_s)
        use_cases.update(use_vals)
        caps.update(cap_vals)
        mem = raw.get("min_ram_gb") or raw.get("recommended_ram_gb") or raw.get("memory_required_gb")
        fits = fit_label(float(mem or 0), ram) if mem else "unknown"
        tools_ok = supports_tools(raw) or any("tool" in c.lower() for c in cap_vals)
        ollama_name = raw.get("ollama_name")
        if isinstance(ollama_name, str):
            ollama_name = ollama_name.strip() or None
        else:
            ollama_name = None
        sharded = is_sharded_gguf(raw.get("gguf_sources"))
        sources = list(raw.get("gguf_sources") or []) if isinstance(raw.get("gguf_sources"), list) else []
        dl_ok = gguf_downloadable(name, sources, sharded=sharded)
        cached = in_download_cache(name, ollama_name, downloaded)
        if sharded:
            state: State = "UNRESOLVABLE"
        elif cached and (ollama_name or dl_ok):
            # A cache hit alone is not enough to call a model installed: the
            # name match is fuzzy, and a safetensors-only repo with no Ollama
            # tag has nothing to run. Those rows rendered as READY with a
            # disabled "No GGUF files in repo" as their only action.
            state = "INSTALLED"
        elif ollama_name:
            state = "PULLABLE"
        elif not dl_ok:
            state = "UNRESOLVABLE"
        else:
            state = "NO_RUNTIME"
        reason = None
        if sharded:
            reason = "sharded GGUF repo; pull refused"
        elif state == "UNRESOLVABLE" and not sharded:
            reason = "no GGUF files in HuggingFace repo; try a *-GGUF repo"
        row = _row(
            name=name,
            source="llmfit",
            params=str(raw.get("parameter_count") or "") or None,
            quant=raw.get("quantization") or raw.get("best_quant"),
            size=float(raw.get("disk_size_gb") or mem) if (raw.get("disk_size_gb") or mem) is not None else None,
            est_vram=raw.get("min_vram_gb") or raw.get("memory_required_gb"),
            fits=fits,
            ctx=raw.get("context_length"),
            supports_tools=tools_ok,
            est_tok_s=raw.get("estimated_tps"),
            measured_tok_s=raw.get("measured_tps"),
            state=state,
            ollama_name=ollama_name,
            reason=reason,
            sharded=sharded,
            gguf_sources=sources,
            gguf_downloadable_flag=dl_ok,
        )
        row["provider"] = provider_s
        row["use_case"] = use_vals[0] if use_vals else ""
        row["capabilities"] = cap_vals
        row["_params_b"] = _params_b(raw.get("parameter_count") or raw.get("params_b"))
        row["_date"] = str(raw.get("release_date") or "")
        built.append(row)

    def keep(r: dict[str, Any]) -> bool:
        if qn and qn not in (r.get("name") or "").lower() and qn not in (r.get("provider") or "").lower():
            return False
        if prov and (r.get("provider") or "").lower() != prov:
            return False
        if uc and (r.get("use_case") or "").lower() != uc:
            return False
        if cap:
            caps_l = [c.lower() for c in (r.get("capabilities") or [])]
            if cap not in caps_l:
                return False
        if fit_want and str(r.get("FITS") or "").lower() != fit_want:
            return False
        if tools is True and not r.get("supports_tools"):
            return False
        if downloadable is True and r.get("state") == "UNRESOLVABLE":
            return False
        return True

    filtered = [r for r in built if keep(r)]

    def sk(r: dict[str, Any]) -> tuple:
        key = (sort or "score").lower()
        if key == "params":
            return (-float(r.get("_params_b") or 0), r.get("name") or "")
        if key == "mem":
            return (float(r.get("est_vram") or r.get("size") or 0), r.get("name") or "")
        if key == "ctx":
            return (-float(r.get("ctx") or 0), r.get("name") or "")
        if key == "date":
            return (r.get("_date") or "", r.get("name") or "")
        if key == "provider":
            return ((r.get("provider") or "").lower(), r.get("name") or "")
        if key == "tps":
            return (-float(r.get("est_tok_s") or 0), r.get("name") or "")
        fits = FITS_RANK.get(str(r.get("FITS") or "").lower(), -1)
        # A row the user can install beats a better-fitting one they cannot.
        actionable = 0 if r.get("state") != "UNRESOLVABLE" else 1
        return (actionable, -fits, 0 if r.get("supports_tools") else 1, (r.get("name") or ""))

    reverse_date = (sort or "").lower() == "date"
    filtered.sort(key=sk, reverse=reverse_date)
    if (sort or "").lower() == "date":
        filtered.sort(key=lambda r: r.get("_date") or "", reverse=True)

    off = max(0, int(offset or 0))
    lim = max(1, min(int(limit or 80), 200))
    page = filtered[off : off + lim]
    for r in page:
        r.pop("_params_b", None)
        r.pop("_date", None)
    return {
        "system": system,
        "rows": page,
        "total": len(filtered),
        "catalog_size": len(built),
        "offset": off,
        "limit": lim,
        "shown": len(page),
        "filters": {
            "providers": sorted(providers, key=str.lower)[:400],
            "use_cases": sorted(use_cases, key=str.lower)[:200],
            "capabilities": sorted(caps, key=str.lower)[:200],
            "downloadable": len([r for r in built if r.get("state") != "UNRESOLVABLE"]),
            "fits": ["perfect", "good", "marginal", "tight", "wont_fit", "unknown"],
            "sorts": ["score", "params", "mem", "ctx", "date", "provider", "tps"],
        },
    }
