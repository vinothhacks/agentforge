"""Hybrid RAG: one page-text pass → LanceDB vectors + SQLite FTS5. One tool: rag_search."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
from pypdf import PdfReader

from gateway.paths import safe_join

DIM = 384
CHUNK_CHARS = 1400
OVERLAP = 180
STOPWORDS = {
    "a",
    "an",
    "and",
    "all",
    "cite",
    "every",
    "file",
    "files",
    "find",
    "for",
    "from",
    "in",
    "into",
    "list",
    "mention",
    "mentioning",
    "mentions",
    "of",
    "page",
    "please",
    "show",
    "that",
    "the",
    "this",
    "those",
    "what",
    "which",
    "with",
    "workspace",
    "folder",
}
SHIP_TERMS = {
    "demurrage",
    "detention",
    "dispatch",
    "pilotage",
    "towage",
    "agency",
    "berth",
    "tonnage",
    "laytime",
    "vessel",
    "port",
    "imo",
    "dues",
    "stevedoring",
    "mooring",
}


def _search_tokens(query: str) -> list[str]:
    raw = re.findall(r"[A-Za-z0-9_]+", query.lower())
    keep = [t for t in raw if t not in STOPWORDS and len(t) > 2]
    return keep or raw[:4]


def _primary_term(query: str) -> str:
    tokens = _search_tokens(query)
    for t in tokens:
        if t in SHIP_TERMS:
            return t
    return tokens[-1] if tokens else query.strip()


def _ngrams(text: str, n: int = 3) -> list[str]:
    s = re.sub(r"\s+", " ", text.lower()).strip()
    if len(s) < n:
        return [s] if s else []
    return [s[i : i + n] for i in range(len(s) - n + 1)]


def embed(text: str, dim: int = DIM) -> np.ndarray:
    """Signed hashing of character 3-grams plus word unigrams. No torch."""
    vec = np.zeros(dim, dtype=np.float32)
    for g in _ngrams(text):
        h = int(hashlib.blake2b(g.encode(), digest_size=8).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if (h >> 8) & 1 else -1.0
        vec[idx] += sign
    for tok in re.findall(r"[A-Za-z0-9]{3,}", text.lower()):
        h = int(hashlib.blake2b(f"w:{tok}".encode(), digest_size=8).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if (h >> 16) & 1 else -1.0
        vec[idx] += 2.0 * sign
    n = np.linalg.norm(vec)
    if n > 0:
        vec /= n
    return vec


def _clean_pdf_text(raw: str, layout: bool) -> str:
    text = raw.replace("\x00", "")
    text = re.sub(r"-\n(?=[A-Za-z])", "", text)
    if layout:
        text = re.sub(r"[ \t]{2,}", " | ", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf(path: Path) -> tuple[list[dict[str, Any]], bool]:
    """Return page records and whether a text layer was found."""
    reader = PdfReader(str(path))
    pages = []
    has_text = False
    for i, page in enumerate(reader.pages, start=1):
        raw = ""
        try:
            raw = page.extract_text(extraction_mode="layout") or ""
            layout = True
        except TypeError:
            raw = page.extract_text() or ""
            layout = False
        if len((raw or "").strip()) < 20:
            raw = page.extract_text() or ""
            layout = False
        text = _clean_pdf_text(raw or "", layout=layout)
        if len(text) > 20:
            has_text = True
        pages.append({"page": i, "text": text})
    return pages, has_text


def extract_text_file(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return [{"page": 1, "text": text}]


def chunk_pages(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pack whole lines so table rows and amount labels stay together."""
    chunks = []
    cid = 0
    for p in pages:
        text = p["text"] or ""
        if not text:
            continue
        lines = text.splitlines() or [text]
        buf: list[str] = []
        size = 0
        for line in lines:
            add = len(line) + 1
            if buf and size + add > CHUNK_CHARS:
                chunks.append({"chunk_id": cid, "page": p["page"], "text": "\n".join(buf)})
                cid += 1
                keep = []
                kept = 0
                for prev in reversed(buf):
                    if kept + len(prev) + 1 > OVERLAP:
                        break
                    keep.append(prev)
                    kept += len(prev) + 1
                buf = list(reversed(keep))
                size = sum(len(x) + 1 for x in buf)
            buf.append(line)
            size += add
        if buf:
            chunks.append({"chunk_id": cid, "page": p["page"], "text": "\n".join(buf)})
            cid += 1
    return chunks


class HybridIndex:
    def __init__(self, workspace: Path):
        self.workspace = workspace.resolve()
        self.dir = self.workspace / ".agentforge"
        self.dir.mkdir(exist_ok=True)
        self.db_path = self.dir / "fts.sqlite"
        self.vec_path = self.dir / "vectors.npz"
        self.lance_dir = self.dir / "lancedb"
        self.meta_path = self.dir / "chunks.jsonl"
        self.warn_path = self.dir / "ingest_warnings.json"
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS docs USING fts5(path, page, text, tokenize = 'porter unicode61')"
        )
        self.conn.commit()
        self._vectors: np.ndarray | None = None
        self._meta: list[dict[str, Any]] = []
        self._load()

    def ready(self) -> bool:
        return self.meta_path.exists() and self.meta_path.stat().st_size > 0

    def _load(self) -> None:
        if self.vec_path.exists():
            self._vectors = np.load(self.vec_path)["v"]
        if self.meta_path.exists():
            self._meta = [json.loads(line) for line in self.meta_path.read_text(encoding="utf-8").splitlines() if line]

    def ingest(self) -> dict[str, Any]:
        files: list[Path] = []
        for ext in ("*.pdf", "*.txt", "*.md"):
            files.extend(self.workspace.rglob(ext))
        files = [f for f in files if ".agentforge" not in f.parts and ".git" not in f.parts]
        skip_names = {"GROUND_TRUTH.md", "QUALITY.json", "LIVE_CHAT.json", "SOURCES.txt"}
        files = [f for f in files if f.name not in skip_names and f.suffix.lower() != ".json"]
        self.conn.execute("DROP TABLE IF EXISTS docs")
        try:
            self.conn.execute(
                "CREATE VIRTUAL TABLE docs USING fts5(path, page, text, tokenize = 'porter unicode61')"
            )
        except sqlite3.OperationalError:
            self.conn.execute(
                "CREATE VIRTUAL TABLE docs USING fts5(path, page, text)"
            )
        meta: list[dict[str, Any]] = []
        vecs: list[np.ndarray] = []
        no_text: list[str] = []
        scanned = 0
        for f in files:
            rel = str(f.relative_to(self.workspace)).replace("\\", "/")
            scanned += 1
            if f.suffix.lower() == ".pdf":
                pages, has_text = extract_pdf(f)
                if not has_text:
                    no_text.append(rel)
            else:
                pages = extract_text_file(f)
            chunks = chunk_pages(pages)
            for ch in chunks:
                self.conn.execute(
                    "INSERT INTO docs(path, page, text) VALUES(?,?,?)",
                    (rel, str(ch["page"]), ch["text"]),
                )
                rec = {"path": rel, "page": ch["page"], "text": ch["text"]}
                meta.append(rec)
                vecs.append(embed(ch["text"]))
        self.conn.commit()
        if vecs:
            arr = np.vstack(vecs)
        else:
            arr = np.zeros((0, DIM), dtype=np.float32)
        np.savez_compressed(self.vec_path, v=arr)
        self._write_lancedb(arr, meta)
        self.meta_path.write_text(
            "\n".join(json.dumps(m, ensure_ascii=False) for m in meta),
            encoding="utf-8",
        )
        warning = {
            "files_scanned": scanned,
            "chunks": len(meta),
            "no_text_layer": no_text,
            "no_text_layer_count": len(no_text),
        }
        self.warn_path.write_text(json.dumps(warning, indent=2), encoding="utf-8")
        self._vectors = arr
        self._meta = meta
        gt: dict[str, list[str]] = {}
        for m in meta:
            gt.setdefault(m["path"], []).append(m["text"])
        (self.dir / "grep.json").write_text(
            json.dumps({p: "\n".join(t) for p, t in gt.items()}, ensure_ascii=False),
            encoding="utf-8",
        )
        return warning

    def _write_lancedb(self, arr: np.ndarray, meta: list[dict[str, Any]]) -> None:
        """Plan store: LanceDB vectors. npz remains the fast local reader."""
        try:
            import lancedb
        except ImportError:
            return
        self.lance_dir.mkdir(exist_ok=True)
        db = lancedb.connect(str(self.lance_dir))
        rows = []
        for i, m in enumerate(meta):
            vec = arr[i].tolist() if arr.shape[0] else [0.0] * DIM
            rows.append(
                {
                    "vector": vec,
                    "path": m["path"],
                    "page": int(m["page"]),
                    "text": m["text"][:2000],
                }
            )
        if rows:
            db.create_table("chunks", data=rows, mode="overwrite")

    def grep_paths(self, term: str) -> list[str]:
        gt_path = self.dir / "grep.json"
        if not gt_path.exists():
            return []
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        needle = term.lower()
        return [p for p, text in gt.items() if needle in text.lower()]

    def search(self, query: str, limit: int = 12) -> dict[str, Any]:
        q = query.strip()
        term = _primary_term(q)
        lexical = self._fts(q, limit=max(limit, 40))
        lexical_paths = self._lexical_paths(term or q)
        semantic = self._vec(q, limit=limit)
        merged = _merge(lexical, semantic, limit=max(limit, 12))
        short_hits = []
        for h in merged[:12]:
            short_hits.append(
                {
                    "path": h["path"],
                    "page": h["page"],
                    "text": (h.get("text") or "")[:280],
                    "source": h.get("source"),
                }
            )
        return {
            "query": q,
            "term": term,
            "paths": lexical_paths,
            "hits": short_hits,
            "related_paths": sorted({h["path"] for h in semantic} - set(lexical_paths))[:8],
            "mode": "hybrid",
        }

    def _lexical_paths(self, term: str) -> list[str]:
        needle = (term or "").strip()
        if not needle:
            return []
        tokens = _search_tokens(needle)
        match = " OR ".join(f'"{t}"' for t in tokens[:6]) or f'"{needle}"'
        paths: list[str] = []
        try:
            rows = self.conn.execute(
                "SELECT DISTINCT path FROM docs WHERE docs MATCH ?",
                (match,),
            ).fetchall()
            paths = [r[0] for r in rows]
        except sqlite3.OperationalError:
            paths = []
        if not paths:
            like = f"%{tokens[0] if tokens else needle}%"
            rows = self.conn.execute(
                "SELECT DISTINCT path FROM docs WHERE text LIKE ? COLLATE NOCASE",
                (like,),
            ).fetchall()
            paths = [r[0] for r in rows]
        # Grep backup so hyphenation / layout artifacts cannot hide a file.
        grep = set(self.grep_paths(tokens[0] if tokens else needle))
        return sorted(set(paths) | grep)

    def _fts(self, query: str, limit: int) -> list[dict[str, Any]]:
        tokens = _search_tokens(query)
        if not tokens:
            return []
        match = " OR ".join(f'"{t}"' for t in tokens[:8])
        try:
            rows = self.conn.execute(
                "SELECT path, page, text FROM docs WHERE docs MATCH ? LIMIT ?",
                (match, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            like = f"%{tokens[0]}%"
            rows = self.conn.execute(
                "SELECT path, page, text FROM docs WHERE text LIKE ? LIMIT ?",
                (like, limit),
            ).fetchall()
        return [{"path": r[0], "page": int(r[1]), "text": r[2][:800], "source": "fts"} for r in rows]

    def _vec(self, query: str, limit: int) -> list[dict[str, Any]]:
        if self._vectors is None or len(self._meta) == 0 or self._vectors.shape[0] == 0:
            return []
        qv = embed(query)
        scores = self._vectors @ qv
        idx = np.argsort(-scores)[:limit]
        hits = []
        for i in idx:
            m = self._meta[int(i)]
            hits.append(
                {
                    "path": m["path"],
                    "page": m["page"],
                    "text": m["text"][:500],
                    "score": float(scores[int(i)]),
                    "source": "vec",
                }
            )
        return hits


def _merge(lexical: list[dict[str, Any]], semantic: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    seen: set[tuple[str, int]] = set()
    out: list[dict[str, Any]] = []
    # Lexical first so enumeration queries surface every matching file.
    for h in lexical + semantic:
        key = (h["path"], int(h["page"]))
        if key in seen:
            continue
        seen.add(key)
        out.append(h)
        if len(out) >= limit:
            break
    return out


def ingest_workspace(workspace: Path) -> dict[str, Any]:
    idx = HybridIndex(workspace)
    return idx.ingest()
