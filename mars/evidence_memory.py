from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from .castle_index import (
    infer_question_cues,
    iter_auxiliary_files,
    iter_transcript_files,
    make_auxiliary_evidence,
    make_transcript_evidence,
)
from .schema import Evidence, Question


def write_jsonl(path: Path, rows: Iterable[Evidence]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row.to_json(), ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> List[Evidence]:
    rows: List[Evidence] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(Evidence.from_json(json.loads(line)))
    return rows


def build_memory(
    castle_root: Path,
    video_summary_dir: Optional[Path] = None,
    max_transcript_chars: int = 1200,
) -> List[Evidence]:
    memory: List[Evidence] = []

    for path in iter_transcript_files(castle_root):
        memory.extend(make_transcript_evidence(path, castle_root, max_transcript_chars))

    for path in iter_auxiliary_files(castle_root):
        memory.append(make_auxiliary_evidence(path, castle_root))

    if video_summary_dir is not None and video_summary_dir.exists():
        memory.extend(load_video_summaries(video_summary_dir, castle_root))

    return memory


def load_video_summaries(summary_dir: Path, castle_root: Path) -> List[Evidence]:
    rows: List[Evidence] = []
    for path in sorted(summary_dir.rglob("*")):
        if not path.is_file() or path.name == ".DS_Store":
            continue
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            rows.extend(load_video_summary_jsonl(path, castle_root))
        elif suffix == ".json":
            rows.extend(load_video_summary_json(path, castle_root))
        elif suffix == ".txt":
            rows.append(video_summary_from_text(path, castle_root))
    return rows


def load_video_summary_jsonl(path: Path, castle_root: Path) -> List[Evidence]:
    rows: List[Evidence] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        rows.append(video_summary_from_mapping(data, path, castle_root))
    return rows


def load_video_summary_json(path: Path, castle_root: Path) -> List[Evidence]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [video_summary_from_mapping(item, path, castle_root) for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        return [video_summary_from_mapping(data, path, castle_root)]
    return []


def video_summary_from_mapping(data: Dict[str, object], path: Path, castle_root: Path) -> Evidence:
    content = (
        data.get("summary")
        or data.get("caption")
        or data.get("content")
        or data.get("text")
        or ""
    )
    rel = safe_relative(path, castle_root)
    return Evidence(
        source_type="video_summary",
        day=optional_str(data.get("day")),
        owner=optional_str(data.get("owner") or data.get("stream") or data.get("person")),
        time=optional_str(data.get("time") or data.get("timestamp")),
        path=rel,
        content=str(content),
        metadata={k: v for k, v in data.items() if k not in {"summary", "caption", "content", "text"}},
    )


def video_summary_from_text(path: Path, castle_root: Path) -> Evidence:
    return Evidence(
        source_type="video_summary",
        path=safe_relative(path, castle_root),
        content=path.read_text(encoding="utf-8", errors="ignore").strip(),
    )


def retrieve_evidence(
    memory: Sequence[Evidence],
    question: Question,
    requested_source: Optional[str] = None,
    requested_day: Optional[str] = None,
    requested_owner: Optional[str] = None,
    limit: int = 8,
) -> List[Evidence]:
    cues = infer_question_cues(question.query + " " + " ".join(question.answers.values()))
    query_terms = tokenize(question.query + " " + " ".join(question.answers.values()))
    scored: List[Evidence] = []

    for item in memory:
        if requested_source and item.source_type != requested_source:
            continue
        if requested_day and item.day and item.day != requested_day:
            continue
        if requested_owner and item.owner and item.owner.lower() != requested_owner.lower():
            continue

        score = lexical_score(query_terms, item.content)
        if item.day in cues["days"]:
            score += 3.0
        if item.owner in cues["people"] or item.owner in cues["rooms"]:
            score += 3.0
        if item.source_type in cues["sources"]:
            score += 2.0
        if requested_source:
            score += 1.0

        if score > 0:
            scored.append(
                Evidence(
                    source_type=item.source_type,
                    content=item.content,
                    day=item.day,
                    owner=item.owner,
                    time=item.time,
                    path=item.path,
                    score=score,
                    metadata=item.metadata,
                )
            )

    scored.sort(key=lambda item: item.score, reverse=True)
    return scored[:limit]


def format_evidence(rows: Sequence[Evidence], max_chars: int = 6500) -> str:
    lines: List[str] = []
    total = 0
    for index, row in enumerate(rows, start=1):
        prefix = f"[{index}] source={row.source_type}"
        if row.day:
            prefix += f", day={row.day}"
        if row.owner:
            prefix += f", owner={row.owner}"
        if row.time:
            prefix += f", time={row.time}"
        if row.path:
            prefix += f", path={row.path}"
        text = compact(row.content)
        line = f"{prefix}\n{text}"
        if total + len(line) > max_chars:
            break
        lines.append(line)
        total += len(line)
    return "\n\n".join(lines) if lines else "No evidence retrieved."


def tokenize(text: str) -> List[str]:
    return [token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) >= 3]


def lexical_score(query_terms: Sequence[str], content: str) -> float:
    content_terms = set(tokenize(content))
    return float(sum(1 for token in query_terms if token in content_terms))


def compact(text: str, max_len: int = 900) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def safe_relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def optional_str(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None

