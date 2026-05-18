from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from .schema import Evidence


PERSONS = [
    "Allie",
    "Bao",
    "Bjorn",
    "Cathal",
    "Florian",
    "Klaus",
    "Luca",
    "Onanong",
    "Stevan",
    "Tien",
    "Werner",
]

ROOMS = ["Kitchen", "Living1", "Living2", "Meeting", "Reading"]

DAY_TO_DATE = {
    "day1": "2024-12-03",
    "day2": "2024-12-04",
    "day3": "2024-12-05",
    "day4": "2024-12-06",
}

DAY_PHRASES = {
    "day1": ["day 1", "day one", "first day", "2024-12-03", "20241203"],
    "day2": ["day 2", "day two", "second day", "2024-12-04", "20241204"],
    "day3": ["day 3", "day three", "third day", "2024-12-05", "20241205"],
    "day4": ["day 4", "day four", "fourth day", "last day", "2024-12-06", "20241206"],
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".ts"}


def infer_question_cues(text: str) -> Dict[str, List[str]]:
    lower = text.lower()
    cues: Dict[str, List[str]] = {"days": [], "people": [], "rooms": [], "sources": []}

    for day, phrases in DAY_PHRASES.items():
        if any(phrase in lower for phrase in phrases):
            cues["days"].append(day)

    for name in PERSONS:
        if re.search(rf"(?<![a-z]){re.escape(name.lower())}(?![a-z])", lower):
            cues["people"].append(name)

    for room in ROOMS:
        if room.lower().replace("1", "").replace("2", "") in lower:
            cues["rooms"].append(room)

    source_terms = {
        "photo": ["photo", "picture", "image", "screen", "vehicle", "car"],
        "thermal": ["thermal", "temperature", "hot", "cold", "heat", "cooking"],
        "gaze": ["look", "looking", "gaze", "attend", "attention", "watch"],
        "heartrate": ["heart", "heartrate", "heart rate", "exercise", "physical"],
        "transcript": ["say", "said", "speak", "conversation", "tell", "mentioned"],
        "video_summary": ["see", "wear", "color", "count", "object", "where"],
    }
    for source_type, terms in source_terms.items():
        if any(term in lower for term in terms):
            cues["sources"].append(source_type)

    return cues


def iter_transcript_files(castle_root: Path) -> Iterator[Path]:
    main_dir = castle_root / "main"
    if not main_dir.exists():
        return
    yield from sorted(main_dir.glob("day*/*/transcript/*.json"))


def iter_auxiliary_files(castle_root: Path) -> Iterator[Path]:
    aux_dir = castle_root / "auxiliary"
    if not aux_dir.exists():
        return
    for path in sorted(aux_dir.rglob("*")):
        if path.is_file() and path.name != ".DS_Store":
            yield path


def transcript_path_metadata(path: Path, castle_root: Path) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    try:
        rel = path.relative_to(castle_root)
    except ValueError:
        return None, None, None
    parts = rel.parts
    if len(parts) >= 5 and parts[0] == "main":
        return parts[1], parts[2], path.stem
    return None, None, None


def aux_path_metadata(path: Path, castle_root: Path) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
    try:
        rel = path.relative_to(castle_root)
    except ValueError:
        rel = path
    parts = rel.parts
    source_type = "auxiliary"
    owner = None
    day = infer_day_from_name(path.name)
    if len(parts) >= 2 and parts[0] == "auxiliary":
        source_type = parts[1]
        if source_type in {"photo", "video", "heartrate"} and len(parts) >= 3:
            owner = parts[2]
        if source_type == "gaze":
            owner = path.stem
        if source_type == "heartrate":
            day = path.stem if path.stem.startswith("day") else day
    time = infer_time_from_name(path.name)
    return source_type, day, owner, time


def infer_day_from_name(name: str) -> Optional[str]:
    compact = re.sub(r"[^0-9]", "", name)
    for day, date in DAY_TO_DATE.items():
        if date.replace("-", "") in compact:
            return day
    return None


def infer_time_from_name(name: str) -> Optional[str]:
    match = re.search(r"(20\d{6})[_-]?(\d{2})(\d{2})(\d{2})", name)
    if match:
        return f"{match.group(2)}:{match.group(3)}:{match.group(4)}"
    hour_match = re.match(r"^(\d{2})$", Path(name).stem)
    if hour_match:
        return f"{hour_match.group(1)}:00"
    return None


def transcript_chunks(path: Path) -> List[Dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    chunks = data.get("chunks", [])
    if isinstance(chunks, list):
        return [chunk for chunk in chunks if isinstance(chunk, dict)]
    return []


def chunk_time(chunk: Dict[str, object], file_time: Optional[str]) -> Optional[str]:
    timestamp = chunk.get("timestamp")
    if isinstance(timestamp, Sequence) and len(timestamp) >= 2:
        start = timestamp[0]
        end = timestamp[1]
        if isinstance(start, (int, float)) and isinstance(end, (int, float)):
            return f"{file_time or ''}+{start:.1f}-{end:.1f}s".strip("+")
    return file_time


def make_transcript_evidence(path: Path, castle_root: Path, max_chars: int) -> List[Evidence]:
    day, owner, file_time = transcript_path_metadata(path, castle_root)
    rows: List[Evidence] = []
    buffer: List[str] = []
    buffer_time = file_time

    for chunk in transcript_chunks(path):
        text = str(chunk.get("text", "")).strip()
        if not text:
            continue
        if not buffer:
            buffer_time = chunk_time(chunk, file_time)
        if sum(len(item) for item in buffer) + len(text) > max_chars and buffer:
            rows.append(
                Evidence(
                    source_type="transcript",
                    day=day,
                    owner=owner,
                    time=buffer_time,
                    path=str(path.relative_to(castle_root)),
                    content=" ".join(buffer),
                )
            )
            buffer = []
            buffer_time = chunk_time(chunk, file_time)
        buffer.append(text)

    if buffer:
        rows.append(
            Evidence(
                source_type="transcript",
                day=day,
                owner=owner,
                time=buffer_time,
                path=str(path.relative_to(castle_root)),
                content=" ".join(buffer),
            )
        )
    return rows


def make_auxiliary_evidence(path: Path, castle_root: Path) -> Evidence:
    source_type, day, owner, time = aux_path_metadata(path, castle_root)
    rel = path.relative_to(castle_root).as_posix()
    suffix = path.suffix.lower()

    if source_type == "video":
        source_type = "auxiliary_video"
    elif source_type == "photo" or suffix in IMAGE_EXTS:
        source_type = "photo" if source_type != "thermal" else "thermal"

    content = f"{source_type} evidence available at {rel}."
    if suffix == ".csv":
        content = summarize_csv_file(path, source_type, rel)

    return Evidence(
        source_type=source_type,
        day=day,
        owner=owner,
        time=time,
        path=rel,
        content=content,
    )


def summarize_csv_file(path: Path, source_type: str, rel: str) -> str:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    header = lines[0] if lines else ""
    sample_count = max(0, len(lines) - 1)
    return f"{source_type} table at {rel}; columns: {header}; rows: {sample_count}."

