# 🏰 MARS for CASTLE Challenge @ EgoVis 2026

**MARS** is the code scaffold for our **runner-up solution** in the CASTLE Challenge @ EgoVis 2026.

MARS stands for **M**ultimodal **A**gentic **R**easoning with **S**ource
Selection: instead of
forcing every modality into one prompt, the system maintains a compact evidence
state and lets a decision agent choose which source should be read next.

## 🥈 Result

| Rank | Participant | Accuracy |
| ---: | --- | ---: |
| 1 | WDL | 0.58 |
| **2** | **ilearn_zhy (ours)** | **0.57** |
| 3 | raghad_khaled | 0.55 |

This repository stores the reproducible code structure for the second-place
MARS pipeline. Some expensive components, especially long-video captioning and
DeepSeek-based summarization, are represented as offline inputs because the
CASTLE videos are too long to place directly into the model context for every
question.

## 🧭 Method Overview

```text
Question + options
        |
        v
Initial evidence state
  - transcript snippets
  - offline video summaries
        |
        v
GPT-5.4 decision agent
        |
        +--> continue_thinking
        +--> add_data: transcript / video_summary / photo / thermal / gaze / heartrate / auxiliary_video
        +--> answer: a / b / c / d
        +--> random_fallback: unsupported case under fixed budget
```

The central idea is **source selection**. Videos are compressed into textual
captions, OCR notes, and summaries only for context-budget reasons. Photos,
gaze, heartrate, thermal images, transcripts, and auxiliary videos remain
source-specific evidence that the agent can request when useful.

## 📁 Repository Layout

```text
.
├── README.md
├── requirements.txt
├── build_evidence_memory.py
├── run_mars_agent.py
└── mars/
    ├── __init__.py
    ├── schema.py
    ├── castle_index.py
    └── evidence_memory.py
```

| File | Purpose |
| --- | --- |
| `build_evidence_memory.py` | Builds JSONL evidence memory from the CASTLE directory. |
| `run_mars_agent.py` | Runs the MARS agent loop and writes submission JSON. |
| `mars/schema.py` | Defines evidence, question, and agent decision objects. |
| `mars/castle_index.py` | Parses CASTLE paths, people, days, rooms, and source cues. |
| `mars/evidence_memory.py` | Loads, retrieves, scores, and formats evidence. |

## ⚙️ Setup

Install optional runtime dependency for OpenAI-compatible API calls:

```bash
pip install -r requirements.txt
```

The dry-run path has no external API dependency and is useful for validating the
pipeline locally.

## 🧱 Step 1: Build Evidence Memory

From the CASTLE dataset root:

```bash
python3 code/build_evidence_memory.py \
  --castle-root . \
  --output code/outputs/evidence_memory.jsonl
```

From this `code/` repository root:

```bash
python3 build_evidence_memory.py \
  --castle-root .. \
  --output outputs/evidence_memory.jsonl
```

If offline video captions, OCR notes, or DeepSeek summaries are available, pass
their folder:

```bash
python3 build_evidence_memory.py \
  --castle-root .. \
  --video-summary-dir ../.multimodal_cache/video_summaries \
  --output outputs/evidence_memory.jsonl
```

Supported video-summary input formats:

- `.jsonl`
- `.json`
- `.txt`

Expected fields for JSON/JSONL are flexible. The loader checks common keys such
as `summary`, `caption`, `content`, `text`, `day`, `owner`, `stream`, and
`timestamp`.

## 🤖 Step 2: Run MARS Agent

### Dry Run

Dry run uses deterministic lexical retrieval and fallback behavior. It validates
the data flow without calling a model API:

```bash
python3 run_mars_agent.py \
  --questions ../EgoVis2026_CVPR_Questions.json \
  --memory outputs/evidence_memory.jsonl \
  --output ../submissions/mars_dry_run.json \
  --log ../submissions/mars_dry_run_log.jsonl \
  --dry-run
```

### API Mode

Use any OpenAI-compatible endpoint:

```bash
OPENAI_API_KEY=... python3 run_mars_agent.py \
  --questions ../EgoVis2026_CVPR_Questions.json \
  --memory outputs/evidence_memory.jsonl \
  --output ../submissions/mars_submission.json \
  --log ../submissions/mars_agent_log.jsonl \
  --model gpt-5.4
```

For a custom endpoint:

```bash
OPENAI_API_KEY=... python3 run_mars_agent.py \
  --base-url https://your.openai-compatible.endpoint/v1 \
  --model gpt-5.4 \
  --questions ../EgoVis2026_CVPR_Questions.json \
  --memory outputs/evidence_memory.jsonl \
  --output ../submissions/mars_submission.json
```

## 🧩 Agent Actions

The decision model must return one JSON action:

```json
{"action": "continue_thinking", "rationale": "Need to compare top options."}
```

```json
{"action": "add_data", "source_type": "photo", "rationale": "Need object-level visual evidence."}
```

```json
{"action": "answer", "answer": "b", "confidence": 0.72, "rationale": "Evidence supports option b."}
```

```json
{"action": "random_fallback", "answer": "a", "rationale": "Evidence remains insufficient."}
```

Valid `source_type` values:

- `transcript`
- `video_summary`
- `photo`
- `thermal`
- `gaze`
- `heartrate`
- `auxiliary_video`

## 📝 Output Files

| Output | Description |
| --- | --- |
| `outputs/evidence_memory.jsonl` | Evidence entries built from CASTLE data. |
| `../submissions/mars_submission.json` | Final answer file in challenge format. |
| `../submissions/mars_agent_log.jsonl` | Per-question agent actions and source usage. |

## ✅ Local Validation

The following checks were used when this scaffold was created:

```bash
PYTHONPYCACHEPREFIX=code/.pycache \
python3 -m py_compile code/build_evidence_memory.py code/run_mars_agent.py code/mars/*.py
```

```bash
python3 code/build_evidence_memory.py \
  --castle-root . \
  --output /private/tmp/castle_mars_memory.jsonl \
  --max-transcript-chars 2000
```

```bash
python3 code/run_mars_agent.py \
  --questions EgoVis2026_CVPR_Questions.json \
  --memory /private/tmp/castle_mars_memory.jsonl \
  --output /private/tmp/castle_mars_dry_run.json \
  --log /private/tmp/castle_mars_dry_run.jsonl \
  --dry-run \
  --limit 2
```

