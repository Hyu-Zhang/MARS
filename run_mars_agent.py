#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from mars.evidence_memory import format_evidence, load_jsonl, retrieve_evidence
from mars.schema import (
    ACTION_ADD_DATA,
    ACTION_ANSWER,
    ACTION_CONTINUE,
    ACTION_RANDOM,
    CHOICES,
    AgentDecision,
    Evidence,
    Question,
)


DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-5.4"
SOURCE_ORDER = ["transcript", "video_summary", "photo", "thermal", "gaze", "heartrate", "auxiliary_video"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MARS source-selection agent.")
    parser.add_argument("--questions", type=Path, default=Path("EgoVis2026_CVPR_Questions.json"))
    parser.add_argument("--memory", type=Path, default=Path("code/outputs/evidence_memory.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("submissions/mars_submission.json"))
    parser.add_argument("--log", type=Path, default=Path("submissions/mars_agent_log.jsonl"))
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--evidence-limit", type=int, default=8)
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_questions(path: Path) -> List[Question]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    questions = [Question.from_json(row) for row in rows]
    return questions


def load_existing_answers(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    return {
        str(key): str(value).lower()
        for key, value in json.loads(path.read_text(encoding="utf-8")).items()
        if str(value).lower() in CHOICES
    }


def write_submission(path: Path, questions: Sequence[Question], answers: Dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = {
        question.question_id: answers[question.question_id]
        for question in questions
        if answers.get(question.question_id) in CHOICES
    }
    path.write_text(json.dumps(ordered, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_log(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_agent_prompt(question: Question, evidence: Sequence[Evidence], step: int, max_steps: int) -> str:
    options = "\n".join(f"{key}. {value}" for key, value in sorted(question.answers.items()))
    evidence_text = format_evidence(evidence)
    return f"""You are the MARS decision agent for CASTLE multiple-choice QA.

You must choose exactly one JSON action:
1. {ACTION_CONTINUE}: reorganize current evidence and think further.
2. {ACTION_ADD_DATA}: request one missing source.
3. {ACTION_ANSWER}: answer with one of a, b, c, d.
4. {ACTION_RANDOM}: use fallback if the evidence remains insufficient.

Valid source_type values for add_data:
transcript, video_summary, photo, thermal, gaze, heartrate, auxiliary_video.

Question ID: {question.question_id}
Question: {question.query}
Options:
{options}

Step: {step}/{max_steps}
Current evidence:
{evidence_text}

Return JSON only, for example:
{{"action":"add_data","source_type":"photo","rationale":"Need visual object evidence."}}
or
{{"action":"answer","answer":"b","confidence":0.72,"rationale":"Evidence supports option b."}}
"""


def request_agent_decision(
    client: object,
    model: str,
    question: Question,
    evidence: Sequence[Evidence],
    step: int,
    max_steps: int,
) -> AgentDecision:
    prompt = build_agent_prompt(question, evidence, step, max_steps)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Return valid JSON only."},
            {"role": "user", "content": prompt},
        ],
    )
    text = response.choices[0].message.content or "{}"
    return AgentDecision.from_json(extract_json_object(text))


def extract_json_object(text: str) -> Dict[str, object]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return {"action": ACTION_CONTINUE, "rationale": text}


def dry_run_decision(
    question: Question,
    evidence: Sequence[Evidence],
    used_sources: Sequence[str],
    step: int,
    max_steps: int,
    rng: random.Random,
) -> AgentDecision:
    choice = option_overlap_choice(question, evidence)
    if choice and step >= 2:
        return AgentDecision(
            action=ACTION_ANSWER,
            answer=choice,
            confidence=0.45,
            rationale="Dry-run lexical overlap selected an option.",
        )
    for source_type in SOURCE_ORDER:
        if source_type not in used_sources:
            return AgentDecision(
                action=ACTION_ADD_DATA,
                source_type=source_type,
                rationale=f"Dry-run requests {source_type}.",
            )
    if step >= max_steps:
        return AgentDecision(
            action=ACTION_RANDOM,
            answer=rng.choice(sorted(CHOICES)),
            rationale="Dry-run reached the reasoning budget.",
        )
    return AgentDecision(action=ACTION_CONTINUE, rationale="Dry-run continues reasoning.")


def option_overlap_choice(question: Question, evidence: Sequence[Evidence]) -> Optional[str]:
    text = " ".join(item.content for item in evidence).lower()
    scores: Dict[str, int] = {}
    for key, option in question.answers.items():
        tokens = [token for token in re.findall(r"[a-z0-9]+", option.lower()) if len(token) >= 3]
        scores[key] = sum(1 for token in tokens if token in text)
    best_key, best_score = max(scores.items(), key=lambda item: item[1])
    return best_key if best_score > 0 else None


def run_question(
    question: Question,
    memory: Sequence[Evidence],
    client: Optional[object],
    args: argparse.Namespace,
    rng: random.Random,
) -> Dict[str, object]:
    evidence: List[Evidence] = []
    used_sources: List[str] = []

    for source_type in ["transcript", "video_summary"]:
        rows = retrieve_evidence(memory, question, requested_source=source_type, limit=args.evidence_limit)
        if rows:
            evidence.extend(rows)
            used_sources.append(source_type)

    decisions: List[Dict[str, object]] = []
    final_answer: Optional[str] = None

    for step in range(1, args.max_steps + 1):
        if args.dry_run:
            decision = dry_run_decision(question, evidence, used_sources, step, args.max_steps, rng)
        else:
            assert client is not None
            decision = request_agent_decision(client, args.model, question, evidence, step, args.max_steps)

        decisions.append(decision.__dict__)

        if decision.action == ACTION_ANSWER and decision.answer in CHOICES:
            final_answer = decision.answer
            break
        if decision.action == ACTION_RANDOM:
            final_answer = decision.answer if decision.answer in CHOICES else rng.choice(sorted(CHOICES))
            break
        if decision.action == ACTION_ADD_DATA and decision.source_type:
            if decision.source_type not in used_sources:
                rows = retrieve_evidence(
                    memory,
                    question,
                    requested_source=decision.source_type,
                    requested_day=decision.day,
                    requested_owner=decision.owner,
                    limit=args.evidence_limit,
                )
                evidence.extend(rows)
                used_sources.append(decision.source_type)
            continue
        if decision.action == ACTION_CONTINUE:
            continue

    if final_answer is None:
        final_answer = option_overlap_choice(question, evidence) or rng.choice(sorted(CHOICES))
        decisions.append(
            {
                "action": ACTION_RANDOM,
                "answer": final_answer,
                "rationale": "Budget exhausted; fallback answer emitted.",
            }
        )

    return {
        "id": question.question_id,
        "answer": final_answer,
        "used_sources": used_sources,
        "decisions": decisions,
        "evidence_count": len(evidence),
    }


def make_client(args: argparse.Namespace) -> Optional[object]:
    if args.dry_run:
        return None
    api_key = args.api_key or os.getenv("OPENAI_API_KEY") or os.getenv("AUTODL_API_KEY")
    if not api_key:
        raise RuntimeError("Missing API key. Pass --api-key or set OPENAI_API_KEY/AUTODL_API_KEY.")
    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url=args.base_url)


def main() -> int:
    args = parse_args()
    rng = random.Random(args.random_seed)
    memory = load_jsonl(args.memory)
    questions = load_questions(args.questions)
    if args.limit is not None:
        questions = questions[: args.limit]
    answers = load_existing_answers(args.output)
    client = make_client(args)

    for index, question in enumerate(questions, start=1):
        if answers.get(question.question_id) in CHOICES:
            print(f"[{index}/{len(questions)}] skip {question.question_id}: {answers[question.question_id]}")
            continue
        result = run_question(question, memory, client, args, rng)
        answers[question.question_id] = str(result["answer"])
        write_submission(args.output, questions, answers)
        append_log(args.log, result)
        print(f"[{index}/{len(questions)}] {question.question_id}: {result['answer']} sources={result['used_sources']}")
        if args.sleep_seconds > 0 and not args.dry_run:
            time.sleep(args.sleep_seconds)

    write_submission(args.output, questions, answers)
    print(f"Wrote {len(answers)} answers to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

