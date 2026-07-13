from pathlib import Path
import py_compile

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

from openai import DefaultHttpxClient, OpenAI


# ============================================================
# 1. Configuration
# ============================================================

# The model specified by the interview task.
MODEL = "gpt-4.1-mini-2025-04-14"

# The directory containing 1_en.txt ... 8_en.txt.
INPUT_DIR = Path(".")

# Intermediate outputs and caches.
WORK_DIR = Path("task1_work")
EXTRACT_DIR = WORK_DIR / "01_extracted"
MERGE_DIR = WORK_DIR / "02_merged"
DRAFT_DIR = WORK_DIR / "03_drafts"

# Required final outputs.
OUTPUT_JSON = Path("task1.json")
OUTPUT_MD = Path("task1.md")
OUTPUT_REPORT = Path("task1_report.json")
OUTPUT_EVIDENCE = Path("task1_evidence.json")

# Local proxy. Change to None when no proxy is needed.
PROXY_URL: str | None = "http://127.0.0.1:1080"

# Set True to ignore all caches and regenerate everything.
FORCE_REGENERATE = False

# API retry settings.
MAX_API_ATTEMPTS = 4
MAX_REPAIR_ROUNDS = 4

# Prompt/cache version. Increase it after changing prompts or schemas.
PIPELINE_VERSION = "task1-merge-v1"

SECTION_NAMES = [
    "夏校项目简介",
    "夏校项目学术课程",
    "夏校项目非学术活动",
    "夏校项目食宿安排",
    "夏校项目费用、报名及其他信息",
]

# Character rules:
# - For minimum requirements, the script uses NON-WHITESPACE characters.
#   This is conservative: it still passes if the evaluator ignores spaces.
# - For the introduction maximum, the script uses ALL characters, including
#   spaces and line breaks. This is also conservative.
#
# The original requirements use strict inequalities:
# introduction > 200 and < 500
# academics > 1500
# non-academic activities > 1600
# accommodation and food > 1400
# fees/application/other information > 1300
SECTION_SPECS: dict[str, dict[str, Any]] = {
    "夏校项目简介": {
        "minimum_non_whitespace": 200,
        "maximum_total": 500,
        "target": (
            "Write approximately 380-450 total characters, including spaces. "
            "The final text must contain more than 200 non-whitespace "
            "characters and fewer than 500 total characters."
        ),
        "focus": (
            "Give a concise overview of the summer school, its setting, "
            "target participants, central educational proposition, and "
            "most distinctive documented features. Do not duplicate the "
            "detailed course, activity, accommodation, or fee sections."
        ),
    },
    "夏校项目学术课程": {
        "minimum_non_whitespace": 1500,
        "maximum_total": None,
        "target": (
            "Write approximately 2,100-2,500 total characters, including "
            "spaces, and ensure more than 1,500 non-whitespace characters."
        ),
        "focus": (
            "Explain the documented academic curriculum, English-language "
            "teaching, class organization, placement, teaching hours, "
            "materials, projects, academies, instructional format, and "
            "learning experience."
        ),
    },
    "夏校项目非学术活动": {
        "minimum_non_whitespace": 1600,
        "maximum_total": None,
        "target": (
            "Write approximately 2,300-2,700 total characters, including "
            "spaces, and ensure more than 1,600 non-whitespace characters."
        ),
        "focus": (
            "Explain the documented sports, academies, multi-activity "
            "options, excursions, cultural experiences, social interaction, "
            "team challenges, evening or weekend activities, and their "
            "practical appeal."
        ),
    },
    "夏校项目食宿安排": {
        "minimum_non_whitespace": 1400,
        "maximum_total": None,
        "target": (
            "Write approximately 2,000-2,400 total characters, including "
            "spaces, and ensure more than 1,400 non-whitespace characters."
        ),
        "focus": (
            "Explain the documented boarding environment, room arrangements, "
            "supervision, facilities, safety and pastoral support, meals, "
            "catering, dining arrangements, and everyday residential "
            "experience."
        ),
    },
    "夏校项目费用、报名及其他信息": {
        "minimum_non_whitespace": 1300,
        "maximum_total": None,
        "target": (
            "Write approximately 1,900-2,300 total characters, including "
            "spaces, and ensure more than 1,300 non-whitespace characters."
        ),
        "focus": (
            "Explain the documented price, minimum stay, optional charges, "
            "dates, age rules, booking or application information, transfers, "
            "conditions, inclusions, exclusions, and other practical points."
        ),
    },
}


# ============================================================
# 2. Logging and client
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def build_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set.\n"
            'PowerShell example:\n'
            '$env:OPENAI_API_KEY="your-api-key"'
        )

    common_args: dict[str, Any] = {
        "api_key": api_key,
        "timeout": 240.0,
        "max_retries": 1,
    }

    if PROXY_URL:
        logger.info("Using proxy: %s", PROXY_URL)
        common_args["http_client"] = DefaultHttpxClient(proxy=PROXY_URL)
    else:
        logger.info("Connecting without a local proxy.")

    return OpenAI(**common_args)


# ============================================================
# 3. Schemas for Structured Outputs
# ============================================================

CLASSIFICATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        section: {
            "type": "array",
            "items": {"type": "string"},
        }
        for section in SECTION_NAMES
    },
    "required": SECTION_NAMES,
    "additionalProperties": False,
}

FACT_ITEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "statement": {"type": "string"},
        "source_ids": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        },
    },
    "required": ["statement", "source_ids"],
    "additionalProperties": False,
}

PACKET_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        section: {
            "type": "array",
            "items": FACT_ITEM_SCHEMA,
        }
        for section in SECTION_NAMES
    },
    "required": SECTION_NAMES,
    "additionalProperties": False,
}

CONTENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "content": {"type": "string"},
    },
    "required": ["content"],
    "additionalProperties": False,
}

AUDIT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "passed": {"type": "boolean"},
        "issues": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["passed", "issues"],
    "additionalProperties": False,
}


# ============================================================
# 4. Basic utilities
# ============================================================

def ensure_directories() -> None:
    for directory in [WORK_DIR, EXTRACT_DIR, MERGE_DIR, DRAFT_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def read_text(path: Path) -> str:
    encodings = ["utf-8", "utf-8-sig", "gb18030"]
    last_error: Exception | None = None

    for encoding in encodings:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError as error:
            last_error = error

    raise RuntimeError(f"Unable to decode {path}") from last_error


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_json(data: Any) -> str:
    return json.dumps(
        data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def normalize_statement(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text


def dedupe_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def extract_numbers(text: str) -> set[str]:
    values = re.findall(r"\d+(?:[.,]\d+)*", text)
    return {value.replace(",", "").rstrip(".") for value in values}


def count_characters(text: str) -> dict[str, int]:
    return {
        "total": len(text),
        "non_whitespace": len(re.sub(r"\s+", "", text)),
    }


def evidence_text_for_section(
    packet: dict[str, list[dict[str, Any]]],
    section: str,
) -> str:
    return "\n".join(
        item["statement"]
        for item in packet.get(section, [])
        if item.get("statement")
    )


def normalize_packet(
    packet: dict[str, Any],
    allowed_source_ids: set[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    normalized: dict[str, list[dict[str, Any]]] = {
        section: [] for section in SECTION_NAMES
    }

    for section in SECTION_NAMES:
        merged_by_key: dict[str, dict[str, Any]] = {}

        for raw_item in packet.get(section, []):
            statement = normalize_statement(str(raw_item.get("statement", "")))
            if not statement:
                continue

            raw_sources = raw_item.get("source_ids", [])
            source_ids = sorted(
                {
                    str(source).strip()
                    for source in raw_sources
                    if str(source).strip()
                }
            )

            if allowed_source_ids is not None:
                source_ids = [
                    source
                    for source in source_ids
                    if source in allowed_source_ids
                ]

            if not source_ids:
                # Source IDs are only for internal traceability. An empty list
                # indicates an invalid merge result, so omit the item.
                continue

            key = dedupe_key(statement)
            if not key:
                continue

            if key in merged_by_key:
                merged_by_key[key]["source_ids"] = sorted(
                    set(merged_by_key[key]["source_ids"]) | set(source_ids)
                )
            else:
                merged_by_key[key] = {
                    "statement": statement,
                    "source_ids": source_ids,
                }

        normalized[section] = list(merged_by_key.values())

    return normalized


def packet_source_ids(packet: dict[str, Any]) -> set[str]:
    result: set[str] = set()
    for section in SECTION_NAMES:
        for item in packet.get(section, []):
            result.update(item.get("source_ids", []))
    return result


# ============================================================
# 5. OpenAI request helper
# ============================================================

def call_json(
    client: OpenAI,
    task_name: str,
    developer_prompt: str,
    user_prompt: str,
    schema_name: str,
    schema: dict[str, Any],
    temperature: float = 0.0,
) -> dict[str, Any]:
    last_error: Exception | None = None

    for attempt in range(1, MAX_API_ATTEMPTS + 1):
        try:
            logger.info(
                "%s: API attempt %d/%d",
                task_name,
                attempt,
                MAX_API_ATTEMPTS,
            )

            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "developer",
                        "content": developer_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                temperature=temperature,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "strict": True,
                        "schema": schema,
                    },
                },
            )

            message = response.choices[0].message

            refusal = getattr(message, "refusal", None)
            if refusal:
                raise RuntimeError(f"Model refusal: {refusal}")

            content = message.content
            if not content:
                raise RuntimeError("The model returned empty content.")

            data = json.loads(content)

            if response.usage:
                logger.info(
                    "%s tokens: input=%s output=%s total=%s",
                    task_name,
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                    response.usage.total_tokens,
                )

            return data

        except Exception as error:
            last_error = error
            logger.warning("%s failed: %s", task_name, error)

            if attempt < MAX_API_ATTEMPTS:
                time.sleep(attempt * 5)

    raise RuntimeError(
        f"{task_name} failed after {MAX_API_ATTEMPTS} attempts: {last_error}"
    )


# ============================================================
# 6. Stage 1: classify each *_en.txt source
# ============================================================

EXTRACTION_DEVELOPER_PROMPT = """
You are a meticulous source analyst for an international summer school
writing task.

The input is an English summary of one source article. Extract its concrete,
source-supported facts and place each fact in exactly one of the five supplied
Chinese-keyed categories.

Rules:
1. Return facts in English.
2. Use only information explicitly present in the supplied source.
3. Do not infer, embellish, advertise, or use external knowledge.
4. Preserve every important number, date, age, price, duration, class size,
   teaching hour, location, named course, activity, room type, meal, condition,
   limitation, and optional charge.
5. Preserve qualifiers such as "up to", "maximum", "minimum", "optional",
   "may", "generally", and "subject to availability".
6. Remove purely repetitive wording and unsupported promotional adjectives.
7. Put each fact in the single best category. Do not duplicate a fact across
   categories.
8. The overview category should contain only overarching facts. Detailed
   academic, activity, accommodation, catering, fee, date, age, booking, and
   transfer information belongs in its dedicated category.
9. Each array item must be one self-contained factual English statement.
10. If the source itself reports conflicting values, preserve both as separate,
    carefully qualified statements.
"""


def extract_source_packet(
    client: OpenAI,
    source_id: str,
    source_text: str,
) -> dict[str, list[dict[str, Any]]]:
    cache_key = short_hash(
        PIPELINE_VERSION + MODEL + source_id + source_text
    )
    cache_path = EXTRACT_DIR / f"{source_id}_{cache_key}.json"

    if cache_path.exists() and not FORCE_REGENERATE:
        logger.info("Reading extraction cache: %s", cache_path)
        return normalize_packet(
            read_json(cache_path),
            allowed_source_ids={source_id},
        )

    user_prompt = f"""
Source identifier: {source_id}

Classify all useful facts from the source below into the five required
categories. Return only the schema-defined JSON object.

<source>
{source_text}
</source>
"""

    raw = call_json(
        client=client,
        task_name=f"extract-{source_id}",
        developer_prompt=EXTRACTION_DEVELOPER_PROMPT,
        user_prompt=user_prompt,
        schema_name="source_fact_classification",
        schema=CLASSIFICATION_SCHEMA,
        temperature=0.0,
    )

    packet: dict[str, list[dict[str, Any]]] = {
        section: [
            {
                "statement": normalize_statement(statement),
                "source_ids": [source_id],
            }
            for statement in raw.get(section, [])
            if normalize_statement(statement)
        ]
        for section in SECTION_NAMES
    }

    packet = normalize_packet(packet, allowed_source_ids={source_id})
    write_json(cache_path, packet)
    return packet


# ============================================================
# 7. Stage 2: pairwise merge, merge-sort style
# ============================================================

MERGE_DEVELOPER_PROMPT = """
You are merging two structured evidence packets about the same summer school.

Produce one consolidated evidence packet using the identical schema.

Rules:
1. Use only facts contained in the two input packets.
2. Do not write the final article yet.
3. Semantically deduplicate repeated facts, even when wording differs.
4. When two duplicate facts have different source_ids, keep one best,
   information-complete statement and combine all source_ids.
5. Preserve every unique concrete detail, especially numbers, dates, ages,
   prices, durations, class sizes, teaching hours, locations, named courses,
   activities, accommodation details, meals, conditions, and optional charges.
6. Remove empty promotional wording that contains no concrete program fact.
7. Never turn an optional item into a compulsory one.
8. Preserve limiting expressions such as "up to", "maximum", "minimum",
   "generally", "may", and "subject to availability".
9. Do not silently reconcile contradictions. Keep contradictory claims as
   separate qualified statements and retain their source_ids.
10. Keep each fact in the single best category and avoid cross-category
    duplication.
11. Do not create source_ids that are absent from the inputs.
"""


def merge_two_packets(
    client: OpenAI,
    left_label: str,
    left_packet: dict[str, Any],
    right_label: str,
    right_packet: dict[str, Any],
    level: int,
    pair_index: int,
) -> tuple[str, dict[str, list[dict[str, Any]]]]:
    allowed_sources = packet_source_ids(left_packet) | packet_source_ids(
        right_packet
    )
    combined_for_hash = stable_json(
        {
            "left": left_packet,
            "right": right_packet,
            "version": PIPELINE_VERSION,
            "model": MODEL,
        }
    )
    cache_key = short_hash(combined_for_hash)
    merged_label = f"({left_label}+{right_label})"
    cache_path = (
        MERGE_DIR
        / f"level_{level:02d}_pair_{pair_index:02d}_{cache_key}.json"
    )

    if cache_path.exists() and not FORCE_REGENERATE:
        logger.info("Reading merge cache: %s", cache_path)
        cached = normalize_packet(
            read_json(cache_path),
            allowed_source_ids=allowed_sources,
        )
        return merged_label, cached

    input_numbers = extract_numbers(
        evidence_text_for_all_sections(left_packet)
        + "\n"
        + evidence_text_for_all_sections(right_packet)
    )
    feedback = ""

    for validation_attempt in range(1, 4):
        user_prompt = f"""
Merge level: {level}
Pair: {pair_index}

LEFT PACKET ({left_label}):
{json.dumps(left_packet, ensure_ascii=False, indent=2)}

RIGHT PACKET ({right_label}):
{json.dumps(right_packet, ensure_ascii=False, indent=2)}

{feedback}

Return one deduplicated evidence packet. Preserve all source-supported,
non-redundant details.
"""

        raw = call_json(
            client=client,
            task_name=(
                f"merge-level-{level}-pair-{pair_index}"
                f"-validation-{validation_attempt}"
            ),
            developer_prompt=MERGE_DEVELOPER_PROMPT,
            user_prompt=user_prompt,
            schema_name="merged_evidence_packet",
            schema=PACKET_SCHEMA,
            temperature=0.0,
        )

        merged = normalize_packet(
            raw,
            allowed_source_ids=allowed_sources,
        )

        output_numbers = extract_numbers(
            evidence_text_for_all_sections(merged)
        )
        missing_numbers = sorted(input_numbers - output_numbers)

        empty_items = sum(
            len(merged.get(section, []))
            for section in SECTION_NAMES
        ) == 0

        if not missing_numbers and not empty_items:
            write_json(cache_path, merged)
            return merged_label, merged

        feedback_parts = []
        if missing_numbers:
            feedback_parts.append(
                "The previous merge omitted numeric details: "
                + ", ".join(missing_numbers)
                + ". Preserve the source-supported facts containing them."
            )
        if empty_items:
            feedback_parts.append(
                "The previous merge returned an empty evidence packet."
            )

        feedback = (
            "IMPORTANT CORRECTION FROM PROGRAMMATIC VALIDATION:\n"
            + "\n".join(feedback_parts)
        )

    raise RuntimeError(
        f"Merge validation failed at level {level}, pair {pair_index}."
    )


def evidence_text_for_all_sections(packet: dict[str, Any]) -> str:
    return "\n".join(
        item["statement"]
        for section in SECTION_NAMES
        for item in packet.get(section, [])
        if item.get("statement")
    )


def merge_sort_packets(
    client: OpenAI,
    source_nodes: list[tuple[str, dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    current = source_nodes
    level = 1

    while len(current) > 1:
        logger.info(
            "Merge level %d starts with %d nodes.",
            level,
            len(current),
        )
        next_level: list[tuple[str, dict[str, Any]]] = []

        pair_index = 1
        index = 0
        while index < len(current):
            if index + 1 >= len(current):
                next_level.append(current[index])
                index += 1
                continue

            left_label, left_packet = current[index]
            right_label, right_packet = current[index + 1]

            logger.info(
                "Merging level %d pair %d: %s + %s",
                level,
                pair_index,
                left_label,
                right_label,
            )

            merged_node = merge_two_packets(
                client=client,
                left_label=left_label,
                left_packet=left_packet,
                right_label=right_label,
                right_packet=right_packet,
                level=level,
                pair_index=pair_index,
            )
            next_level.append(merged_node)

            pair_index += 1
            index += 2

        current = next_level
        level += 1

    return normalize_packet(current[0][1])


# ============================================================
# 8. Stage 3: generate, validate, audit, and repair each section
# ============================================================

WRITING_DEVELOPER_PROMPT = """
You are a senior English-language editor writing a factual and persuasive
summer-school overview for prospective students and families.

Rules:
1. Use only the supplied evidence statements.
2. Do not invent or import facts, rankings, outcomes, guarantees, facilities,
   qualifications, prices, dates, or application rules.
3. Preserve all limiting language. Optional items must remain optional.
4. Resolve no contradiction unless the evidence explicitly resolves it.
5. Write polished, coherent English with a clear logical flow.
6. Make the content appealing through specificity and organization, not hype.
7. Avoid unsupported superlatives and guarantees.
8. Do not mention source files, evidence packets, JSON, or the writing process.
9. Do not include the section title; the Markdown exporter adds it.
10. Use connected prose paragraphs rather than bullet lists.
11. Avoid repeating the same point merely to reach the character target.
12. The output must be entirely in English.
"""


def hard_validate_section(
    section: str,
    content: str,
    evidence_text: str,
) -> list[str]:
    spec = SECTION_SPECS[section]
    counts = count_characters(content)
    issues: list[str] = []

    minimum = int(spec["minimum_non_whitespace"])
    if counts["non_whitespace"] <= minimum:
        issues.append(
            f"The draft has {counts['non_whitespace']} non-whitespace "
            f"characters; it must have more than {minimum}."
        )

    maximum_total = spec["maximum_total"]
    if maximum_total is not None and counts["total"] >= int(maximum_total):
        issues.append(
            f"The draft has {counts['total']} total characters; it must have "
            f"fewer than {maximum_total}."
        )

    if re.search(r"\b[1-8]_en\.txt\b", content):
        issues.append("The draft leaks internal source filenames.")

    if re.search(r"[\u4e00-\u9fff]", content):
        issues.append("The value contains Chinese text; the body must be English.")

    evidence_numbers = extract_numbers(evidence_text)
    draft_numbers = extract_numbers(content)
    unsupported_numbers = sorted(draft_numbers - evidence_numbers)
    if unsupported_numbers:
        issues.append(
            "The draft contains numeric strings absent from the evidence: "
            + ", ".join(unsupported_numbers)
            + "."
        )

    if not content.strip():
        issues.append("The draft is empty.")

    return issues


AUDIT_DEVELOPER_PROMPT = """
You are a strict factual and editorial auditor.

Compare the draft against the supplied evidence for the same section.

Fail the draft only for substantive issues:
1. A factual statement is unsupported by the evidence.
2. A number, date, price, age, duration, class size, or condition is wrong.
3. An optional or limited item is presented as guaranteed or compulsory.
4. A contradiction in the evidence is silently resolved or misrepresented.
5. A major evidence-supported detail relevant to this section is omitted.
6. The draft contains excessive repetition, incoherent organization, or
   unsupported promotional claims.
7. The draft discusses material belonging primarily to another section.

Do not fail it for harmless stylistic preferences. Return specific,
actionable issues.
"""


def semantic_audit(
    client: OpenAI,
    section: str,
    content: str,
    evidence_items: list[dict[str, Any]],
) -> dict[str, Any]:
    user_prompt = f"""
SECTION:
{section}

EVIDENCE:
{json.dumps(evidence_items, ensure_ascii=False, indent=2)}

DRAFT:
{content}

Audit the draft and return the schema-defined result.
"""

    return call_json(
        client=client,
        task_name=f"audit-{short_hash(section + content)}",
        developer_prompt=AUDIT_DEVELOPER_PROMPT,
        user_prompt=user_prompt,
        schema_name="section_audit",
        schema=AUDIT_SCHEMA,
        temperature=0.0,
    )


def create_initial_draft(
    client: OpenAI,
    section: str,
    evidence_items: list[dict[str, Any]],
) -> str:
    spec = SECTION_SPECS[section]
    cache_key = short_hash(
        PIPELINE_VERSION
        + MODEL
        + section
        + stable_json(evidence_items)
        + stable_json(spec)
    )
    cache_path = DRAFT_DIR / f"{SECTION_NAMES.index(section) + 1}_{cache_key}.json"

    if cache_path.exists() and not FORCE_REGENERATE:
        logger.info("Reading draft cache: %s", cache_path)
        return str(read_json(cache_path)["content"]).strip()

    user_prompt = f"""
Write the final English body for this required section:

SECTION KEY:
{section}

CONTENT FOCUS:
{spec["focus"]}

CHARACTER TARGET:
{spec["target"]}

EVIDENCE:
{json.dumps(evidence_items, ensure_ascii=False, indent=2)}

Return only a JSON object containing the complete body in the "content" field.
Do not include the section title.
"""

    result = call_json(
        client=client,
        task_name=f"draft-{SECTION_NAMES.index(section) + 1}",
        developer_prompt=WRITING_DEVELOPER_PROMPT,
        user_prompt=user_prompt,
        schema_name="section_draft",
        schema=CONTENT_SCHEMA,
        temperature=0.25,
    )

    content = str(result["content"]).strip()
    write_json(cache_path, {"content": content})
    return content


REPAIR_DEVELOPER_PROMPT = """
You are revising an English summer-school section under strict factual and
character-count constraints.

Rules:
1. Fix every listed issue.
2. Use only the supplied evidence.
3. Return the complete revised section, not a patch or commentary.
4. Do not invent facts to increase length.
5. Increase useful detail by organizing and explaining documented features,
   but do not repeat sentences or make unsupported promises.
6. Preserve every relevant number, condition, and optional limitation.
7. Do not mention sources, evidence, validation, or character counting.
8. Do not include the section title.
9. Use coherent English paragraphs, not bullet lists.
"""


def repair_draft(
    client: OpenAI,
    section: str,
    content: str,
    evidence_items: list[dict[str, Any]],
    issues: list[str],
    round_number: int,
) -> str:
    spec = SECTION_SPECS[section]

    user_prompt = f"""
SECTION:
{section}

CONTENT FOCUS:
{spec["focus"]}

CHARACTER TARGET:
{spec["target"]}

EVIDENCE:
{json.dumps(evidence_items, ensure_ascii=False, indent=2)}

CURRENT DRAFT:
{content}

ISSUES TO FIX:
{json.dumps(issues, ensure_ascii=False, indent=2)}

Return the complete corrected body in the schema-defined "content" field.
"""

    result = call_json(
        client=client,
        task_name=(
            f"repair-{SECTION_NAMES.index(section) + 1}-round-{round_number}"
        ),
        developer_prompt=REPAIR_DEVELOPER_PROMPT,
        user_prompt=user_prompt,
        schema_name="repaired_section",
        schema=CONTENT_SCHEMA,
        temperature=0.15,
    )
    return str(result["content"]).strip()


def generate_validated_section(
    client: OpenAI,
    section: str,
    final_packet: dict[str, list[dict[str, Any]]],
) -> str:
    evidence_items = final_packet.get(section, [])
    if not evidence_items:
        raise RuntimeError(
            f"No evidence was extracted for required section: {section}"
        )

    evidence_text = "\n".join(
        item["statement"] for item in evidence_items
    )
    content = create_initial_draft(client, section, evidence_items)

    for round_number in range(1, MAX_REPAIR_ROUNDS + 1):
        issues = hard_validate_section(section, content, evidence_text)

        audit = semantic_audit(
            client=client,
            section=section,
            content=content,
            evidence_items=evidence_items,
        )

        if not audit.get("passed", False):
            audit_issues = [
                normalize_statement(str(issue))
                for issue in audit.get("issues", [])
                if normalize_statement(str(issue))
            ]
            if audit_issues:
                issues.extend(audit_issues)
            else:
                issues.append(
                    "The semantic audit failed without a specific explanation."
                )

        # Preserve order while removing duplicate issue strings.
        issues = list(dict.fromkeys(issues))

        counts = count_characters(content)
        logger.info(
            "%s round %d: total=%d non-whitespace=%d issues=%d",
            section,
            round_number,
            counts["total"],
            counts["non_whitespace"],
            len(issues),
        )

        if not issues:
            return content

        content = repair_draft(
            client=client,
            section=section,
            content=content,
            evidence_items=evidence_items,
            issues=issues,
            round_number=round_number,
        )

    final_issues = hard_validate_section(section, content, evidence_text)
    if final_issues:
        raise RuntimeError(
            f"{section} still fails hard validation: {final_issues}"
        )

    final_audit = semantic_audit(
        client=client,
        section=section,
        content=content,
        evidence_items=evidence_items,
    )
    if not final_audit.get("passed", False):
        raise RuntimeError(
            f"{section} still fails semantic audit: "
            f"{final_audit.get('issues', [])}"
        )

    return content


# ============================================================
# 9. Final output validation and export
# ============================================================

def validate_final_result(result: dict[str, str]) -> dict[str, Any]:
    if list(result.keys()) != SECTION_NAMES:
        raise RuntimeError(
            "Final JSON keys or their order are incorrect.\n"
            f"Expected: {SECTION_NAMES}\n"
            f"Actual: {list(result.keys())}"
        )

    report: dict[str, Any] = {
        "model": MODEL,
        "counting_policy": {
            "minimums": "non-whitespace characters",
            "introduction_maximum": "all characters including whitespace",
        },
        "sections": {},
    }

    for section in SECTION_NAMES:
        content = result[section]
        counts = count_characters(content)
        spec = SECTION_SPECS[section]

        passed_minimum = (
            counts["non_whitespace"]
            > int(spec["minimum_non_whitespace"])
        )

        maximum_total = spec["maximum_total"]
        passed_maximum = (
            maximum_total is None
            or counts["total"] < int(maximum_total)
        )

        report["sections"][section] = {
            **counts,
            "required_non_whitespace_greater_than": spec[
                "minimum_non_whitespace"
            ],
            "required_total_less_than": maximum_total,
            "passed": passed_minimum and passed_maximum,
        }

        if not passed_minimum or not passed_maximum:
            raise RuntimeError(
                f"Final character validation failed for {section}: "
                f"{report['sections'][section]}"
            )

    return report


def export_markdown(result: dict[str, str], path: Path) -> None:
    blocks = [
        f"# {section}\n\n{result[section].strip()}"
        for section in SECTION_NAMES
    ]
    write_text(path, "\n\n".join(blocks))


# ============================================================
# 10. Main one-shot pipeline
# ============================================================

def main() -> None:
    ensure_directories()
    client = build_client()

    input_files = [INPUT_DIR / f"{index}_en.txt" for index in range(1, 9)]
    missing = [str(path) for path in input_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "The following required input files are missing:\n"
            + "\n".join(missing)
        )

    logger.warning("Stage 1/3: extracting and classifying eight sources.")
    source_nodes: list[tuple[str, dict[str, Any]]] = []

    for path in input_files:
        source_id = path.name
        source_text = read_text(path).strip()
        if not source_text:
            raise RuntimeError(f"Input file is empty: {path}")

        logger.info(
            "Processing %s (%d characters)",
            source_id,
            len(source_text),
        )
        packet = extract_source_packet(
            client=client,
            source_id=source_id,
            source_text=source_text,
        )
        source_nodes.append((source_id, packet))

    logger.warning("Stage 2/3: pairwise merge-sort consolidation.")
    final_packet = merge_sort_packets(client, source_nodes)
    write_json(OUTPUT_EVIDENCE, final_packet)

    logger.warning("Stage 3/3: writing and validating five final sections.")
    final_result: dict[str, str] = {}

    for section in SECTION_NAMES:
        logger.warning("Writing section: %s", section)
        final_result[section] = generate_validated_section(
            client=client,
            section=section,
            final_packet=final_packet,
        )

    report = validate_final_result(final_result)

    write_json(OUTPUT_JSON, final_result)
    export_markdown(final_result, OUTPUT_MD)
    write_json(OUTPUT_REPORT, report)

    print()
    print("=" * 72)
    print("TASK COMPLETED")
    print("=" * 72)
    print(f"JSON:     {OUTPUT_JSON.resolve()}")
    print(f"Markdown: {OUTPUT_MD.resolve()}")
    print(f"Evidence: {OUTPUT_EVIDENCE.resolve()}")
    print(f"Report:   {OUTPUT_REPORT.resolve()}")
    print()

    for section in SECTION_NAMES:
        counts = report["sections"][section]
        print(
            f"{section}: total={counts['total']}, "
            f"non-whitespace={counts['non_whitespace']}, "
            f"passed={counts['passed']}"
        )


if __name__ == "__main__":
    main()
