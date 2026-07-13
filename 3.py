from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from openai import DefaultHttpxClient, OpenAI


# ============================================================
# 1. Configuration
# ============================================================

# Keep this model if the interview requires the GPT-4.1 mini snapshot.
MODEL = "gpt-4.1-mini-2025-04-14"

# Preferred input: one large raw corpus.
PRIMARY_INPUT_FILE = Path("web_contents.txt")

# Fallback input: when web_contents.txt does not exist, combine all *_en.txt
# files in the current directory as one corpus.
FALLBACK_INPUT_GLOB = "*_en.txt"

# The final article language.
OUTPUT_LANGUAGE = "English"

# Fixed-size chunking. Paragraph boundaries are preferred; a single oversized
# paragraph is hard-split with a small overlap.
MAX_CHARS_PER_CHUNK = 16000
HARD_SPLIT_OVERLAP_CHARS = 400

# The model may create at most ten evidence-supported sections.
MIN_SECTIONS = 2
MAX_SECTIONS = 10

# A taxonomy section should have enough source substance to support a long,
# grounded final section. Lower this only when the source is unusually sparse.
MIN_EVIDENCE_CHARS_PER_SECTION = 500

# Every final section must contain more than 1,500 English characters.
# The script validates NON-WHITESPACE characters, which is stricter than
# counting spaces.
MIN_FINAL_NON_WHITESPACE_CHARS = 1500
TARGET_FINAL_TOTAL_CHARS = "approximately 2,300-2,800 total characters"

# API and repair settings.
MAX_API_ATTEMPTS = 4
MAX_MERGE_VALIDATION_ATTEMPTS = 3
MAX_TAXONOMY_ATTEMPTS = 4
MAX_DRAFT_REPAIR_ROUNDS = 4

# Local proxy. Set to None when no proxy is needed.
PROXY_URL: str | None = "http://127.0.0.1:1080"

# Set True to ignore caches and rerun every stage.
FORCE_REGENERATE = False

# Increment this after materially changing prompts or schemas.
PIPELINE_VERSION = "task2-generic-v1"

# Output files.
OUTPUT_JSON = Path("task2.json")
OUTPUT_MD = Path("task2.md")
OUTPUT_EVIDENCE = Path("task2_evidence.json")
OUTPUT_TAXONOMY = Path("task2_taxonomy.json")
OUTPUT_REPORT = Path("task2_report.json")

# Intermediate cache directories.
WORK_DIR = Path("task2_work")
CHUNK_DIR = WORK_DIR / "00_chunks"
EXTRACT_DIR = WORK_DIR / "01_extracted"
MERGE_DIR = WORK_DIR / "02_merged"
TAXONOMY_DIR = WORK_DIR / "03_taxonomy"
DRAFT_DIR = WORK_DIR / "04_drafts"


# ============================================================
# 2. Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
# 3. Structured-output schemas
# ============================================================

EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "theme_hints": {
            "type": "array",
            "items": {"type": "string"},
        },
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "statement": {"type": "string"},
                    "quote": {"type": "string"},
                    "candidate_dimensions": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "statement",
                    "quote",
                    "candidate_dimensions",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["theme_hints", "facts"],
    "additionalProperties": False,
}

MERGED_FACT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "statement": {"type": "string"},
        "evidence_ids": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        },
        "candidate_dimensions": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": [
        "statement",
        "evidence_ids",
        "candidate_dimensions",
    ],
    "additionalProperties": False,
}

MERGED_PACKET_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "theme_hints": {
            "type": "array",
            "items": {"type": "string"},
        },
        "facts": {
            "type": "array",
            "items": MERGED_FACT_SCHEMA,
        },
    },
    "required": ["theme_hints", "facts"],
    "additionalProperties": False,
}

TAXONOMY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "document_theme": {"type": "string"},
        "intended_audience": {"type": "string"},
        "section_plan": {
            "type": "array",
            "minItems": MIN_SECTIONS,
            "maxItems": MAX_SECTIONS,
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "purpose": {"type": "string"},
                    "fact_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                    },
                },
                "required": ["name", "purpose", "fact_ids"],
                "additionalProperties": False,
            },
        },
    },
    "required": [
        "document_theme",
        "intended_audience",
        "section_plan",
    ],
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
        "omitted_fact_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["passed", "issues", "omitted_fact_ids"],
    "additionalProperties": False,
}


# ============================================================
# 4. Generic utilities
# ============================================================

def ensure_directories() -> None:
    for directory in [
        WORK_DIR,
        CHUNK_DIR,
        EXTRACT_DIR,
        MERGE_DIR,
        TAXONOMY_DIR,
        DRAFT_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def read_text(path: Path) -> str:
    encodings = ["utf-8", "utf-8-sig", "gb18030"]
    last_error: Exception | None = None

    for encoding in encodings:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError as error:
            last_error = error

    raise RuntimeError(f"Unable to decode text file: {path}") from last_error


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def stable_json(data: Any) -> str:
    return json.dumps(
        data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_for_quote_match(text: str) -> str:
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u00a0": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return re.sub(r"\s+", "", text).lower()


def dedupe_strings(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()

    for value in values:
        cleaned = normalize_space(value)
        key = re.sub(r"[^a-z0-9]+", "", cleaned.lower())
        if cleaned and key and key not in seen:
            result.append(cleaned)
            seen.add(key)

    return result


def extract_numbers(text: str) -> set[str]:
    values = re.findall(r"\d+(?:[.,]\d+)*", text)
    return {value.replace(",", "").rstrip(".") for value in values}


def count_characters(text: str) -> dict[str, int]:
    return {
        "total": len(text),
        "non_whitespace": len(re.sub(r"\s+", "", text)),
    }


def build_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set.\n"
            'PowerShell:\n$env:OPENAI_API_KEY="your-api-key"'
        )

    arguments: dict[str, Any] = {
        "api_key": api_key,
        "timeout": 240.0,
        "max_retries": 1,
    }

    if PROXY_URL:
        logger.info("Using proxy: %s", PROXY_URL)
        arguments["http_client"] = DefaultHttpxClient(proxy=PROXY_URL)
    else:
        logger.info("Using a direct API connection.")

    return OpenAI(**arguments)


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

            result = json.loads(content)

            if response.usage:
                logger.info(
                    "%s tokens: input=%s output=%s total=%s",
                    task_name,
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                    response.usage.total_tokens,
                )

            return result

        except Exception as error:
            last_error = error
            logger.warning("%s failed: %s", task_name, error)

            if attempt < MAX_API_ATTEMPTS:
                time.sleep(attempt * 5)

    raise RuntimeError(
        f"{task_name} failed after {MAX_API_ATTEMPTS} attempts: {last_error}"
    )


# ============================================================
# 5. Load and fixed-size chunk the corpus
# ============================================================

def discover_input_files() -> list[Path]:
    if PRIMARY_INPUT_FILE.is_file():
        return [PRIMARY_INPUT_FILE]

    fallback_files = sorted(
        path
        for path in Path(".").glob(FALLBACK_INPUT_GLOB)
        if path.is_file()
    )

    if not fallback_files:
        raise FileNotFoundError(
            f"Neither {PRIMARY_INPUT_FILE} nor any "
            f"{FALLBACK_INPUT_GLOB} files were found."
        )

    return fallback_files


def load_corpus() -> tuple[str, list[str]]:
    input_files = discover_input_files()
    blocks: list[str] = []
    names: list[str] = []

    for path in input_files:
        text = read_text(path).strip()
        if not text:
            logger.warning("Skipping empty input file: %s", path)
            continue

        names.append(path.name)
        blocks.append(
            f"<DOCUMENT filename={json.dumps(path.name)}>\n"
            f"{text}\n"
            f"</DOCUMENT>"
        )

    if not blocks:
        raise RuntimeError("All discovered input files were empty.")

    return "\n\n".join(blocks), names


def hard_split_text(
    text: str,
    max_chars: int,
    overlap_chars: int,
) -> list[str]:
    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = min(start + max_chars, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(end - overlap_chars, start + 1)

    return chunks


def split_corpus(
    text: str,
    max_chars: int = MAX_CHARS_PER_CHUNK,
) -> list[str]:
    text = text.strip()
    if not text:
        return []

    if len(text) <= max_chars:
        return [text]

    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", text)
        if paragraph.strip()
    ]

    if len(paragraphs) <= 1:
        paragraphs = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

    chunks: list[str] = []
    current: list[str] = []
    current_length = 0

    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if current:
                chunks.append("\n\n".join(current))
                current = []
                current_length = 0

            chunks.extend(
                hard_split_text(
                    paragraph,
                    max_chars=max_chars,
                    overlap_chars=HARD_SPLIT_OVERLAP_CHARS,
                )
            )
            continue

        separator_length = 2 if current else 0
        candidate_length = current_length + separator_length + len(paragraph)

        if current and candidate_length > max_chars:
            chunks.append("\n\n".join(current))
            current = [paragraph]
            current_length = len(paragraph)
        else:
            current.append(paragraph)
            current_length = candidate_length

    if current:
        chunks.append("\n\n".join(current))

    return [chunk for chunk in chunks if chunk.strip()]


def save_chunks(chunks: list[str]) -> None:
    for index, chunk in enumerate(chunks, start=1):
        write_text(CHUNK_DIR / f"C{index:03d}.txt", chunk)


# ============================================================
# 6. Stage 1: topic-neutral evidence extraction per chunk
# ============================================================

EXTRACTION_DEVELOPER_PROMPT = """
You are a rigorous, topic-neutral source analyst.

The subject of the corpus is unknown. Do not assume any domain, product,
industry, organization, service, event, or audience before reading the source.

Extract atomic, source-supported facts from the supplied chunk.

Rules:
1. Use only information explicitly stated in the chunk.
2. Do not infer missing information or use external knowledge.
3. Do not produce an article or promotional copy.
4. Each fact must be self-contained and materially useful for a later article.
5. Preserve names, dates, quantities, prices, measurements, eligibility rules,
   technical specifications, processes, conditions, limitations, optional
   items, exclusions, and qualifications exactly in meaning.
6. Remove navigation labels, repeated boilerplate, and content-free slogans.
7. For every fact, copy a short supporting quote verbatim from the chunk.
8. Suggest one or more concise candidate dimensions derived from the source
   itself. These are provisional labels, not a fixed taxonomy.
9. Do not create a dimension for information absent from the chunk.
10. Keep uncertainty and limiting language such as "may", "up to", "usually",
    "optional", "minimum", "maximum", and "subject to".
11. Write extracted fact statements and labels in English.
"""


def extract_chunk(
    client: OpenAI,
    chunk_id: str,
    chunk_text: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    cache_key = short_hash(
        PIPELINE_VERSION + MODEL + chunk_id + chunk_text
    )
    cache_path = EXTRACT_DIR / f"{chunk_id}_{cache_key}.json"

    if cache_path.exists() and not FORCE_REGENERATE:
        cached = read_json(cache_path)
        return cached["packet"], cached["registry"]

    user_prompt = f"""
Chunk identifier: {chunk_id}

Analyze the following source chunk without assuming its topic.

<source_chunk>
{chunk_text}
</source_chunk>
"""

    raw = call_json(
        client=client,
        task_name=f"extract-{chunk_id}",
        developer_prompt=EXTRACTION_DEVELOPER_PROMPT,
        user_prompt=user_prompt,
        schema_name="generic_chunk_extraction",
        schema=EXTRACTION_SCHEMA,
        temperature=0.0,
    )

    normalized_source = normalize_for_quote_match(chunk_text)
    registry: dict[str, dict[str, Any]] = {}
    packet_facts: list[dict[str, Any]] = []

    valid_index = 1
    for raw_fact in raw.get("facts", []):
        statement = normalize_space(str(raw_fact.get("statement", "")))
        quote = normalize_space(str(raw_fact.get("quote", "")))
        candidate_dimensions = dedupe_strings(
            [
                str(item)
                for item in raw_fact.get("candidate_dimensions", [])
            ]
        )

        if not statement or not quote:
            continue

        if normalize_for_quote_match(quote) not in normalized_source:
            logger.warning(
                "Discarding unverifiable quote in %s: %s",
                chunk_id,
                quote[:100],
            )
            continue

        evidence_id = f"E_{chunk_id}_{valid_index:04d}"
        valid_index += 1

        registry[evidence_id] = {
            "evidence_id": evidence_id,
            "chunk_id": chunk_id,
            "statement": statement,
            "quote": quote,
            "candidate_dimensions": candidate_dimensions,
        }

        packet_facts.append(
            {
                "statement": statement,
                "evidence_ids": [evidence_id],
                "candidate_dimensions": candidate_dimensions,
            }
        )

    if not packet_facts:
        raise RuntimeError(
            f"No verifiable facts were extracted from chunk {chunk_id}."
        )

    packet = {
        "theme_hints": dedupe_strings(
            [str(item) for item in raw.get("theme_hints", [])]
        ),
        "facts": packet_facts,
    }

    write_json(
        cache_path,
        {
            "packet": packet,
            "registry": registry,
        },
    )
    return packet, registry


# ============================================================
# 7. Stage 2: pairwise merge and semantic deduplication
# ============================================================

MERGE_DEVELOPER_PROMPT = """
You are consolidating two topic-neutral evidence packets derived from the same
larger corpus.

Rules:
1. Use only the facts in the two input packets.
2. Do not write the final article.
3. Semantically merge duplicate facts even when wording differs.
4. Preserve every unique source-supported detail.
5. The union of evidence_ids must be preserved exactly: no missing IDs,
   no invented IDs, and each ID must appear in exactly one output fact.
6. When merging duplicate facts, combine their evidence_ids and write one
   precise, information-complete statement.
7. Preserve all names, numbers, dates, prices, units, conditions, limitations,
   exclusions, optional items, and uncertainty.
8. Do not reconcile genuine contradictions. Keep them as separate, qualified
   facts.
9. Deduplicate and refine candidate dimensions, but derive them only from the
   input facts.
10. Remove only redundancy, never unique information.
"""


def packet_evidence_ids(packet: dict[str, Any]) -> list[str]:
    return [
        evidence_id
        for fact in packet.get("facts", [])
        for evidence_id in fact.get("evidence_ids", [])
    ]


def validate_merged_packet(
    packet: dict[str, Any],
    expected_evidence_ids: set[str],
    registry: dict[str, dict[str, Any]],
) -> list[str]:
    issues: list[str] = []
    output_ids = packet_evidence_ids(packet)
    output_counter = Counter(output_ids)

    missing = sorted(expected_evidence_ids - set(output_ids))
    unknown = sorted(set(output_ids) - expected_evidence_ids)
    duplicated = sorted(
        evidence_id
        for evidence_id, count in output_counter.items()
        if count != 1
    )

    if missing:
        issues.append("Missing evidence_ids: " + ", ".join(missing))
    if unknown:
        issues.append("Unknown evidence_ids: " + ", ".join(unknown))
    if duplicated:
        issues.append(
            "Evidence_ids must appear exactly once: "
            + ", ".join(duplicated)
        )

    for index, fact in enumerate(packet.get("facts", []), start=1):
        statement = normalize_space(str(fact.get("statement", "")))
        ids = [str(item) for item in fact.get("evidence_ids", [])]

        if not statement:
            issues.append(f"Fact {index} has an empty statement.")
            continue

        supporting_text = "\n".join(
            registry[evidence_id]["statement"]
            + "\n"
            + registry[evidence_id]["quote"]
            for evidence_id in ids
            if evidence_id in registry
        )

        unsupported_numbers = sorted(
            extract_numbers(statement) - extract_numbers(supporting_text)
        )
        if unsupported_numbers:
            issues.append(
                f"Fact {index} contains unsupported numbers: "
                + ", ".join(unsupported_numbers)
            )

    return issues


def normalize_merged_packet(packet: dict[str, Any]) -> dict[str, Any]:
    facts: list[dict[str, Any]] = []

    for raw_fact in packet.get("facts", []):
        statement = normalize_space(str(raw_fact.get("statement", "")))
        evidence_ids = sorted(
            {
                str(item).strip()
                for item in raw_fact.get("evidence_ids", [])
                if str(item).strip()
            }
        )
        dimensions = dedupe_strings(
            [
                str(item)
                for item in raw_fact.get("candidate_dimensions", [])
            ]
        )

        if statement and evidence_ids:
            facts.append(
                {
                    "statement": statement,
                    "evidence_ids": evidence_ids,
                    "candidate_dimensions": dimensions,
                }
            )

    return {
        "theme_hints": dedupe_strings(
            [str(item) for item in packet.get("theme_hints", [])]
        ),
        "facts": facts,
    }


def merge_two_packets(
    client: OpenAI,
    left_label: str,
    left_packet: dict[str, Any],
    right_label: str,
    right_packet: dict[str, Any],
    registry: dict[str, dict[str, Any]],
    level: int,
    pair_index: int,
) -> tuple[str, dict[str, Any]]:
    expected_ids = set(packet_evidence_ids(left_packet)) | set(
        packet_evidence_ids(right_packet)
    )

    cache_key = short_hash(
        PIPELINE_VERSION
        + MODEL
        + stable_json(left_packet)
        + stable_json(right_packet)
    )
    cache_path = (
        MERGE_DIR
        / f"level_{level:02d}_pair_{pair_index:03d}_{cache_key}.json"
    )

    if cache_path.exists() and not FORCE_REGENERATE:
        cached = normalize_merged_packet(read_json(cache_path))
        issues = validate_merged_packet(cached, expected_ids, registry)
        if not issues:
            return f"({left_label}+{right_label})", cached

    feedback = ""

    for attempt in range(1, MAX_MERGE_VALIDATION_ATTEMPTS + 1):
        user_prompt = f"""
Merge level: {level}
Pair number: {pair_index}

LEFT PACKET ({left_label}):
{json.dumps(left_packet, ensure_ascii=False, indent=2)}

RIGHT PACKET ({right_label}):
{json.dumps(right_packet, ensure_ascii=False, indent=2)}

{feedback}

Return one consolidated packet.
"""

        raw = call_json(
            client=client,
            task_name=f"merge-L{level}-P{pair_index}-A{attempt}",
            developer_prompt=MERGE_DEVELOPER_PROMPT,
            user_prompt=user_prompt,
            schema_name="generic_merged_packet",
            schema=MERGED_PACKET_SCHEMA,
            temperature=0.0,
        )

        merged = normalize_merged_packet(raw)
        issues = validate_merged_packet(
            merged,
            expected_evidence_ids=expected_ids,
            registry=registry,
        )

        if not issues:
            write_json(cache_path, merged)
            return f"({left_label}+{right_label})", merged

        feedback = (
            "PROGRAMMATIC VALIDATION FAILED. Correct every issue below:\n- "
            + "\n- ".join(issues)
        )

    raise RuntimeError(
        f"Unable to produce a valid merge at level {level}, "
        f"pair {pair_index}."
    )


def merge_sort_packets(
    client: OpenAI,
    source_nodes: list[tuple[str, dict[str, Any]]],
    registry: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    current = source_nodes
    level = 1

    while len(current) > 1:
        logger.info(
            "Merge level %d begins with %d nodes.",
            level,
            len(current),
        )
        next_level: list[tuple[str, dict[str, Any]]] = []

        index = 0
        pair_index = 1

        while index < len(current):
            if index + 1 >= len(current):
                next_level.append(current[index])
                index += 1
                continue

            left_label, left_packet = current[index]
            right_label, right_packet = current[index + 1]

            merged_node = merge_two_packets(
                client=client,
                left_label=left_label,
                left_packet=left_packet,
                right_label=right_label,
                right_packet=right_packet,
                registry=registry,
                level=level,
                pair_index=pair_index,
            )
            next_level.append(merged_node)

            index += 2
            pair_index += 1

        current = next_level
        level += 1

    return normalize_merged_packet(current[0][1])


# ============================================================
# 8. Stage 3: infer a source-driven taxonomy with N <= 10
# ============================================================

TAXONOMY_DEVELOPER_PROMPT = f"""
You are designing a topic-neutral article structure from a consolidated fact
inventory. The corpus topic is unknown until you inspect the facts.

Create a source-driven taxonomy with between {MIN_SECTIONS} and
{MAX_SECTIONS} sections.

Rules:
1. Do not use a pre-existing domain template.
2. Infer the theme, audience, and section dimensions only from the facts.
3. Do not create a section for a type of information absent from the facts.
4. Every canonical fact_id must be assigned to exactly one section.
5. Every section must contain at least one fact_id.
6. Use the smallest number of sections that still creates a coherent,
   complete, non-overlapping article.
7. Each section must have enough evidence to support a grounded final body
   longer than 1,500 characters. Merge narrow or weak dimensions instead of
   creating thin sections.
8. Section names must be distinct, informative, and written in
   {OUTPUT_LANGUAGE}.
9. The section order should create a natural persuasive progression for the
   audience supported by the corpus.
10. Do not mention evidence IDs or the taxonomy process in section names.
"""


def canonicalize_facts(
    merged_packet: dict[str, Any],
) -> list[dict[str, Any]]:
    canonical: list[dict[str, Any]] = []

    for index, fact in enumerate(merged_packet.get("facts", []), start=1):
        canonical.append(
            {
                "fact_id": f"F{index:04d}",
                "statement": normalize_space(str(fact["statement"])),
                "evidence_ids": list(fact["evidence_ids"]),
                "candidate_dimensions": list(
                    fact.get("candidate_dimensions", [])
                ),
            }
        )

    return canonical


def taxonomy_validation_issues(
    taxonomy: dict[str, Any],
    canonical_facts: list[dict[str, Any]],
) -> list[str]:
    issues: list[str] = []
    valid_ids = {fact["fact_id"] for fact in canonical_facts}
    fact_map = {fact["fact_id"]: fact for fact in canonical_facts}
    sections = taxonomy.get("section_plan", [])

    if not (MIN_SECTIONS <= len(sections) <= MAX_SECTIONS):
        issues.append(
            f"The taxonomy must contain {MIN_SECTIONS}-{MAX_SECTIONS} "
            f"sections; it contains {len(sections)}."
        )

    names = [
        normalize_space(str(section.get("name", "")))
        for section in sections
    ]
    if any(not name for name in names):
        issues.append("Every section must have a non-empty name.")
    if len(set(name.lower() for name in names)) != len(names):
        issues.append("Section names must be unique.")

    assigned_ids = [
        str(fact_id)
        for section in sections
        for fact_id in section.get("fact_ids", [])
    ]
    counts = Counter(assigned_ids)

    missing = sorted(valid_ids - set(assigned_ids))
    unknown = sorted(set(assigned_ids) - valid_ids)
    duplicated = sorted(
        fact_id for fact_id, count in counts.items() if count != 1
    )

    if missing:
        issues.append("Unassigned fact_ids: " + ", ".join(missing))
    if unknown:
        issues.append("Unknown fact_ids: " + ", ".join(unknown))
    if duplicated:
        issues.append(
            "Fact_ids must appear in exactly one section: "
            + ", ".join(duplicated)
        )

    for index, section in enumerate(sections, start=1):
        fact_ids = [str(item) for item in section.get("fact_ids", [])]
        if not fact_ids:
            issues.append(f"Section {index} has no facts.")
            continue

        evidence_chars = sum(
            len(fact_map[fact_id]["statement"])
            for fact_id in fact_ids
            if fact_id in fact_map
        )

        if evidence_chars < MIN_EVIDENCE_CHARS_PER_SECTION:
            issues.append(
                f'Section "{names[index - 1]}" has only {evidence_chars} '
                "characters of canonical evidence. Merge it with a related "
                "section or reassign facts."
            )

    return issues


def infer_taxonomy(
    client: OpenAI,
    canonical_facts: list[dict[str, Any]],
    theme_hints: list[str],
) -> dict[str, Any]:
    cache_key = short_hash(
        PIPELINE_VERSION
        + MODEL
        + stable_json(canonical_facts)
        + stable_json(theme_hints)
    )
    cache_path = TAXONOMY_DIR / f"taxonomy_{cache_key}.json"

    if cache_path.exists() and not FORCE_REGENERATE:
        cached = read_json(cache_path)
        if not taxonomy_validation_issues(cached, canonical_facts):
            return cached

    feedback = ""

    for attempt in range(1, MAX_TAXONOMY_ATTEMPTS + 1):
        user_prompt = f"""
Provisional source-derived theme hints:
{json.dumps(theme_hints, ensure_ascii=False, indent=2)}

Canonical facts:
{json.dumps(canonical_facts, ensure_ascii=False, indent=2)}

{feedback}

Infer the theme and produce the complete source-driven section plan.
"""

        taxonomy = call_json(
            client=client,
            task_name=f"infer-taxonomy-A{attempt}",
            developer_prompt=TAXONOMY_DEVELOPER_PROMPT,
            user_prompt=user_prompt,
            schema_name="generic_source_taxonomy",
            schema=TAXONOMY_SCHEMA,
            temperature=0.1,
        )

        issues = taxonomy_validation_issues(taxonomy, canonical_facts)
        if not issues:
            write_json(cache_path, taxonomy)
            return taxonomy

        feedback = (
            "PROGRAMMATIC VALIDATION FAILED. Redesign the taxonomy and fix "
            "all issues:\n- "
            + "\n- ".join(issues)
        )

    raise RuntimeError("Unable to infer a valid source-driven taxonomy.")


# ============================================================
# 9. Stage 4: write, audit, and repair each section
# ============================================================

WRITING_DEVELOPER_PROMPT = f"""
You are a senior {OUTPUT_LANGUAGE}-language commercial editor.

Write one section of a persuasive, coherent article based solely on the
supplied facts and source excerpts.

Rules:
1. Use only the assigned facts and supporting excerpts.
2. Do not add external knowledge, assumptions, invented benefits, rankings,
   prices, specifications, guarantees, outcomes, or missing details.
3. Preserve all numbers, names, limitations, exclusions, options, uncertainty,
   and conditions accurately.
4. Persuade through concrete documented information, reader-oriented
   organization, clarity, and explanation—not hype or fabrication.
5. Do not claim that a feature is included, guaranteed, compulsory, or
   universal unless the evidence says so.
6. Do not mention fact IDs, evidence IDs, chunks, files, prompts, JSON, or the
   generation process.
7. Do not include the section title in the body.
8. Use connected prose paragraphs rather than bullet lists.
9. Avoid padding, circular repetition, and unsupported adjectives.
10. Write entirely in {OUTPUT_LANGUAGE}.
11. The section must contain more than
    {MIN_FINAL_NON_WHITESPACE_CHARS} non-whitespace characters.
12. Aim for {TARGET_FINAL_TOTAL_CHARS}.
"""


def build_section_evidence(
    section: dict[str, Any],
    fact_map: dict[str, dict[str, Any]],
    registry: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []

    for fact_id in section["fact_ids"]:
        fact = fact_map[fact_id]
        excerpts = [
            {
                "evidence_id": evidence_id,
                "quote": registry[evidence_id]["quote"],
            }
            for evidence_id in fact["evidence_ids"]
            if evidence_id in registry
        ]

        evidence.append(
            {
                "fact_id": fact_id,
                "statement": fact["statement"],
                "supporting_excerpts": excerpts,
            }
        )

    return evidence


def hard_validate_draft(
    content: str,
    evidence: list[dict[str, Any]],
) -> list[str]:
    issues: list[str] = []
    counts = count_characters(content)

    if counts["non_whitespace"] <= MIN_FINAL_NON_WHITESPACE_CHARS:
        issues.append(
            f"The body has {counts['non_whitespace']} non-whitespace "
            f"characters; it must have more than "
            f"{MIN_FINAL_NON_WHITESPACE_CHARS}."
        )

    evidence_text = "\n".join(
        item["statement"]
        + "\n"
        + "\n".join(
            excerpt["quote"]
            for excerpt in item["supporting_excerpts"]
        )
        for item in evidence
    )

    unsupported_numbers = sorted(
        extract_numbers(content) - extract_numbers(evidence_text)
    )
    if unsupported_numbers:
        issues.append(
            "The body contains numeric strings absent from its evidence: "
            + ", ".join(unsupported_numbers)
        )

    if re.search(r"\b(?:F|E|C)\d{3,}\b", content):
        issues.append("The body leaks internal evidence identifiers.")

    if re.search(
        r"\b(?:source chunk|evidence packet|fact id|prompt|json schema)\b",
        content,
        flags=re.IGNORECASE,
    ):
        issues.append("The body mentions the internal generation process.")

    if OUTPUT_LANGUAGE.lower() == "english" and re.search(
        r"[\u4e00-\u9fff]", content
    ):
        issues.append("The body contains Chinese text but must be English.")

    if not content.strip():
        issues.append("The body is empty.")

    return issues


AUDIT_DEVELOPER_PROMPT = """
You are a strict factual and editorial auditor.

Compare the section body against its assigned facts and supporting excerpts.

Fail the body for substantive problems:
1. It contains an unsupported factual or promotional claim.
2. It changes a number, name, date, specification, price, condition,
   limitation, exclusion, or optional status.
3. It treats uncertain, conditional, or optional information as guaranteed.
4. It omits a major assigned fact needed for an information-complete section.
5. It contains excessive repetition, incoherent organization, or irrelevant
   material.
6. It drifts outside the stated section purpose.
7. Its persuasive language goes beyond what the evidence supports.

Do not fail it for harmless stylistic preferences. Return concrete,
actionable issues and the IDs of materially omitted assigned facts.
"""


def semantic_audit(
    client: OpenAI,
    section_name: str,
    section_purpose: str,
    content: str,
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    user_prompt = f"""
Section name:
{section_name}

Section purpose:
{section_purpose}

Assigned evidence:
{json.dumps(evidence, ensure_ascii=False, indent=2)}

Draft body:
{content}
"""

    return call_json(
        client=client,
        task_name=f"audit-{short_hash(section_name + content)}",
        developer_prompt=AUDIT_DEVELOPER_PROMPT,
        user_prompt=user_prompt,
        schema_name="generic_section_audit",
        schema=AUDIT_SCHEMA,
        temperature=0.0,
    )


def create_initial_draft(
    client: OpenAI,
    section: dict[str, Any],
    evidence: list[dict[str, Any]],
) -> str:
    cache_key = short_hash(
        PIPELINE_VERSION
        + MODEL
        + stable_json(section)
        + stable_json(evidence)
    )
    cache_path = DRAFT_DIR / f"draft_{cache_key}.json"

    if cache_path.exists() and not FORCE_REGENERATE:
        return str(read_json(cache_path)["content"]).strip()

    user_prompt = f"""
Article theme context is inferred from the evidence; do not add facts beyond it.

Section name:
{section["name"]}

Section purpose:
{section["purpose"]}

Assigned evidence:
{json.dumps(evidence, ensure_ascii=False, indent=2)}

Write the complete section body. Return it only in the schema-defined
"content" field.
"""

    result = call_json(
        client=client,
        task_name=f"draft-{short_hash(section['name'])}",
        developer_prompt=WRITING_DEVELOPER_PROMPT,
        user_prompt=user_prompt,
        schema_name="generic_section_draft",
        schema=CONTENT_SCHEMA,
        temperature=0.25,
    )

    content = str(result["content"]).strip()
    write_json(cache_path, {"content": content})
    return content


REPAIR_DEVELOPER_PROMPT = f"""
You are revising one {OUTPUT_LANGUAGE} article section under strict factual
and character-count constraints.

Rules:
1. Fix every listed issue.
2. Use only the assigned evidence.
3. Return the complete revised body, not commentary or a patch.
4. Do not invent facts merely to increase length.
5. Add useful explanation and organization only when directly grounded in
   the supplied facts.
6. Preserve every number, condition, limitation, optional status, and
   uncertainty.
7. Do not mention source files, IDs, validation, prompts, or character counts.
8. Do not include the section title.
9. Use coherent prose paragraphs.
10. The revised body must contain more than
    {MIN_FINAL_NON_WHITESPACE_CHARS} non-whitespace characters.
"""


def repair_draft(
    client: OpenAI,
    section: dict[str, Any],
    evidence: list[dict[str, Any]],
    content: str,
    issues: list[str],
    round_number: int,
) -> str:
    user_prompt = f"""
Section name:
{section["name"]}

Section purpose:
{section["purpose"]}

Assigned evidence:
{json.dumps(evidence, ensure_ascii=False, indent=2)}

Current body:
{content}

Issues to fix:
{json.dumps(issues, ensure_ascii=False, indent=2)}

Return the complete corrected body in the schema-defined "content" field.
"""

    result = call_json(
        client=client,
        task_name=(
            f"repair-{short_hash(section['name'])}-R{round_number}"
        ),
        developer_prompt=REPAIR_DEVELOPER_PROMPT,
        user_prompt=user_prompt,
        schema_name="generic_repaired_section",
        schema=CONTENT_SCHEMA,
        temperature=0.15,
    )
    return str(result["content"]).strip()


def generate_validated_section(
    client: OpenAI,
    section: dict[str, Any],
    fact_map: dict[str, dict[str, Any]],
    registry: dict[str, dict[str, Any]],
) -> str:
    evidence = build_section_evidence(section, fact_map, registry)
    content = create_initial_draft(client, section, evidence)

    for round_number in range(1, MAX_DRAFT_REPAIR_ROUNDS + 1):
        issues = hard_validate_draft(content, evidence)

        audit = semantic_audit(
            client=client,
            section_name=section["name"],
            section_purpose=section["purpose"],
            content=content,
            evidence=evidence,
        )

        if not audit.get("passed", False):
            issues.extend(
                normalize_space(str(issue))
                for issue in audit.get("issues", [])
                if normalize_space(str(issue))
            )

            omitted = [
                str(fact_id)
                for fact_id in audit.get("omitted_fact_ids", [])
            ]
            if omitted:
                issues.append(
                    "Materially omitted assigned fact_ids: "
                    + ", ".join(omitted)
                )

            if not issues:
                issues.append(
                    "The semantic audit failed without a specific issue."
                )

        issues = list(dict.fromkeys(issues))
        counts = count_characters(content)

        logger.info(
            '%s round %d: total=%d non-whitespace=%d issues=%d',
            section["name"],
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
            evidence=evidence,
            content=content,
            issues=issues,
            round_number=round_number,
        )

    final_issues = hard_validate_draft(content, evidence)
    final_audit = semantic_audit(
        client=client,
        section_name=section["name"],
        section_purpose=section["purpose"],
        content=content,
        evidence=evidence,
    )

    if final_issues or not final_audit.get("passed", False):
        raise RuntimeError(
            f'Section "{section["name"]}" failed final validation. '
            f"Hard issues: {final_issues}; "
            f"Audit issues: {final_audit.get('issues', [])}"
        )

    return content


# ============================================================
# 10. Export and final validation
# ============================================================

def validate_final_output(
    result: dict[str, str],
    taxonomy: dict[str, Any],
) -> dict[str, Any]:
    expected_names = [
        section["name"] for section in taxonomy["section_plan"]
    ]

    if list(result.keys()) != expected_names:
        raise RuntimeError(
            "task2.json keys or order do not match the inferred taxonomy."
        )

    if len(result) > MAX_SECTIONS:
        raise RuntimeError(
            f"task2.json contains more than {MAX_SECTIONS} sections."
        )

    report: dict[str, Any] = {
        "model": MODEL,
        "output_language": OUTPUT_LANGUAGE,
        "input_character_chunk_size": MAX_CHARS_PER_CHUNK,
        "minimum_required_non_whitespace_characters_per_section": (
            MIN_FINAL_NON_WHITESPACE_CHARS
        ),
        "section_count": len(result),
        "sections": {},
    }

    for name, content in result.items():
        counts = count_characters(content)
        passed = (
            counts["non_whitespace"]
            > MIN_FINAL_NON_WHITESPACE_CHARS
        )

        report["sections"][name] = {
            **counts,
            "passed": passed,
        }

        if not passed:
            raise RuntimeError(
                f'Section "{name}" does not exceed '
                f"{MIN_FINAL_NON_WHITESPACE_CHARS} non-whitespace characters."
            )

    return report


def export_markdown(
    result: dict[str, str],
    path: Path,
) -> None:
    blocks = [
        f"# {section_name}\n\n{body.strip()}"
        for section_name, body in result.items()
    ]
    write_text(path, "\n\n".join(blocks))


# ============================================================
# 11. One-shot main pipeline
# ============================================================

def main() -> None:
    ensure_directories()
    client = build_client()

    logger.warning("Stage 0/4: loading and chunking the corpus.")
    corpus, input_names = load_corpus()
    chunks = split_corpus(corpus)

    if not chunks:
        raise RuntimeError("The corpus produced no chunks.")

    save_chunks(chunks)

    logger.info(
        "Loaded %d input file(s), %d corpus characters, %d chunks.",
        len(input_names),
        len(corpus),
        len(chunks),
    )

    logger.warning("Stage 1/4: extracting verifiable facts from each chunk.")
    source_nodes: list[tuple[str, dict[str, Any]]] = []
    registry: dict[str, dict[str, Any]] = {}

    for index, chunk in enumerate(chunks, start=1):
        chunk_id = f"C{index:03d}"
        logger.info(
            "Extracting %s (%d characters)",
            chunk_id,
            len(chunk),
        )

        packet, chunk_registry = extract_chunk(
            client=client,
            chunk_id=chunk_id,
            chunk_text=chunk,
        )
        source_nodes.append((chunk_id, packet))
        registry.update(chunk_registry)

    logger.warning(
        "Stage 2/4: pairwise merge-sort consolidation and deduplication."
    )
    merged_packet = merge_sort_packets(
        client=client,
        source_nodes=source_nodes,
        registry=registry,
    )

    canonical_facts = canonicalize_facts(merged_packet)
    if not canonical_facts:
        raise RuntimeError("No canonical facts remain after consolidation.")

    evidence_output = {
        "input_files": input_names,
        "chunk_count": len(chunks),
        "atomic_evidence_registry": registry,
        "canonical_facts": canonical_facts,
        "theme_hints": merged_packet["theme_hints"],
    }
    write_json(OUTPUT_EVIDENCE, evidence_output)

    logger.warning(
        "Stage 3/4: inferring a source-driven taxonomy with N <= %d.",
        MAX_SECTIONS,
    )
    taxonomy = infer_taxonomy(
        client=client,
        canonical_facts=canonical_facts,
        theme_hints=merged_packet["theme_hints"],
    )
    write_json(OUTPUT_TAXONOMY, taxonomy)

    logger.warning(
        "Stage 4/4: writing, auditing, and validating final sections."
    )
    fact_map = {
        fact["fact_id"]: fact
        for fact in canonical_facts
    }

    final_result: dict[str, str] = {}

    for section in taxonomy["section_plan"]:
        logger.warning("Writing section: %s", section["name"])
        final_result[section["name"]] = generate_validated_section(
            client=client,
            section=section,
            fact_map=fact_map,
            registry=registry,
        )

    report = validate_final_output(final_result, taxonomy)

    write_json(OUTPUT_JSON, final_result)
    export_markdown(final_result, OUTPUT_MD)
    write_json(OUTPUT_REPORT, report)

    print()
    print("=" * 76)
    print("TASK 2 COMPLETED")
    print("=" * 76)
    print(f"Input files: {', '.join(input_names)}")
    print(f"Corpus chunks: {len(chunks)}")
    print(f"Inferred sections: {len(final_result)}")
    print(f"Final JSON: {OUTPUT_JSON.resolve()}")
    print(f"Readable Markdown: {OUTPUT_MD.resolve()}")
    print(f"Evidence trace: {OUTPUT_EVIDENCE.resolve()}")
    print(f"Taxonomy: {OUTPUT_TAXONOMY.resolve()}")
    print(f"Validation report: {OUTPUT_REPORT.resolve()}")
    print()

    for name, details in report["sections"].items():
        print(
            f"{name}: total={details['total']}, "
            f"non-whitespace={details['non_whitespace']}, "
            f"passed={details['passed']}"
        )


if __name__ == "__main__":
    main()
