from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

from openai import DefaultHttpxClient, OpenAI


# ============================================================
# 1. Default configuration
# ============================================================

MODEL = "gpt-4.1-mini-2025-04-14"

# Default input/output paths. They can also be supplied on the command line.
DEFAULT_INPUT_JSON = Path("task2.json")
DEFAULT_OUTPUT_JSON = Path("task2_cn.json")

# Cache successful translations so rerunning does not pay for the same text.
DEFAULT_CACHE_JSON = Path("task2_translation_cache.json")
DEFAULT_COUNT_REPORT_JSON = Path("task2_cn_character_report.json")

# Task 2 section names are JSON keys, so keys should normally be translated too.
TRANSLATE_KEYS = True

# Maximum total source characters included in one API request.
# This controls batching; it is not the model context limit.
MAX_BATCH_SOURCE_CHARS = 12000

# Maximum number of separate strings in one request.
MAX_BATCH_ITEMS = 20

MAX_API_ATTEMPTS = 4

# Local proxy. Set to None when no proxy is needed.
PROXY_URL: str | None = "http://127.0.0.1:1080"


# ============================================================
# 2. Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
# 3. Structured-output schema
# ============================================================

TRANSLATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "translations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "translation": {"type": "string"},
                },
                "required": ["id", "translation"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["translations"],
    "additionalProperties": False,
}


# ============================================================
# 4. Text detection and validation
# ============================================================

# Covers common CJK Unified Ideographs and extensions used in Chinese text.
CHINESE_RE = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]"
)

NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
URL_RE = re.compile(r"https?://[^\s)\]}>\"']+", re.IGNORECASE)
EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)


def contains_chinese(text: str) -> bool:
    """
    Follow the user's exact rule:
    if a string contains at least one Chinese character, leave the entire
    string unchanged.
    """
    return bool(CHINESE_RE.search(text))


def count_chinese_characters(text: str) -> int:
    """
    Count Chinese Han characters only.

    Spaces, punctuation, English letters, Arabic digits, and line breaks
    are not counted.
    """
    return len(CHINESE_RE.findall(text))


def count_chinese_in_json_values(value: Any) -> int:
    """
    Recursively count Chinese characters in JSON string VALUES only.

    JSON object keys are deliberately excluded because the task's length
    requirement normally applies to article bodies rather than section names.
    """
    if isinstance(value, dict):
        return sum(
            count_chinese_in_json_values(child)
            for child in value.values()
        )

    if isinstance(value, list):
        return sum(
            count_chinese_in_json_values(child)
            for child in value
        )

    if isinstance(value, str):
        return count_chinese_characters(value)

    return 0


def count_chinese_in_json_keys(value: Any) -> int:
    """Recursively count Chinese characters appearing in JSON object keys."""
    if isinstance(value, dict):
        return sum(
            count_chinese_characters(str(key))
            + count_chinese_in_json_keys(child)
            for key, child in value.items()
        )

    if isinstance(value, list):
        return sum(
            count_chinese_in_json_keys(child)
            for child in value
        )

    return 0


def build_chinese_character_report(value: Any) -> dict[str, Any]:
    """
    Build a report for the translated JSON.

    For a top-level object, each top-level field receives a separate count.
    Nested structures are counted recursively.
    """
    report: dict[str, Any] = {
        "counting_rule": (
            "Only Chinese Han characters are counted. Spaces, punctuation, "
            "English letters, digits, and line breaks are excluded. "
            "Per-section counts include JSON values only, not keys."
        ),
        "total_chinese_characters_in_values": (
            count_chinese_in_json_values(value)
        ),
        "total_chinese_characters_in_keys": (
            count_chinese_in_json_keys(value)
        ),
        "sections": {},
    }

    if isinstance(value, dict):
        for key, child in value.items():
            report["sections"][str(key)] = {
                "chinese_characters_in_value": (
                    count_chinese_in_json_values(child)
                )
            }
    else:
        report["sections"]["$"] = {
            "chinese_characters_in_value": (
                count_chinese_in_json_values(value)
            )
        }

    return report


def should_translate(text: str) -> bool:
    return bool(text.strip()) and not contains_chinese(text)


def extract_numbers(text: str) -> set[str]:
    return {
        value.replace(",", "").rstrip(".")
        for value in NUMBER_RE.findall(text)
    }


def extract_urls(text: str) -> set[str]:
    return set(URL_RE.findall(text))


def extract_emails(text: str) -> set[str]:
    return set(EMAIL_RE.findall(text))


def translation_validation_issues(
    source: str,
    translated: str,
) -> list[str]:
    issues: list[str] = []

    if not translated.strip():
        issues.append("The translation is empty.")

    # Numbers, URLs, and email addresses must not disappear or change.
    missing_numbers = sorted(
        extract_numbers(source) - extract_numbers(translated)
    )
    if missing_numbers:
        issues.append(
            "Missing or changed numeric strings: "
            + ", ".join(missing_numbers)
        )

    missing_urls = sorted(extract_urls(source) - extract_urls(translated))
    if missing_urls:
        issues.append(
            "Missing or changed URLs: " + ", ".join(missing_urls)
        )

    missing_emails = sorted(
        extract_emails(source) - extract_emails(translated)
    )
    if missing_emails:
        issues.append(
            "Missing or changed email addresses: "
            + ", ".join(missing_emails)
        )

    # Natural-language English should normally become Chinese.
    # Very short codes/acronyms may legitimately remain unchanged.
    has_letters = bool(re.search(r"[A-Za-z]", source))
    looks_like_code_or_identifier = bool(
        re.fullmatch(r"[\w./:@+-]{1,30}", source.strip())
    )

    if (
        has_letters
        and not looks_like_code_or_identifier
        and not contains_chinese(translated)
    ):
        issues.append(
            "The result still contains no Chinese characters."
        )

    return issues


# ============================================================
# 5. JSON traversal
# ============================================================

def collect_translatable_strings(
    value: Any,
    translate_keys: bool,
) -> list[str]:
    """
    Recursively collect unique string keys/values that contain no Chinese.
    """
    ordered: list[str] = []
    seen: set[str] = set()

    def add(text: str) -> None:
        if should_translate(text) and text not in seen:
            ordered.append(text)
            seen.add(text)

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, child in node.items():
                if translate_keys and isinstance(key, str):
                    add(key)
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)
        elif isinstance(node, str):
            add(node)

    walk(value)
    return ordered


def apply_translations(
    value: Any,
    translations: dict[str, str],
    translate_keys: bool,
) -> Any:
    """
    Recursively replace translated strings. Non-string JSON values are
    preserved exactly.
    """
    if isinstance(value, dict):
        result: dict[str, Any] = {}

        for original_key, child in value.items():
            if (
                translate_keys
                and isinstance(original_key, str)
                and should_translate(original_key)
            ):
                new_key = translations[original_key]
            else:
                new_key = original_key

            if new_key in result:
                raise RuntimeError(
                    "Two JSON keys became identical after translation: "
                    f"{new_key!r}. Please review the source keys."
                )

            result[new_key] = apply_translations(
                child,
                translations,
                translate_keys,
            )

        return result

    if isinstance(value, list):
        return [
            apply_translations(item, translations, translate_keys)
            for item in value
        ]

    if isinstance(value, str) and should_translate(value):
        return translations[value]

    return value


# ============================================================
# 6. File helpers
# ============================================================

def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(
            value,
            file,
            ensure_ascii=False,
            indent=2,
        )
        file.write("\n")


def load_cache(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    raw = read_json(path)
    if not isinstance(raw, dict):
        raise RuntimeError(
            f"Translation cache must be a JSON object: {path}"
        )

    return {
        str(source): str(translation)
        for source, translation in raw.items()
    }


# ============================================================
# 7. OpenAI client and API
# ============================================================

def build_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set.\n"
            'PowerShell example:\n'
            '$env:OPENAI_API_KEY="your-api-key"'
        )

    arguments: dict[str, Any] = {
        "api_key": api_key,
        "timeout": 240.0,
        "max_retries": 1,
    }

    if PROXY_URL:
        logger.info("Using proxy: %s", PROXY_URL)
        arguments["http_client"] = DefaultHttpxClient(
            proxy=PROXY_URL
        )
    else:
        logger.info("Using a direct API connection.")

    return OpenAI(**arguments)


TRANSLATION_DEVELOPER_PROMPT = """
你是一名严谨的中英文翻译编辑。你的任务是把输入的非中文字符串翻译为自然、准确、完整的简体中文。

必须遵守：

1. 只做翻译，不总结、不删减、不扩写、不补充外部信息。
2. 保留原文事实、语气、逻辑关系、条件、限制和不确定性。
3. 数字、日期、价格、百分比、单位、专有名词和缩写必须准确。
4. URL、电子邮箱、文件路径、代码、JSON 片段、Markdown 标记和占位符必须原样保留。
5. 标题要翻译成适合作为中文文章小标题的表达。
6. 正文要翻译成自然、通顺、专业的中文，不要保留不必要的英文句式。
7. 常见缩写可以保留，并在有必要时给出中文含义。
8. 不要输出解释、备注或翻译过程。
9. 每个输入 id 必须且只能返回一次，不得改变 id。
"""


def call_translation_batch(
    client: OpenAI,
    batch: list[tuple[str, str]],
) -> dict[str, str]:
    """
    Translate one batch. The batch contains (id, source_text) pairs.
    """
    expected_ids = {item_id for item_id, _ in batch}
    feedback = ""
    last_error: Exception | None = None

    for attempt in range(1, MAX_API_ATTEMPTS + 1):
        try:
            payload = [
                {
                    "id": item_id,
                    "text": source_text,
                }
                for item_id, source_text in batch
            ]

            user_prompt = f"""
请把下面各项翻译为简体中文。

输入：
{json.dumps(payload, ensure_ascii=False, indent=2)}

{feedback}

只返回符合既定 JSON Schema 的结果。
"""

            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "developer",
                        "content": TRANSLATION_DEVELOPER_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                temperature=0.0,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "batch_chinese_translation",
                        "strict": True,
                        "schema": TRANSLATION_SCHEMA,
                    },
                },
            )

            message = response.choices[0].message
            refusal = getattr(message, "refusal", None)
            if refusal:
                raise RuntimeError(f"Model refusal: {refusal}")

            if not message.content:
                raise RuntimeError("The model returned empty content.")

            parsed = json.loads(message.content)
            items = parsed["translations"]

            returned_ids = [
                str(item["id"])
                for item in items
            ]
            returned_id_set = set(returned_ids)

            issues: list[str] = []

            if returned_id_set != expected_ids:
                missing = sorted(expected_ids - returned_id_set)
                unknown = sorted(returned_id_set - expected_ids)

                if missing:
                    issues.append(
                        "Missing IDs: " + ", ".join(missing)
                    )
                if unknown:
                    issues.append(
                        "Unknown IDs: " + ", ".join(unknown)
                    )

            if len(returned_ids) != len(returned_id_set):
                issues.append("One or more IDs were returned more than once.")

            source_by_id = dict(batch)
            translated_by_id = {
                str(item["id"]): str(item["translation"])
                for item in items
                if str(item["id"]) in expected_ids
            }

            for item_id in expected_ids:
                if item_id not in translated_by_id:
                    continue

                item_issues = translation_validation_issues(
                    source_by_id[item_id],
                    translated_by_id[item_id],
                )
                for issue in item_issues:
                    issues.append(f"{item_id}: {issue}")

            if issues:
                feedback = (
                    "上一次输出未通过程序检查，请重新翻译全部项目并修复：\n- "
                    + "\n- ".join(issues)
                )
                logger.warning(
                    "Batch validation failed on attempt %d: %s",
                    attempt,
                    "; ".join(issues),
                )
                continue

            if response.usage:
                logger.info(
                    "Translation tokens: input=%s output=%s total=%s",
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                    response.usage.total_tokens,
                )

            return translated_by_id

        except Exception as error:
            last_error = error
            logger.warning(
                "Translation request attempt %d/%d failed: %s",
                attempt,
                MAX_API_ATTEMPTS,
                error,
            )

            if attempt < MAX_API_ATTEMPTS:
                time.sleep(attempt * 5)

    raise RuntimeError(
        "Translation batch failed after "
        f"{MAX_API_ATTEMPTS} attempts: {last_error}"
    )


# ============================================================
# 8. Batching
# ============================================================

def make_batches(
    texts: list[str],
) -> list[list[str]]:
    batches: list[list[str]] = []
    current: list[str] = []
    current_chars = 0

    for text in texts:
        text_chars = len(text)

        if (
            current
            and (
                len(current) >= MAX_BATCH_ITEMS
                or current_chars + text_chars > MAX_BATCH_SOURCE_CHARS
            )
        ):
            batches.append(current)
            current = []
            current_chars = 0

        current.append(text)
        current_chars += text_chars

    if current:
        batches.append(current)

    return batches


def translate_all_strings(
    client: OpenAI,
    strings: list[str],
    cache_path: Path,
) -> dict[str, str]:
    cache = load_cache(cache_path)

    # Revalidate cached entries. Invalid old cache entries are regenerated.
    valid_cache: dict[str, str] = {}
    for source, translated in cache.items():
        if (
            should_translate(source)
            and not translation_validation_issues(source, translated)
        ):
            valid_cache[source] = translated

    pending = [
        text
        for text in strings
        if text not in valid_cache
    ]

    logger.info(
        "Translatable unique strings: %d; cached: %d; pending: %d",
        len(strings),
        len(strings) - len(pending),
        len(pending),
    )

    batches = make_batches(pending)

    for batch_index, batch_texts in enumerate(batches, start=1):
        logger.info(
            "Translating batch %d/%d: %d strings, %d source chars",
            batch_index,
            len(batches),
            len(batch_texts),
            sum(len(text) for text in batch_texts),
        )

        batch_pairs = [
            (f"T{index:04d}", text)
            for index, text in enumerate(batch_texts, start=1)
        ]

        translated_by_id = call_translation_batch(
            client,
            batch_pairs,
        )

        for item_id, source_text in batch_pairs:
            valid_cache[source_text] = translated_by_id[item_id]

        # Save after every batch for interruption-safe reruns.
        write_json(cache_path, valid_cache)

    return {
        text: valid_cache[text]
        for text in strings
    }


# ============================================================
# 9. Final output validation
# ============================================================

def find_untranslated_strings(
    value: Any,
    path: str = "$",
) -> list[str]:
    """
    Find remaining non-Chinese strings after translation. Very short
    identifiers, URLs, emails, and pure numbers are allowed.
    """
    issues: list[str] = []

    def is_allowed_non_chinese(text: str) -> bool:
        stripped = text.strip()

        if not stripped:
            return True
        if contains_chinese(stripped):
            return True
        if URL_RE.fullmatch(stripped):
            return True
        if EMAIL_RE.fullmatch(stripped):
            return True
        if re.fullmatch(r"[\d\s.,%$£€¥:/+\-]+", stripped):
            return True
        if re.fullmatch(r"[\w./:@+\-]{1,20}", stripped):
            return True

        return False

    if isinstance(value, dict):
        for key, child in value.items():
            if isinstance(key, str) and not is_allowed_non_chinese(key):
                issues.append(f"{path} key remains non-Chinese: {key!r}")
            issues.extend(
                find_untranslated_strings(
                    child,
                    f"{path}.{key}",
                )
            )

    elif isinstance(value, list):
        for index, child in enumerate(value):
            issues.extend(
                find_untranslated_strings(
                    child,
                    f"{path}[{index}]",
                )
            )

    elif isinstance(value, str) and not is_allowed_non_chinese(value):
        preview = value[:120].replace("\n", "\\n")
        issues.append(
            f"{path} value remains non-Chinese: {preview!r}"
        )

    return issues


# ============================================================
# 10. Main
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively translate non-Chinese JSON keys and values "
            "into Simplified Chinese."
        )
    )

    parser.add_argument(
        "input_json",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT_JSON,
        help="Input JSON path; default: task2.json",
    )
    parser.add_argument(
        "output_json",
        nargs="?",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help="Output JSON path; default: task2_cn.json",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=DEFAULT_CACHE_JSON,
        help=(
            "Translation cache path; "
            "default: task2_translation_cache.json"
        ),
    )
    parser.add_argument(
        "--count-report",
        type=Path,
        default=DEFAULT_COUNT_REPORT_JSON,
        help=(
            "Chinese character count report path; "
            "default: task2_cn_character_report.json"
        ),
    )
    parser.add_argument(
        "--values-only",
        action="store_true",
        help="Translate string values but leave JSON keys unchanged.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path: Path = args.input_json
    output_path: Path = args.output_json
    cache_path: Path = args.cache
    count_report_path: Path = args.count_report
    translate_keys = TRANSLATE_KEYS and not args.values_only

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Input JSON does not exist: {input_path}"
        )

    source_json = read_json(input_path)

    strings = collect_translatable_strings(
        source_json,
        translate_keys=translate_keys,
    )

    if not strings:
        logger.info(
            "No non-Chinese strings were found. Copying JSON unchanged."
        )
        write_json(output_path, source_json)

        character_report = build_chinese_character_report(source_json)
        write_json(count_report_path, character_report)

        print(
            "Chinese characters in JSON values:",
            character_report["total_chinese_characters_in_values"],
        )
        print("Count report:", count_report_path.resolve())
        return

    client = build_client()

    translations = translate_all_strings(
        client=client,
        strings=strings,
        cache_path=cache_path,
    )

    translated_json = apply_translations(
        source_json,
        translations=translations,
        translate_keys=translate_keys,
    )

    remaining = find_untranslated_strings(translated_json)

    write_json(output_path, translated_json)

    character_report = build_chinese_character_report(translated_json)
    write_json(count_report_path, character_report)

    print()
    print("=" * 72)
    print("JSON TRANSLATION COMPLETED")
    print("=" * 72)
    print(f"Input:  {input_path.resolve()}")
    print(f"Output: {output_path.resolve()}")
    print(f"Cache:  {cache_path.resolve()}")
    print(f"Count report: {count_report_path.resolve()}")
    print(f"Translated unique strings: {len(strings)}")
    print(
        "Total Chinese characters in JSON values:",
        character_report["total_chinese_characters_in_values"],
    )
    print(
        "Total Chinese characters in JSON keys:",
        character_report["total_chinese_characters_in_keys"],
    )

    print()
    print("Chinese character count by top-level field:")
    for section_name, details in character_report["sections"].items():
        print(
            f"- {section_name}: "
            f"{details['chinese_characters_in_value']}"
        )

    if remaining:
        print()
        print(
            "Warning: the following non-Chinese-looking strings remain. "
            "They may be proper nouns, acronyms, code, or items that require "
            "manual review:"
        )
        for issue in remaining[:30]:
            print("-", issue)

        if len(remaining) > 30:
            print(f"... and {len(remaining) - 30} more.")
    else:
        print("Final check: no unexpected English prose remains.")


if __name__ == "__main__":
    main()
