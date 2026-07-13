from __future__ import annotations

import argparse
import json
import logging
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any

from openai import DefaultHttpxClient, OpenAI


# ============================================================
# 1. Configuration
# ============================================================

MODEL = "gpt-4.1-mini-2025-04-14"

# Default JSON file. You can also pass another path on the command line.
DEFAULT_JSON_PATH = Path("task2_cn.json")

# Local proxy. Set to None when no proxy is needed.
PROXY_URL: str | None = "http://127.0.0.1:1080"

MAX_API_ATTEMPTS = 4
MAX_ADJUST_ROUNDS = 5

# When a range is supplied, aim near its middle instead of barely touching
# the lower or upper boundary.
TARGET_POSITION_IN_RANGE = 0.55


# ============================================================
# 2. Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
# 3. Structured output schema
# ============================================================

CONTENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "content": {"type": "string"},
    },
    "required": ["content"],
    "additionalProperties": False,
}


# ============================================================
# 4. Chinese-character counting
# ============================================================

CHINESE_RE = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]"
)


def count_chinese_characters(text: str) -> int:
    """
    Count Chinese Han characters only.

    Spaces, punctuation, English letters, Arabic digits, and line breaks
    are excluded.
    """
    return len(CHINESE_RE.findall(text))


# ============================================================
# 5. Range input parsing
# ============================================================

def parse_length_range(raw: str) -> tuple[int, int] | None:
    """
    Supported input examples:

        1500-1800
        1500~1800
        1500,1800
        1500 1800

    Empty input means skip this key.
    """
    raw = raw.strip()

    if not raw:
        return None

    match = re.fullmatch(
        r"\s*(\d+)\s*(?:-|~|～|,|，|\s)\s*(\d+)\s*",
        raw,
    )

    if not match:
        raise ValueError(
            "请输入范围，例如 1500-1800；直接回车表示跳过。"
        )

    minimum = int(match.group(1))
    maximum = int(match.group(2))

    if minimum < 0 or maximum < 0:
        raise ValueError("字符数不能为负数。")

    if minimum > maximum:
        raise ValueError("最小字符数不能大于最大字符数。")

    if minimum == maximum:
        raise ValueError(
            "不建议要求精确字符数，请提供一个范围，例如 1450-1550。"
        )

    return minimum, maximum


def ask_ranges(
    data: dict[str, Any],
) -> dict[str, tuple[int, int]]:
    """
    Display every top-level key and ask the user for its required Chinese
    character range.
    """
    print()
    print("=" * 76)
    print("检测到以下 JSON 顶层 Key")
    print("字符数只统计中文汉字，不统计标点、空格、英文、数字和换行。")
    print("=" * 76)

    keys = list(data.keys())

    for index, key in enumerate(keys, start=1):
        value = data[key]

        if isinstance(value, str):
            count = count_chinese_characters(value)
            print(f"{index}. {key}（当前中文字符数：{count}）")
        else:
            print(
                f"{index}. {key}（Value 类型为 "
                f"{type(value).__name__}，脚本将跳过）"
            )

    print()
    print(
        "请逐个输入目标范围，例如 1500-1800。"
        "直接回车表示该 Key 不处理。"
    )

    requirements: dict[str, tuple[int, int]] = {}

    for key in keys:
        value = data[key]

        if not isinstance(value, str):
            continue

        current_count = count_chinese_characters(value)

        while True:
            raw = input(
                f'\nKey「{key}」当前 {current_count} 字，'
                "请输入目标范围："
            )

            try:
                parsed = parse_length_range(raw)
                break
            except ValueError as error:
                print(f"输入无效：{error}")

        if parsed is None:
            print(f"已跳过「{key}」。")
            continue

        requirements[key] = parsed

        minimum, maximum = parsed
        if minimum <= current_count <= maximum:
            print(
                f"「{key}」已经位于 {minimum}-{maximum} 范围内，"
                "后续不会调用模型。"
            )
        elif current_count < minimum:
            print(
                f"「{key}」需要扩写："
                f"{current_count} → {minimum}-{maximum}。"
            )
        else:
            print(
                f"「{key}」需要压缩："
                f"{current_count} → {minimum}-{maximum}。"
            )

    return requirements


# ============================================================
# 6. JSON file helpers
# ============================================================

def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise TypeError(
            "JSON 顶层必须是对象，例如："
            '{"部分名称": "对应正文"}'
        )

    return data


def overwrite_json_atomically(
    path: Path,
    data: dict[str, Any],
) -> None:
    """
    Atomically overwrite the original JSON.

    The temporary file is written first. os.replace then replaces the original
    file in one operation, reducing the chance of leaving a broken JSON file
    if the process stops during writing.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            suffix=".json",
            prefix=f".{path.stem}_",
            dir=path.parent,
            delete=False,
        ) as temporary_file:
            json.dump(
                data,
                temporary_file,
                ensure_ascii=False,
                indent=2,
            )
            temporary_file.write("\n")
            temporary_path = Path(temporary_file.name)

        os.replace(temporary_path, path)

    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink(missing_ok=True)


# ============================================================
# 7. OpenAI client and request helper
# ============================================================

def build_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "没有检测到 OPENAI_API_KEY。\n"
            "PowerShell 示例：\n"
            '$env:OPENAI_API_KEY="你的API密钥"'
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


def call_rewriter(
    client: OpenAI,
    developer_prompt: str,
    user_prompt: str,
    task_name: str,
) -> str:
    last_error: Exception | None = None

    for attempt in range(1, MAX_API_ATTEMPTS + 1):
        try:
            logger.info(
                "%s：API 请求 %d/%d",
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
                temperature=0.15,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "resized_chinese_content",
                        "strict": True,
                        "schema": CONTENT_SCHEMA,
                    },
                },
            )

            message = response.choices[0].message
            refusal = getattr(message, "refusal", None)

            if refusal:
                raise RuntimeError(f"模型拒绝回答：{refusal}")

            if not message.content:
                raise RuntimeError("模型返回空内容。")

            result = json.loads(message.content)
            content = str(result["content"]).strip()

            if not content:
                raise RuntimeError("模型返回的 content 为空。")

            if response.usage:
                logger.info(
                    "%s tokens：input=%s output=%s total=%s",
                    task_name,
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                    response.usage.total_tokens,
                )

            return content

        except Exception as error:
            last_error = error
            logger.warning("%s 请求失败：%s", task_name, error)

            if attempt < MAX_API_ATTEMPTS:
                time.sleep(attempt * 5)

    raise RuntimeError(
        f"{task_name} 连续请求失败：{last_error}"
    )


# ============================================================
# 8. Length adjustment
# ============================================================

REWRITE_DEVELOPER_PROMPT = """
你是一名严谨的中文长文编辑。你的任务是调整一段已有中文正文的长度，
使其落入用户给定的中文字符范围。

“中文字符数”只统计中文汉字，不统计标点、空格、英文、阿拉伯数字和换行。

必须遵守：

1. 只能基于原文已有信息进行改写。
2. 不得添加原文没有的事实、数据、价格、日期、条件、功能、结论或承诺。
3. 不得改变原文中的事实含义、数字、专有名词、限制条件和不确定性。
4. 扩写时，可以：
   - 重组原有信息；
   - 更清楚地解释原文已有观点之间的关系；
   - 补充自然的过渡、背景说明和阅读引导；
   - 将原文已有要点表达得更完整。
   但不得虚构新事实，也不得机械重复同一句话。
5. 压缩时，应删除重复表达、空泛修饰和冗余过渡，但保留所有重要事实。
6. 正文必须使用自然、通顺、结构合理的简体中文。
7. 不要输出标题，除非标题本来就是正文不可分割的一部分。
8. 不要提及字符数、模型、Prompt、JSON、修改过程或校验过程。
9. 只返回修改后的完整正文。
"""


def choose_target(
    minimum: int,
    maximum: int,
) -> int:
    width = maximum - minimum
    return minimum + round(width * TARGET_POSITION_IN_RANGE)


def adjust_one_value(
    client: OpenAI,
    key: str,
    original_text: str,
    minimum: int,
    maximum: int,
) -> str:
    current_text = original_text.strip()
    original_count = count_chinese_characters(current_text)

    if minimum <= original_count <= maximum:
        return original_text

    target = choose_target(minimum, maximum)

    for round_number in range(1, MAX_ADJUST_ROUNDS + 1):
        current_count = count_chinese_characters(current_text)

        if minimum <= current_count <= maximum:
            return current_text

        if current_count < minimum:
            operation = "扩写"
            direction_instruction = (
                "当前正文过短。请在不新增事实的前提下，"
                "把原有信息表达得更完整、更连贯。"
            )
        else:
            operation = "压缩"
            direction_instruction = (
                "当前正文过长。请删除重复、冗余和空泛表达，"
                "但保留全部重要事实。"
            )

        user_prompt = f"""
JSON Key：
{key}

操作：
{operation}

允许范围：
{minimum}-{maximum} 个中文汉字

建议目标：
约 {target} 个中文汉字

当前正文中文字符数：
{current_count}

要求：
{direction_instruction}

请注意，程序会重新统计中文汉字数量。不要刚好贴近边界，
尽量靠近建议目标。

当前正文：
<content>
{current_text}
</content>
"""

        rewritten = call_rewriter(
            client=client,
            developer_prompt=REWRITE_DEVELOPER_PROMPT,
            user_prompt=user_prompt,
            task_name=f"{key}-第{round_number}轮{operation}",
        )

        rewritten_count = count_chinese_characters(rewritten)

        print(
            f"「{key}」第 {round_number} 轮："
            f"{current_count} → {rewritten_count}"
        )

        current_text = rewritten

    final_count = count_chinese_characters(current_text)

    if not (minimum <= final_count <= maximum):
        raise RuntimeError(
            f"「{key}」经过 {MAX_ADJUST_ROUNDS} 轮后仍未达标："
            f"{final_count}，要求 {minimum}-{maximum}。"
            "原 JSON 尚未被覆盖。"
        )

    return current_text


# ============================================================
# 9. Main
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "逐个读取 JSON 顶层 Key，由用户输入中文字符范围，"
            "自动扩写或压缩不达标的正文，并覆盖原 JSON。"
        )
    )

    parser.add_argument(
        "json_path",
        nargs="?",
        type=Path,
        default=DEFAULT_JSON_PATH,
        help="要直接覆盖的 JSON 路径，默认 task2_cn.json",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    json_path: Path = args.json_path

    if not json_path.is_file():
        raise FileNotFoundError(
            f"JSON 文件不存在：{json_path}"
        )

    data = read_json(json_path)
    requirements = ask_ranges(data)

    if not requirements:
        print("\n没有 Key 需要处理，文件未修改。")
        return

    pending_keys = [
        key
        for key, (minimum, maximum) in requirements.items()
        if not (
            minimum
            <= count_chinese_characters(str(data[key]))
            <= maximum
        )
    ]

    if not pending_keys:
        print("\n所有已设置范围的正文都已达标，文件未修改。")
        return

    print()
    print("=" * 76)
    print("开始调用大模型调整以下 Key：")
    for key in pending_keys:
        minimum, maximum = requirements[key]
        current = count_chinese_characters(str(data[key]))
        print(f"- {key}：当前 {current}，目标 {minimum}-{maximum}")
    print("=" * 76)

    client = build_client()
    updated_data = dict(data)

    # Only overwrite the original file after every requested adjustment
    # succeeds. A failed run leaves the original JSON untouched.
    for key in pending_keys:
        minimum, maximum = requirements[key]

        updated_data[key] = adjust_one_value(
            client=client,
            key=key,
            original_text=str(updated_data[key]),
            minimum=minimum,
            maximum=maximum,
        )

    # Final local validation before overwriting.
    failures: list[str] = []

    for key, (minimum, maximum) in requirements.items():
        if not isinstance(updated_data.get(key), str):
            continue

        count = count_chinese_characters(updated_data[key])

        if not (minimum <= count <= maximum):
            failures.append(
                f"{key}: {count}，要求 {minimum}-{maximum}"
            )

    if failures:
        raise RuntimeError(
            "最终检查失败，原文件未覆盖：\n"
            + "\n".join(failures)
        )

    overwrite_json_atomically(json_path, updated_data)

    print()
    print("=" * 76)
    print("处理完成，已直接覆盖原 JSON")
    print("=" * 76)
    print(f"文件：{json_path.resolve()}")

    for key, (minimum, maximum) in requirements.items():
        value = updated_data.get(key)

        if isinstance(value, str):
            count = count_chinese_characters(value)
            status = "达标" if minimum <= count <= maximum else "未达标"
            print(
                f"- {key}：{count} 字，"
                f"要求 {minimum}-{maximum}，{status}"
            )


if __name__ == "__main__":
    main()
