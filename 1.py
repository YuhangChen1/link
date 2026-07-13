from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from pathlib import Path

from openai import DefaultHttpxClient, OpenAI


# ============================================================
# 1. 基本配置
# ============================================================

# 面试要求使用的模型
MODEL = "gpt-4.1-mini-2025-04-14"

# 原始文件所在目录
# 当前设置表示与 main.py 在同一目录
INPUT_DIR = Path(".")

# 英文摘要输出目录
OUTPUT_DIR = Path(".")

# 分块英文摘要缓存目录
CHUNK_CACHE_DIR = Path("summary_cache_en")

# 多轮合并摘要缓存目录
MERGE_CACHE_DIR = Path("merge_cache_en")

# 要处理的文件编号
PART_NUMBERS = range(1, 9)

# 单个原始文本块最大字符数
# 这不是模型上下文极限，而是为了提高摘要稳定性
MAX_CHARS_PER_CHUNK = 18000

# 合并摘要时，单次请求输入的大致最大字符数
MAX_CHARS_PER_MERGE = 30000

# 如果遇到一个极长的单行或段落，强制切分时保留的重叠字符
OVERLAP_CHARS = 300

# API 请求失败后的最大尝试次数
MAX_API_ATTEMPTS = 4

# 是否强制重新生成所有最终英文摘要
#
# False：
#   如果 1_en.txt 已经存在，就跳过该文件。
#
# True：
#   即使已经存在，也重新生成。
FORCE_REGENERATE = False

# 是否在终端打印最终完整英文摘要
PRINT_FINAL_SUMMARY = True

# 代理地址
#
# 使用本机代理时保留：
PROXY_URL: str | None = "http://127.0.0.1:1080"

# 不使用代理时改为：
# PROXY_URL = None


# ============================================================
# 2. 日志配置
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# 3. 创建 OpenAI 客户端
# ============================================================

def build_openai_client() -> OpenAI:
    """
    创建 OpenAI 客户端。

    API Key 默认从环境变量 OPENAI_API_KEY 中读取。
    """

    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "没有找到环境变量 OPENAI_API_KEY。\n"
            "请先在 PowerShell 中运行：\n"
            '$env:OPENAI_API_KEY="你的API密钥"'
        )

    if PROXY_URL:
        logger.info("使用代理：%s", PROXY_URL)

        return OpenAI(
            api_key=api_key,
            timeout=180.0,
            max_retries=2,
            http_client=DefaultHttpxClient(
                proxy=PROXY_URL,
            ),
        )

    logger.info("不使用代理，直接连接 OpenAI API")

    return OpenAI(
        api_key=api_key,
        timeout=180.0,
        max_retries=2,
    )


client = build_openai_client()


# ============================================================
# 4. 文件读写函数
# ============================================================

def read_text(file_path: Path) -> str:
    """
    读取文本文件。

    优先尝试 UTF-8，也兼容部分 Windows 中文编码文件。
    """

    encodings = [
        "utf-8",
        "utf-8-sig",
        "gb18030",
    ]

    last_error: Exception | None = None

    for encoding in encodings:
        try:
            with file_path.open(
                "r",
                encoding=encoding,
            ) as file:
                return file.read()

        except UnicodeDecodeError as error:
            last_error = error

    raise RuntimeError(
        f"无法读取文件 {file_path}，请确认文本编码。"
    ) from last_error


def write_text(
    file_path: Path,
    content: str,
) -> None:
    """使用 UTF-8 编码保存文本。"""

    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with file_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:
        file.write(content.strip())
        file.write("\n")


def find_input_file(
    part_number: int,
) -> Path | None:
    """
    查找输入文件。

    同时支持：

    1.txt
    1

    两种命名方式。
    """

    candidates = [
        INPUT_DIR / f"{part_number}.txt",
        INPUT_DIR / str(part_number),
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    return None


def text_hash(text: str) -> str:
    """为文本生成短哈希，用于区分不同版本的缓存。"""

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()[:16]


# ============================================================
# 5. OpenAI 请求函数
# ============================================================

def call_gpt(
    developer_prompt: str,
    user_prompt: str,
    task_name: str,
) -> str:
    """
    调用 GPT-4.1 mini。

    包含：
    - 自动重试；
    - token 数量打印；
    - 空结果检查。
    """

    last_error: Exception | None = None

    for attempt in range(
        1,
        MAX_API_ATTEMPTS + 1,
    ):
        try:
            logger.info(
                "%s：发送 API 请求，第 %d/%d 次尝试",
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
                temperature=0.1,
            )

            answer = response.choices[0].message.content

            if answer is None or not answer.strip():
                raise RuntimeError("模型返回了空内容")

            if response.usage is not None:
                print(
                    f"[{task_name}] "
                    f"输入 tokens："
                    f"{response.usage.prompt_tokens}"
                )

                print(
                    f"[{task_name}] "
                    f"输出 tokens："
                    f"{response.usage.completion_tokens}"
                )

                print(
                    f"[{task_name}] "
                    f"总 tokens："
                    f"{response.usage.total_tokens}"
                )

            return answer.strip()

        except Exception as error:
            last_error = error

            logger.warning(
                "%s：第 %d 次请求失败：%s",
                task_name,
                attempt,
                error,
            )

            if attempt < MAX_API_ATTEMPTS:
                wait_seconds = attempt * 5

                logger.info(
                    "%d 秒后重新请求",
                    wait_seconds,
                )

                time.sleep(wait_seconds)

    raise RuntimeError(
        f"{task_name} 连续请求失败：{last_error}"
    )


# ============================================================
# 6. 长文本切分
# ============================================================

def hard_split_long_text(
    text: str,
    max_chars: int,
    overlap_chars: int = OVERLAP_CHARS,
) -> list[str]:
    """
    对无法按段落正常切开的超长文本执行字符切分。

    相邻块之间保留少量重叠，避免边界信息完全丢失。
    """

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = min(
            start + max_chars,
            len(text),
        )

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(
            end - overlap_chars,
            start + 1,
        )

    return chunks


def split_text(
    text: str,
    max_chars: int = MAX_CHARS_PER_CHUNK,
) -> list[str]:
    """
    优先按照段落切分长文本。

    处理顺序：

    1. 如果全文不超过限制，直接返回；
    2. 优先根据空行分段；
    3. 如果没有空行，则根据每一行分段；
    4. 单个段落仍然过长时，按字符强制切分。
    """

    text = text.strip()

    if not text:
        return []

    if len(text) <= max_chars:
        return [text]

    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(
            r"\n\s*\n",
            text,
        )
        if paragraph.strip()
    ]

    # 网页抓取内容经常没有空行，每一行就是一个信息单元
    if len(paragraphs) <= 1:
        paragraphs = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

    chunks: list[str] = []
    current_paragraphs: list[str] = []
    current_length = 0

    for paragraph in paragraphs:
        # 单个段落本身已经超过最大长度
        if len(paragraph) > max_chars:
            if current_paragraphs:
                chunks.append(
                    "\n\n".join(
                        current_paragraphs
                    )
                )

                current_paragraphs = []
                current_length = 0

            long_chunks = hard_split_long_text(
                paragraph,
                max_chars=max_chars,
            )

            chunks.extend(long_chunks)
            continue

        separator_length = 2 if current_paragraphs else 0
        candidate_length = (
            current_length
            + separator_length
            + len(paragraph)
        )

        if (
            current_paragraphs
            and candidate_length > max_chars
        ):
            chunks.append(
                "\n\n".join(
                    current_paragraphs
                )
            )

            current_paragraphs = [paragraph]
            current_length = len(paragraph)

        else:
            current_paragraphs.append(paragraph)
            current_length = candidate_length

    if current_paragraphs:
        chunks.append(
            "\n\n".join(
                current_paragraphs
            )
        )

    return [
        chunk.strip()
        for chunk in chunks
        if chunk.strip()
    ]


# ============================================================
# 7. 英文分块摘要 Prompt
# ============================================================

SUMMARY_DEVELOPER_PROMPT = """
You are a rigorous information analyst specializing in international
summer school programs.

Your current task is to produce a detailed factual English summary of the
provided source material. You are not writing the final promotional article.

Follow every requirement below:

1. Use only information explicitly stated in the source.
2. Do not use external knowledge.
3. Do not infer, speculate, or fill in missing details.
4. Do not invent courses, dates, prices, age requirements, facilities,
   accommodation arrangements, meal arrangements, application requirements,
   or program benefits.
5. Preserve important details such as:
   - program names;
   - school names;
   - locations;
   - dates;
   - duration;
   - age limits;
   - prices and additional charges;
   - minimum or maximum requirements;
   - class sizes;
   - teaching hours;
   - course names;
   - activity names;
   - accommodation types;
   - meal arrangements;
   - application conditions.
6. Preserve limiting language such as:
   "up to", "maximum", "minimum", "may", "optional", "usually",
   "subject to availability", and similar qualifications.
7. Clearly distinguish compulsory items from optional items.
8. Combine similar statements when appropriate, but do not remove unique
   facts merely to make the summary shorter.
9. If the source contains inconsistent or conflicting facts, retain both
   versions and identify the inconsistency. Do not decide which one is true.
10. Write entirely in English.
11. Use clear Markdown headings and bullet points.
12. Do not include promotional slogans, exaggerated claims, or unsupported
    recommendations.
13. Do not describe the source as complete if it is only one chunk of a
    larger document.

Use the following headings when relevant:

1. Program Overview
2. School and Location
3. Target Students and Age Requirements
4. Dates and Program Duration
5. Academic Courses and Teaching Arrangements
6. Non-Academic Activities, Sports, and Excursions
7. Accommodation
8. Meals and Catering
9. Fees and Additional Charges
10. Application, Booking, and Other Requirements
11. Missing, Unclear, or Conflicting Information

Omit a heading only when the current source contains no relevant information.
"""


def summarize_chunk(
    part_number: int,
    chunk_number: int,
    total_chunks: int,
    chunk_text: str,
) -> str:
    """为一个原始文本块生成详细英文摘要。"""

    user_prompt = f"""
The following material is chunk {chunk_number} of {total_chunks}
from source document {part_number}.

Produce a detailed, accurate, and well-structured English factual summary
of this chunk.

Important:

- This is only one part of source document {part_number}.
- Do not claim that this chunk contains all information about the program.
- Preserve concrete facts, especially dates, fees, age requirements,
  course hours, class sizes, activity names, room types, meals,
  limitations, and optional charges.
- Do not add information that is absent from the source.
- Do not write a Chinese summary.
- Do not merely translate sentence by sentence. Organize the information
  by topic while preserving the original meaning.

<source_document
    part="{part_number}"
    chunk="{chunk_number}"
    total_chunks="{total_chunks}">
{chunk_text}
</source_document>
"""

    task_name = (
        f"文档{part_number}-"
        f"分块{chunk_number}/{total_chunks}"
    )

    return call_gpt(
        developer_prompt=SUMMARY_DEVELOPER_PROMPT,
        user_prompt=user_prompt,
        task_name=task_name,
    )


# ============================================================
# 8. 英文摘要合并 Prompt
# ============================================================

MERGE_DEVELOPER_PROMPT = """
You are a rigorous information editor.

You will receive several English partial summaries derived from different
chunks of the same original summer school document.

Merge them into one complete, accurate, deduplicated, and well-organized
English factual summary.

Requirements:

1. Use only facts contained in the partial summaries.
2. Do not introduce external information.
3. Remove duplicated statements without deleting unique facts.
4. Preserve all important:
   - names;
   - dates;
   - locations;
   - age requirements;
   - prices;
   - additional charges;
   - class sizes;
   - teaching hours;
   - course and activity names;
   - accommodation details;
   - meal arrangements;
   - application conditions.
5. Clearly distinguish optional arrangements from compulsory arrangements.
6. Preserve limiting terms such as:
   "up to", "maximum", "minimum", "may", "optional", "usually",
   and "subject to availability".
7. If the partial summaries contain conflicting information, report each
   version and explicitly identify the conflict. Do not choose one version.
8. Reorganize the material by topic rather than concatenating the partial
   summaries in their original order.
9. Write entirely in English.
10. Use clear Markdown headings and bullet points.
11. Do not use promotional slogans or unsupported claims.
12. Do not mention internal chunk numbers in the final merged summary.

Use this structure when relevant:

1. Program Overview
2. School and Location
3. Target Students and Age Requirements
4. Dates and Program Duration
5. Academic Courses and Teaching Arrangements
6. Non-Academic Activities, Sports, and Excursions
7. Accommodation
8. Meals and Catering
9. Fees and Additional Charges
10. Application, Booking, and Other Requirements
11. Missing, Unclear, or Conflicting Information
"""


def pack_summaries(
    summaries: list[str],
    max_chars: int = MAX_CHARS_PER_MERGE,
) -> list[list[str]]:
    """
    将多份摘要分组，避免合并时一次性输入过长。
    """

    groups: list[list[str]] = []
    current_group: list[str] = []
    current_length = 0

    for summary in summaries:
        additional_length = len(summary) + 100

        if (
            current_group
            and current_length + additional_length > max_chars
        ):
            groups.append(current_group)

            current_group = [summary]
            current_length = additional_length

        else:
            current_group.append(summary)
            current_length += additional_length

    if current_group:
        groups.append(current_group)

    return groups


def merge_summary_group(
    part_number: int,
    summaries: list[str],
    merge_level: int,
    group_number: int,
    total_groups: int,
) -> str:
    """合并一组英文子摘要。"""

    summary_blocks: list[str] = []

    for index, summary in enumerate(
        summaries,
        start=1,
    ):
        summary_blocks.append(
            f"### Partial Summary {index}\n\n"
            f"{summary}"
        )

    combined_text = "\n\n".join(
        summary_blocks
    )

    cache_key = text_hash(
        f"{MODEL}\n"
        f"{part_number}\n"
        f"{merge_level}\n"
        f"{combined_text}"
    )

    cache_path = (
        MERGE_CACHE_DIR
        / (
            f"{part_number}_"
            f"level_{merge_level:02d}_"
            f"group_{group_number:02d}_"
            f"{cache_key}.txt"
        )
    )

    if (
        cache_path.exists()
        and not FORCE_REGENERATE
    ):
        logger.info(
            "读取合并缓存：%s",
            cache_path,
        )

        return read_text(cache_path).strip()

    user_prompt = f"""
All partial summaries below were derived from source document
{part_number}.

This is merge level {merge_level}, group
{group_number} of {total_groups}.

Merge the partial summaries into one complete, accurate, deduplicated,
and topic-organized English factual summary.

Do not mention the merge level, group number, or partial-summary numbers
in the resulting summary.

<partial_summaries>
{combined_text}
</partial_summaries>
"""

    task_name = (
        f"文档{part_number}-"
        f"合并层{merge_level}-"
        f"组{group_number}/{total_groups}"
    )

    result = call_gpt(
        developer_prompt=MERGE_DEVELOPER_PROMPT,
        user_prompt=user_prompt,
        task_name=task_name,
    )

    write_text(
        cache_path,
        result,
    )

    return result


def merge_all_summaries(
    part_number: int,
    summaries: list[str],
) -> str:
    """
    递归合并同一原始文件的所有分块摘要。
    """

    if not summaries:
        raise ValueError(
            f"第 {part_number} 份资料没有可合并的摘要"
        )

    if len(summaries) == 1:
        return summaries[0].strip()

    current_summaries = summaries[:]
    merge_level = 1

    while len(current_summaries) > 1:
        groups = pack_summaries(
            current_summaries
        )

        # 极端情况下，每组只有一个摘要，
        # 会造成合并轮数无法减少。
        # 此时强制两个一组。
        if (
            len(groups) == len(current_summaries)
            and len(current_summaries) > 1
        ):
            groups = [
                current_summaries[index:index + 2]
                for index in range(
                    0,
                    len(current_summaries),
                    2,
                )
            ]

        print()
        print(
            f"文档 {part_number}："
            f"开始第 {merge_level} 层合并，"
            f"本层共 {len(groups)} 组"
        )

        next_level_summaries: list[str] = []

        for group_index, group in enumerate(
            groups,
            start=1,
        ):
            merged_summary = merge_summary_group(
                part_number=part_number,
                summaries=group,
                merge_level=merge_level,
                group_number=group_index,
                total_groups=len(groups),
            )

            next_level_summaries.append(
                merged_summary
            )

        current_summaries = next_level_summaries
        merge_level += 1

    return current_summaries[0].strip()


# ============================================================
# 9. 处理单个文件
# ============================================================

def summarize_file(
    part_number: int,
    input_path: Path,
    output_path: Path,
) -> None:
    """
    处理一个完整原始文件：

    原文
      → 自动切块
      → 每块英文摘要
      → 多块摘要合并
      → 保存 N_en.txt
    """

    if (
        output_path.exists()
        and not FORCE_REGENERATE
    ):
        existing_content = read_text(
            output_path
        ).strip()

        if existing_content:
            print()
            print(
                f"[跳过] {output_path.name} 已经存在。"
            )
            print(
                "如需重新生成，请把 "
                "FORCE_REGENERATE 改为 True。"
            )
            return

    original_text = read_text(
        input_path
    ).strip()

    if not original_text:
        logger.warning(
            "输入文件为空，跳过：%s",
            input_path,
        )
        return

    chunks = split_text(
        original_text,
        max_chars=MAX_CHARS_PER_CHUNK,
    )

    if not chunks:
        raise RuntimeError(
            f"文件 {input_path} 没有成功生成文本块"
        )

    print()
    print("=" * 72)
    print(f"开始处理原始文件：{input_path}")
    print(f"原始字符数：{len(original_text)}")
    print(f"自动分块数量：{len(chunks)}")
    print(f"最终输出文件：{output_path}")
    print("=" * 72)

    chunk_summaries: list[str] = []

    for chunk_index, chunk in enumerate(
        chunks,
        start=1,
    ):
        chunk_key = text_hash(
            f"{MODEL}\n{chunk}"
        )

        cache_path = (
            CHUNK_CACHE_DIR
            / (
                f"{part_number}_"
                f"chunk_{chunk_index:03d}_"
                f"{chunk_key}_en.txt"
            )
        )

        print()
        print("-" * 72)
        print(
            f"正在处理文档 {part_number}："
            f"分块 {chunk_index}/{len(chunks)}"
        )
        print(
            f"当前分块字符数：{len(chunk)}"
        )
        print("-" * 72)

        if (
            cache_path.exists()
            and not FORCE_REGENERATE
        ):
            logger.info(
                "读取分块缓存：%s",
                cache_path,
            )

            chunk_summary = read_text(
                cache_path
            ).strip()

        else:
            chunk_summary = summarize_chunk(
                part_number=part_number,
                chunk_number=chunk_index,
                total_chunks=len(chunks),
                chunk_text=chunk,
            )

            write_text(
                cache_path,
                chunk_summary,
            )

            logger.info(
                "分块摘要已保存：%s",
                cache_path,
            )

        if not chunk_summary:
            raise RuntimeError(
                f"文档 {part_number} 的第 "
                f"{chunk_index} 个分块摘要为空"
            )

        chunk_summaries.append(
            chunk_summary
        )

    print()
    print(
        f"文档 {part_number} 的所有分块摘要已经完成，"
        "开始合并。"
    )

    final_summary = merge_all_summaries(
        part_number=part_number,
        summaries=chunk_summaries,
    )

    if not final_summary:
        raise RuntimeError(
            f"文档 {part_number} 的最终摘要为空"
        )

    write_text(
        output_path,
        final_summary,
    )

    print()
    print("=" * 72)
    print(f"文档 {part_number} 处理完成")
    print(f"英文摘要字符数：{len(final_summary)}")
    print(f"已保存到：{output_path.resolve()}")
    print("=" * 72)

    if PRINT_FINAL_SUMMARY:
        print()
        print(
            f"========== {part_number}_en.txt =========="
        )
        print()
        print(final_summary)
        print()
        print(
            "=" * 72
        )


# ============================================================
# 10. 主程序
# ============================================================

def main() -> None:
    """依次处理第 1～8 份资料。"""

    logger.warning("程序开始运行")

    CHUNK_CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MERGE_CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    successful_files: list[str] = []
    failed_files: list[str] = []
    missing_files: list[str] = []

    for part_number in PART_NUMBERS:
        input_file = find_input_file(
            part_number
        )

        if input_file is None:
            logger.warning(
                "没有找到第 %d 份文件。"
                "程序尝试过 %d.txt 和 %d。",
                part_number,
                part_number,
                part_number,
            )

            missing_files.append(
                str(part_number)
            )
            continue

        output_file = (
            OUTPUT_DIR
            / f"{part_number}_en.txt"
        )

        try:
            summarize_file(
                part_number=part_number,
                input_path=input_file,
                output_path=output_file,
            )

            successful_files.append(
                input_file.name
            )

        except KeyboardInterrupt:
            logger.warning(
                "用户中断程序。已经生成的缓存不会丢失。"
            )
            raise

        except Exception as error:
            failed_files.append(
                input_file.name
            )

            logger.exception(
                "处理文件 %s 时发生错误：%s",
                input_file,
                error,
            )

            # 某一个文件失败后，继续处理后续文件
            continue

    print()
    print("=" * 72)
    print("全部任务执行结束")
    print("=" * 72)
    print(
        "成功处理：",
        successful_files or "无",
    )
    print(
        "处理失败：",
        failed_files or "无",
    )
    print(
        "未找到文件：",
        missing_files or "无",
    )

    logger.warning("程序运行结束")


if __name__ == "__main__":
    main()