import logging
import re
import time
from pathlib import Path

from openai import OpenAI


# 你原来的代理配置可以继续放在这里
# openai.proxy = {
#     "http": "http://127.0.0.1:1080",
#     "https": "http://127.0.0.1:1080"
# }

MODEL = "gpt-4.1-mini-2025-04-14"

# 输入文件和输出文件都放在当前目录
INPUT_DIR = Path(".")
OUTPUT_DIR = Path(".")

# 单个请求最多传入约 18000 个字符
# 这个值不是模型极限，而是为了提高总结稳定性
MAX_CHARS_PER_CHUNK = 18000

# 长单行被强制切分时，保留少量重叠内容
OVERLAP_CHARS = 300

# 保存长文件每个分块的中间摘要
CACHE_DIR = Path("summary_cache")

# True：重新生成已经存在的摘要
# False：存在 1_cn.txt 时直接跳过
FORCE_REGENERATE = False


client = OpenAI()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)


def read_text(file_path: Path) -> str:
    """读取 UTF-8 文本文件。"""

    with file_path.open("r", encoding="utf-8") as file:
        return file.read()


def write_text(file_path: Path, content: str) -> None:
    """将文本保存为 UTF-8 文件。"""

    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", encoding="utf-8") as file:
        file.write(content)


def call_gpt(
    developer_prompt: str,
    user_prompt: str,
    max_retries: int = 3,
) -> str:
    """调用 GPT，并在请求失败时自动重试。"""

    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
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
                temperature=0,
            )

            answer = response.choices[0].message.content

            if not answer:
                raise RuntimeError("模型返回了空内容")

            if response.usage:
                print(
                    f"Input tokens: "
                    f"{response.usage.prompt_tokens}"
                )
                print(
                    f"Output tokens: "
                    f"{response.usage.completion_tokens}"
                )

            return answer.strip()

        except Exception as error:
            last_error = error

            logger.warning(
                "第 %d 次请求失败：%s",
                attempt,
                error,
            )

            if attempt < max_retries:
                time.sleep(attempt * 3)

    raise RuntimeError(
        f"连续 {max_retries} 次请求失败：{last_error}"
    )


def hard_split_long_paragraph(
    paragraph: str,
    max_chars: int,
    overlap_chars: int,
) -> list[str]:
    """
    当单个段落本身就特别长时，按照字符数量强制切分。
    相邻块保留少量重叠，避免边界信息丢失。
    """

    chunks = []
    start = 0

    while start < len(paragraph):
        end = min(start + max_chars, len(paragraph))
        chunks.append(paragraph[start:end])

        if end >= len(paragraph):
            break

        start = end - overlap_chars

    return chunks


def split_text(
    text: str,
    max_chars: int = MAX_CHARS_PER_CHUNK,
) -> list[str]:
    """
    优先按照段落切分文本。

    如果原始网页文本没有空行，则退化为按行切分。
    单个段落超过限制时，再按照字符强制切分。
    """

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

    # 网页提取内容可能每行一段，但没有空行
    if len(paragraphs) <= 1:
        paragraphs = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

    chunks = []
    current_paragraphs = []
    current_length = 0

    for paragraph in paragraphs:
        # 当前单个段落本身就超过限制
        if len(paragraph) > max_chars:
            if current_paragraphs:
                chunks.append(
                    "\n\n".join(current_paragraphs)
                )
                current_paragraphs = []
                current_length = 0

            long_chunks = hard_split_long_paragraph(
                paragraph=paragraph,
                max_chars=max_chars,
                overlap_chars=OVERLAP_CHARS,
            )

            chunks.extend(long_chunks)
            continue

        additional_length = len(paragraph) + 2

        # 加入新段落后超过限制，先保存现有块
        if (
            current_paragraphs
            and current_length + additional_length > max_chars
        ):
            chunks.append(
                "\n\n".join(current_paragraphs)
            )

            current_paragraphs = [paragraph]
            current_length = len(paragraph)
        else:
            current_paragraphs.append(paragraph)
            current_length += additional_length

    if current_paragraphs:
        chunks.append(
            "\n\n".join(current_paragraphs)
        )

    return chunks


SUMMARY_DEVELOPER_PROMPT = """
你是一名严谨的夏校项目资料整理员。

你的当前任务是总结原始资料，而不是撰写最终招生宣传文章。

必须遵守以下要求：

1. 只能总结资料中明确出现的内容。
2. 不得使用外部知识补充。
3. 不得根据常识推测。
4. 不得编造不存在的课程、费用、日期、住宿或申请要求。
5. 必须尽可能保留原文中的数字、日期、年龄、价格、课时、
   班级人数、课程名称、地点和限制条件。
6. 对含义相近的信息进行归纳，但不能因为压缩而遗漏重要事实。
7. 如果资料中存在前后不一致的信息，应分别保留并明确指出，
   不得自行选择其中一个作为正确答案。
8. 输出中文。
9. 使用清晰的小标题和条目。
10. 不要写“欢迎报名”“不容错过”等宣传口号。

请优先按照以下类别整理：

一、项目基本信息
二、招生对象与年龄要求
三、日期与项目周期
四、学术课程与教学安排
五、非学术活动、体育及游览
六、住宿安排
七、餐饮安排
八、费用及额外收费
九、申请、预订与其他要求
十、资料中未明确说明或存在矛盾的信息
"""


def summarize_chunk(
    part_number: int,
    chunk_number: int,
    total_chunks: int,
    chunk_text: str,
) -> str:
    """总结一个文本分块。"""

    user_prompt = f"""
下面是第 {part_number} 份原始资料的第
{chunk_number}/{total_chunks} 个分块。

请对当前分块进行详细的中文事实摘要。

注意：

- 当前只是整个第 {part_number} 份资料的一部分；
- 不要声称这是完整资料；
- 尽可能保留具体事实；
- 不要为了简洁而删除价格、日期、年龄、课程时间和住宿信息；
- 不要添加当前资料中没有的信息。

<source part="{part_number}"
chunk="{chunk_number}/{total_chunks}">
{chunk_text}
</source>
"""

    return call_gpt(
        developer_prompt=SUMMARY_DEVELOPER_PROMPT,
        user_prompt=user_prompt,
    )


MERGE_DEVELOPER_PROMPT = """
你是一名严谨的资料合并编辑。

你会收到同一份夏校原始资料的多个分块摘要。
请将这些摘要合并为一份完整的中文事实摘要。

要求：

1. 删除重复信息，但不能删除独有事实。
2. 保留所有重要数字、日期、费用、年龄、地点和课程名称。
3. 不得引入分块摘要中没有的新信息。
4. 不得把可选项目写成必选项目。
5. 不得把“最高”“最多”“可能”等限制性表达删除。
6. 如果不同分块出现矛盾，分别列出，不要自行判断。
7. 按主题重新组织，而不是简单地将摘要依次拼接。
8. 这是资料摘要，不是最终宣传文章。
9. 输出中文。
10. 使用清晰的小标题和条目。

推荐结构：

一、项目概况
二、招生对象与时间
三、学术课程
四、非学术活动与游览
五、住宿
六、餐饮
七、费用
八、申请和其他要求
九、资料中的缺失或矛盾信息
"""


def pack_summaries(
    summaries: list[str],
    max_chars: int = MAX_CHARS_PER_CHUNK,
) -> list[list[str]]:
    """
    将多份分块摘要分组，防止合并摘要时输入再次过长。
    """

    groups = []
    current_group = []
    current_length = 0

    for summary in summaries:
        summary_length = len(summary)

        if (
            current_group
            and current_length + summary_length > max_chars
        ):
            groups.append(current_group)
            current_group = [summary]
            current_length = summary_length
        else:
            current_group.append(summary)
            current_length += summary_length

    if current_group:
        groups.append(current_group)

    return groups


def merge_summary_group(
    part_number: int,
    summaries: list[str],
    level: int,
    group_number: int,
    total_groups: int,
) -> str:
    """合并一组摘要。"""

    summary_blocks = []

    for index, summary in enumerate(summaries, start=1):
        summary_blocks.append(
            f"### 子摘要 {index}\n\n{summary}"
        )

    combined_text = "\n\n".join(summary_blocks)

    user_prompt = f"""
以下内容均来自第 {part_number} 份原始资料。

当前正在进行第 {level} 层摘要合并，
这是第 {group_number}/{total_groups} 组。

请将下面的子摘要合并成一份完整、准确、去重后的中文事实摘要。

<partial_summaries>
{combined_text}
</partial_summaries>
"""

    return call_gpt(
        developer_prompt=MERGE_DEVELOPER_PROMPT,
        user_prompt=user_prompt,
    )


def merge_all_summaries(
    part_number: int,
    summaries: list[str],
) -> str:
    """
    递归合并所有分块摘要。

    即使一个文件被切成很多块，也不会在最终合并时再次超过长度限制。
    """

    if not summaries:
        raise ValueError("没有可供合并的摘要")

    if len(summaries) == 1:
        return summaries[0]

    current_summaries = summaries
    level = 1

    while len(current_summaries) > 1:
        groups = pack_summaries(current_summaries)

        print(
            f"第 {part_number} 份资料："
            f"第 {level} 层合并，"
            f"共 {len(groups)} 组"
        )

        next_level_summaries = []

        for group_index, group in enumerate(groups, start=1):
            merged_summary = merge_summary_group(
                part_number=part_number,
                summaries=group,
                level=level,
                group_number=group_index,
                total_groups=len(groups),
            )

            next_level_summaries.append(merged_summary)

        current_summaries = next_level_summaries
        level += 1

    return current_summaries[0]


def summarize_file(
    part_number: int,
    input_path: Path,
    output_path: Path,
) -> None:
    """总结一个完整的 txt 文件并保存。"""

    if output_path.exists() and not FORCE_REGENERATE:
        print(
            f"[跳过] {output_path} 已存在"
        )
        return

    if not input_path.exists():
        logger.warning(
            "文件不存在，跳过：%s",
            input_path,
        )
        return

    original_text = read_text(input_path)

    if not original_text.strip():
        logger.warning(
            "文件为空，跳过：%s",
            input_path,
        )
        return

    chunks = split_text(original_text)

    print("\n" + "=" * 60)
    print(f"开始处理：{input_path}")
    print(f"原文字符数：{len(original_text)}")
    print(f"自动分块数：{len(chunks)}")
    print("=" * 60)

    chunk_summaries = []

    for chunk_index, chunk in enumerate(chunks, start=1):
        cache_path = (
            CACHE_DIR
            / f"{part_number}_chunk_{chunk_index:02d}_cn.txt"
        )

        print(
            f"\n正在总结第 {part_number} 份资料："
            f"{chunk_index}/{len(chunks)}"
        )
        print(
            f"当前块字符数：{len(chunk)}"
        )

        # 已经生成的中间摘要直接复用，支持断点续跑
        if cache_path.exists() and not FORCE_REGENERATE:
            print(
                f"[缓存] 读取 {cache_path}"
            )
            chunk_summary = read_text(cache_path)
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

        chunk_summaries.append(chunk_summary)

    # 多个分块摘要再次合并
    final_summary = merge_all_summaries(
        part_number=part_number,
        summaries=chunk_summaries,
    )

    write_text(
        output_path,
        final_summary,
    )

    print("\n" + "-" * 60)
    print(f"已保存：{output_path}")
    print(f"摘要字符数：{len(final_summary)}")
    print("-" * 60)

    print("\n生成的摘要如下：\n")
    print(final_summary)


if __name__ == "__main__":
    logger.warning("运行开始")

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for part_number in range(1, 9):
        input_file = INPUT_DIR / f"{part_number}.txt"
        output_file = OUTPUT_DIR / f"{part_number}_cn.txt"

        try:
            summarize_file(
                part_number=part_number,
                input_path=input_file,
                output_path=output_file,
            )

        except Exception as error:
            logger.exception(
                "处理 %s 时发生错误：%s",
                input_file,
                error,
            )

            # 一个文件失败后继续处理其他文件
            continue

    logger.warning("运行结束")