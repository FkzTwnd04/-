"""定义生成提示中的来源编号和引用元数据，并校验引用编号的来源范围。

检索块会被编号为 [S1]、[S2] 并加入模型上下文；非流式答案会移除本轮资料中不存在的编号。
流式回答会把实际检索块的文件名返回前端展示，但还没有在整段答案生成后校验引用编号。
"""

from dataclasses import asdict, dataclass
import re
from typing import Iterable, List, Tuple

from rag_qa.schema.chunk import Chunk


@dataclass(frozen=True)
class Citation:
    """source_id 是提示词中的编号；其余字段用于前端展示引用出处。"""
    source_id: str
    chunk_id: str
    document_id: str
    source: str
    filename: str
    page: int | None
    excerpt: str

    def to_dict(self) -> dict:
        """转成 API 可直接序列化的字典。"""
        return asdict(self)


def build_citation_context(chunks: Iterable[Chunk]) -> Tuple[str, List[Citation]]:
    """给上下文块编号为 S1、S2……并生成同序的可展示引用元数据。"""
    sections = []
    citations = []
    for index, chunk in enumerate(chunks, start=1):
        source_id = f"S{index}"
        sections.append(f"[{source_id}] {chunk.text}")
        citations.append(Citation(
            source_id=source_id,
            chunk_id=chunk.chunk_id,
            document_id=chunk.doc_id,
            source=chunk.source,
            filename=chunk.filename,
            page=chunk.page,
            excerpt=chunk.text[:300],
        ))
    return "\n\n".join(sections), citations


def validate_answer_citations(answer: str, citations: Iterable[Citation]) -> Tuple[str, List[Citation]]:
    """移除答案中不属于本轮检索上下文的引用编号，并只返回答案實際引用的来源。"""
    by_id = {citation.source_id: citation for citation in citations}
    used = []

    def replace(match):
        source_id = match.group(1)
        if source_id not in by_id:
            return ""
        if source_id not in {item.source_id for item in used}:
            used.append(by_id[source_id])
        return match.group(0)

    cleaned = re.sub(r"\[(S\d+)\]", replace, answer or "")
    return cleaned, used
