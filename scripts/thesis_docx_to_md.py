#!/usr/bin/env python3
"""将定稿论文 docx 转为 Markdown 镜像（Phase 0 基线，不改写正文）。"""

from __future__ import annotations

import argparse
import re
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from docx import Document  # noqa: E402
from docx.oxml.ns import qn  # noqa: E402
from docx.table import Table  # noqa: E402
from docx.text.paragraph import Paragraph  # noqa: E402


def _iter_block_items(parent):
    from docx.document import Document as Doc

    if isinstance(parent, Doc):
        parent_elm = parent.element.body
    else:
        parent_elm = parent._tc
    for child in parent_elm.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, parent)
        elif child.tag == qn("w:tbl"):
            yield Table(child, parent)


def _para_style_level(style_name: str | None) -> int | None:
    if not style_name:
        return None
    m = re.search(r"(?:Heading|标题)\s*(\d+)", style_name, re.I)
    if m:
        return int(m.group(1))
    if style_name in ("Title", "标题"):
        return 1
    return None


def _escape_md(text: str) -> str:
    return text.replace("|", "\\|").strip()


def _table_to_md(table: Table) -> str:
    rows: list[list[str]] = []
    for row in table.rows:
        rows.append([_escape_md(cell.text.replace("\n", " ")) for cell in row.cells])
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    for r in rows:
        while len(r) < width:
            r.append("")
    lines = [
        "| " + " | ".join(rows[0]) + " |",
        "| " + " | ".join("---" for _ in rows[0]) + " |",
    ]
    for r in rows[1:]:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def _extract_images(docx_path: Path, assets_dir: Path) -> dict[str, str]:
    """从 docx zip 提取 media，返回 rId→相对路径。"""
    assets_dir.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, str] = {}
    with zipfile.ZipFile(docx_path, "r") as zf:
        media = [n for n in zf.namelist() if n.startswith("word/media/")]
        for i, name in enumerate(sorted(media), start=1):
            suffix = Path(name).suffix or ".png"
            out_name = f"media_{i:03d}{suffix}"
            out_path = assets_dir / out_name
            out_path.write_bytes(zf.read(name))
            mapping[name] = f"assets/{out_name}"
    return mapping


def docx_to_md(docx_path: Path, out_md: Path, assets_dir: Path) -> dict[str, int]:
    doc = Document(str(docx_path))
    _extract_images(docx_path, assets_dir)

    lines: list[str] = []
    stats = {"paragraphs": 0, "tables": 0, "headings": 0, "chars": 0}

    for block in _iter_block_items(doc):
        if isinstance(block, Paragraph):
            text = block.text.strip()
            if not text:
                lines.append("")
                continue
            stats["paragraphs"] += 1
            stats["chars"] += len(text)
            level = _para_style_level(block.style.name if block.style else None)
            if level:
                stats["headings"] += 1
                lines.append("#" * min(level, 6) + " " + text)
            else:
                lines.append(text)
            lines.append("")
        elif isinstance(block, Table):
            stats["tables"] += 1
            md_table = _table_to_md(block)
            if md_table:
                lines.append(md_table)
                lines.append("")

    body = "\n".join(lines)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(body, encoding="utf-8")
    return stats


def build_toc_stats(md_path: Path, out_stats: Path) -> None:
    text = md_path.read_text(encoding="utf-8")
    headings: list[tuple[int, str]] = []
    for line in text.splitlines():
        m = re.match(r"^(#{1,6})\s+(.+)$", line)
        if m:
            headings.append((len(m.group(1)), m.group(2).strip()))
    total_chars = len(re.sub(r"\s+", "", text))
    lines_out = [
        "# 章节目录与字数统计",
        "",
        f"- 源 md：`{md_path.name}`",
        f"- 总字符（去空白）：{total_chars}",
        f"- 标题数：{len(headings)}",
        "",
        "## 目录树",
        "",
    ]
    for lvl, title in headings:
        indent = "  " * (lvl - 1)
        lines_out.append(f"{indent}- {'#' * lvl} {title}")
    out_stats.write_text("\n".join(lines_out), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="定稿论文 docx → Markdown 镜像")
    parser.add_argument(
        "--docx",
        type=str,
        default=r"c:\Users\HxiaoL\Desktop\系统模式论文\基于深度学习的海洋涡旋识别与风浪预警系统设计.docx",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="submission/thesis/基于深度学习的海洋涡旋识别与风浪预警系统设计_原文镜像.md",
    )
    parser.add_argument("--assets", type=str, default="submission/thesis/assets")
    parser.add_argument("--stats", type=str, default="submission/thesis/章节目录与字数统计.md")
    parser.add_argument("--copy-working", action="store_true", help="复制为降AIGC工作稿")
    args = parser.parse_args()

    docx_path = Path(args.docx)
    if not docx_path.is_file():
        raise FileNotFoundError(f"未找到 docx: {docx_path}")

    out_md = REPO_ROOT / args.out
    assets_dir = REPO_ROOT / args.assets
    stats_path = REPO_ROOT / args.stats

    st = docx_to_md(docx_path, out_md, assets_dir)
    build_toc_stats(out_md, stats_path)

    if args.copy_working:
        working = out_md.with_name(out_md.stem.replace("_原文镜像", "_降AIGC") + out_md.suffix)
        if "_原文镜像" in out_md.stem:
            working = out_md.parent / "基于深度学习的海洋涡旋识别与风浪预警系统设计_降AIGC.md"
        working.write_text(out_md.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"工作稿: {working}")

    print(
        f"OK paragraphs={st['paragraphs']} tables={st['tables']} headings={st['headings']} "
        f"chars={st['chars']}\n→ {out_md}\n→ {stats_path}"
    )


if __name__ == "__main__":
    main()
