"""Extract thesis docx to markdown with OMML formula -> LaTeX-ish conversion."""
from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
}


def local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def omml_to_latex(node) -> str:
    tag = local(node.tag)
    if tag == "t":
        return (node.text or "") + (node.tail or "")

    children = list(node)
    child_text = "".join(omml_to_latex(c) for c in children)

    if tag in ("oMath", "oMathPara", "e", "r"):
        return child_text
    if tag == "f":
        num = node.find("m:num", NS)
        den = node.find("m:den", NS)
        n = omml_to_latex(num) if num is not None else "?"
        d = omml_to_latex(den) if den is not None else "?"
        return f"\\frac{{{n}}}{{{d}}}"
    if tag == "sSub":
        e = node.find("m:e", NS)
        sub = node.find("m:sub", NS)
        return f"{omml_to_latex(e) if e is not None else ''}_{{{omml_to_latex(sub) if sub is not None else ''}}}"
    if tag == "sSup":
        e = node.find("m:e", NS)
        sup = node.find("m:sup", NS)
        return f"{omml_to_latex(e) if e is not None else ''}^{{{omml_to_latex(sup) if sup is not None else ''}}}"
    if tag == "sSubSup":
        e = node.find("m:e", NS)
        sub = node.find("m:sub", NS)
        sup = node.find("m:sup", NS)
        base = omml_to_latex(e) if e is not None else ""
        return f"{base}_{{{omml_to_latex(sub) if sub is not None else ''}}}^{{{omml_to_latex(sup) if sup is not None else ''}}}"
    if tag == "rad":
        e = node.find("m:e", NS)
        return f"\\sqrt{{{omml_to_latex(e) if e is not None else ''}}}"
    if tag == "d":
        e = node.find("m:e", NS)
        return f"\\left({omml_to_latex(e) if e is not None else child_text}\\right)"
    if tag == "bar":
        e = node.find("m:e", NS)
        return f"\\overline{{{omml_to_latex(e) if e is not None else ''}}}"
    if tag == "func":
        fname = node.find("m:fName", NS)
        e = node.find("m:e", NS)
        fn = omml_to_latex(fname) if fname is not None else ""
        arg = omml_to_latex(e) if e is not None else ""
        return f"{fn}{arg}"
    return child_text


def para_text_and_math(p) -> tuple[str, list[str]]:
    parts: list[str] = []
    formulas: list[str] = []
    for child in p:
        lt = local(child.tag)
        if lt == "r":
            for sub in child:
                sl = local(sub.tag)
                if sl == "t":
                    parts.append(sub.text or "")
                    if sub.tail:
                        parts.append(sub.tail)
                elif sl == "tab":
                    parts.append("\t")
        elif lt in ("oMath", "oMathPara"):
            fx = omml_to_latex(child).strip()
            if fx:
                formulas.append(fx)
                parts.append(f" ${fx}$ ")
        elif lt == "hyperlink":
            parts.append(para_text_and_math(child)[0])
    return "".join(parts).strip(), formulas


def heading_level(p) -> int:
    pPr = p.find("w:pPr", NS)
    if pPr is None:
        return 0
    pStyle = pPr.find("w:pStyle", NS)
    if pStyle is None:
        return 0
    val = pStyle.get(f"{{{NS['w']}}}val", "")
    if re.search(r"标题|Heading|heading|Title", val, re.I):
        m = re.search(r"(\d+)", val)
        if m:
            return min(max(int(m.group(1)), 1), 6)
        return 2
    return 0


def extract(docx: Path, out_md: Path, out_audit: Path) -> dict[str, int]:
    with zipfile.ZipFile(docx) as z:
        root = ET.fromstring(z.read("word/document.xml"))

    body = root.find("w:body", NS)
    lines: list[str] = []
    formula_audit: list[str] = []
    para_idx = 0
    formula_idx = 0

    for child in body:
        tag = local(child.tag)
        if tag == "p":
            para_idx += 1
            txt, fxs = para_text_and_math(child)
            for fx in fxs:
                formula_idx += 1
                ctx = txt[:80].replace("|", "/") if txt else "(公式独立段)"
                formula_audit.append(f"| {formula_idx} | p{para_idx} | {ctx} | `{fx}` |")
            if not txt and not fxs:
                lines.append("")
                continue
            lvl = heading_level(child)
            lines.append(("#" * lvl + " " + txt) if lvl else txt)
        elif tag == "tbl":
            rows = []
            for tr in child.findall(".//w:tr", NS):
                cells = []
                for tc in tr.findall("w:tc", NS):
                    cell_parts = []
                    for p in tc.findall(".//w:p", NS):
                        t, _ = para_text_and_math(p)
                        if t:
                            cell_parts.append(t)
                    cells.append(" ".join(cell_parts))
                if any(cells):
                    rows.append(cells)
            if rows:
                ncol = max(len(r) for r in rows)
                hdr = rows[0] + [""] * (ncol - len(rows[0]))
                lines.append("| " + " | ".join(hdr) + " |")
                lines.append("| " + " | ".join(["---"] * ncol) + " |")
                for r in rows[1:]:
                    r = r + [""] * (ncol - len(r))
                    lines.append("| " + " | ".join(r[:ncol]) + " |")
                lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8")
    audit = [
        "# 公式提取审计",
        "",
        f"- 源: `{docx}`",
        f"- 段落: {para_idx}",
        f"- 公式: {formula_idx}",
        "",
        "| # | 段落 | 上下文 | LaTeX 近似 |",
        "|---|---|---|---|",
        *formula_audit,
    ]
    out_audit.write_text("\n".join(audit), encoding="utf-8")

    issues: list[str] = []
    for row in formula_audit:
        fx = row.split("| `{")[-1].rstrip("` |") if "| `" in row else ""
        if "_{}" in fx or fx.endswith("_{"):
            issues.append(f"- 空下标: `{fx}`")
        if "Mask" in fx and "=" not in fx and "σ" in fx:
            issues.append(f"- 掩膜公式缺等号: `{fx}`")
        if "min\\left" in fx and "," not in fx and "D(i" in fx:
            issues.append(f"- DTW min 缺分隔: `{fx}`")
    if issues:
        issue_path = out_audit.with_name(out_audit.stem + "_issues.md")
        issue_path.write_text(
            "# 公式自动质检\n\n" + "\n".join(issues),
            encoding="utf-8",
        )
    return {"paragraphs": para_idx, "formulas": formula_idx, "lines": len(lines), "issues": len(issues)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("docx", type=Path)
    ap.add_argument("--out-md", type=Path, required=True)
    ap.add_argument("--out-audit", type=Path, required=True)
    args = ap.parse_args()
    stats = extract(args.docx, args.out_md, args.out_audit)
    print(stats)


if __name__ == "__main__":
    main()
