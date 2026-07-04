"""Extract readable Chinese text from legacy .doc WordDocument stream."""
from __future__ import annotations

import re
from pathlib import Path

import olefile

SRC = Path(
    r"c:\Users\HxiaoL\Desktop\黄柏霖归档\8.本科毕业设计周记_黄柏霖_软件222_202201050577"
    r"\8.本科毕业设计周记_黄柏霖_软件222_202201050577.doc"
)
OUT = Path(r"f:\创赛\submission\_weekly_journal_original.txt")


def extract_utf16_runs(data: bytes) -> list[str]:
    out: list[str] = []
    i = 0
    n = len(data)
    while i < n - 1:
        if data[i + 1] != 0:
            i += 1
            continue
        j = i
        buf = bytearray()
        while j < n - 1 and data[j + 1] == 0 and data[j] != 0:
            buf.extend((data[j], data[j + 1]))
            j += 2
        if len(buf) >= 4:
            try:
                s = buf.decode("utf-16-le")
                if re.search(r"[\u4e00-\u9fff]", s):
                    out.append(s)
            except UnicodeDecodeError:
                pass
        i = max(j, i + 1)
    return out


def extract_gbk_runs(data: bytes) -> list[str]:
    out: list[str] = []
    i = 0
    n = len(data)
    while i < n - 1:
        b1, b2 = data[i], data[i + 1]
        if 0x81 <= b1 <= 0xFE and 0x40 <= b2 <= 0xFE:
            j = i
            buf = bytearray()
            while j < n - 1:
                b1, b2 = data[j], data[j + 1]
                if 0x81 <= b1 <= 0xFE and 0x40 <= b2 <= 0xFE:
                    buf.extend((b1, b2))
                    j += 2
                elif 0x20 <= b1 <= 0x7E:
                    buf.append(b1)
                    j += 1
                else:
                    break
            if len(buf) >= 4:
                try:
                    s = buf.decode("gbk")
                    if re.search(r"[\u4e00-\u9fff]", s) and len(s) >= 2:
                        out.append(s)
                except UnicodeDecodeError:
                    pass
            i = max(j, i + 1)
        else:
            i += 1
    return out


def main() -> None:
    ole = olefile.OleFileIO(str(SRC))
    data = ole.openstream("WordDocument").read()
    chunks = extract_utf16_runs(data) + extract_gbk_runs(data)
    # de-dup while preserving order
    seen: set[str] = set()
    ordered: list[str] = []
    for c in chunks:
        c = c.strip("\x00\r\n\t ")
        if len(c) < 2 or c in seen:
            continue
        seen.add(c)
        ordered.append(c)
    text = "\n".join(ordered)
    OUT.write_text(text, encoding="utf-8")
    print(f"saved {len(ordered)} chunks, {len(text)} chars -> {OUT}")


if __name__ == "__main__":
    main()
