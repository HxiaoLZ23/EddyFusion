"""One-off: extract weekly journal .doc to UTF-8 text."""
from __future__ import annotations

from pathlib import Path

SRC = Path(
    r"c:\Users\HxiaoL\Desktop\黄柏霖归档\8.本科毕业设计周记_黄柏霖_软件222_202201050577"
    r"\8.本科毕业设计周记_黄柏霖_软件222_202201050577.doc"
)
OUT = Path(r"f:\创赛\submission\_weekly_journal_original.txt")


def main() -> None:
    import win32com.client  # type: ignore

    word = win32com.client.Dispatch("Word.Application")
    word.Visible = False
    try:
        doc = word.Documents.Open(str(SRC))
        text = doc.Content.Text
        doc.Close(False)
    finally:
        word.Quit()
    OUT.write_text(text, encoding="utf-8")
    print(f"saved {len(text)} chars -> {OUT}")


if __name__ == "__main__":
    main()
