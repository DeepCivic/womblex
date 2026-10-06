"""The pdfium text engine: characters in content order, rebuilt into the seam's shapes.

pdfium reports a page's text as characters with boxes and fonts, not as lines or
blocks, so this module does the segmentation MuPDF does natively. Phase 0
(`docs/decisions.md`) chose this over pdfminer's layout analysis on speed. The
segmentation here (`segment`) is pure — it takes `Char`s — so it is tested
without a PDF; `read_chars` is the only part that touches pdfium.

Reading order is content-stream order, as in MuPDF; a column jump starts a new
block but no reordering is attempted, so multi-column pages differ from MuPDF
exactly where Phase 0 measured the divergence tail.

Dehyphenation follows MuPDF's observable behaviour: a line ending in a hyphen
after a letter, followed in the same block by a line starting with a letter,
loses the hyphen and joins that line. It does not check case, as MuPDF does not.
"""

from __future__ import annotations

import ctypes
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pypdfium2.raw as pdfium_c  # type: ignore[import-untyped]

from womblex.ingest.pdf.types import Block, Line, Rect, Span, Word

if TYPE_CHECKING:
    from collections.abc import Callable

    import pypdfium2 as pdfium  # type: ignore[import-untyped]

#: PDF font descriptor flag ForceBold, and the weight from which a font is bold.
_FORCE_BOLD = 1 << 18
_BOLD_WEIGHT = 600
_SUBSET_PREFIX = re.compile(r"^[A-Z]{6}\+")
_HYPHENS = ("-", "­")


@dataclass(frozen=True)
class Char:
    """One character, its box in top-left page space, and its font."""

    text: str
    box: Rect
    size: float
    font: str
    bold: bool


@dataclass
class _Line:
    chars: list[Char]

    @property
    def box(self) -> Rect:
        return _union([c.box for c in self.chars])

    @property
    def text(self) -> str:
        return "".join(c.text for c in self.chars)


def _union(boxes: list[Rect]) -> Rect:
    return Rect(
        min(b.x0 for b in boxes), min(b.y0 for b in boxes),
        max(b.x1 for b in boxes), max(b.y1 for b in boxes),
    )


def _is_bold(weight: int, flags: int, font: str) -> bool:
    return weight >= _BOLD_WEIGHT or bool(flags & _FORCE_BOLD) or "bold" in font.lower()


def read_chars(page: pdfium.PdfPage, to_rect: Callable[[tuple[float, float, float, float]], Rect]) -> list[Char]:
    """Every printable character of *page*, in content order.

    *to_rect* turns a pdfium user-space box into top-left page space. Control
    characters (pdfium's generated line breaks) are dropped: lines are rebuilt
    from geometry. A surrogate pair is one character, boxed by its first unit.
    """
    textpage = page.get_textpage()
    try:
        raw = textpage.raw
        out: list[Char] = []
        high = 0
        name = ctypes.create_string_buffer(256)
        flags = ctypes.c_int(0)
        for i in range(textpage.count_chars()):
            code = pdfium_c.FPDFText_GetUnicode(raw, i)
            if 0xD800 <= code < 0xDC00:
                high = code
                continue
            if 0xDC00 <= code < 0xE000 and high:
                text = chr(0x10000 + ((high - 0xD800) << 10) + (code - 0xDC00))
            else:
                text = chr(code)
            high = 0
            if text == "\x02":
                # pdfium's stand-in for a hyphen at a line end.
                text = "-"
            if text in "\r\n\x00￾" or (ord(text) < 0x20 and text != "\t"):
                continue
            length = pdfium_c.FPDFText_GetFontInfo(raw, i, name, len(name), ctypes.byref(flags))
            font = _SUBSET_PREFIX.sub("", name.value.decode("latin-1")) if length > 0 else ""
            weight = pdfium_c.FPDFText_GetFontWeight(raw, i)
            left, bottom, right, top = textpage.get_charbox(i, loose=True)
            out.append(Char(
                text=" " if text == "\t" else text,
                box=to_rect((left, bottom, right, top)),
                size=float(pdfium_c.FPDFText_GetFontSize(raw, i)),
                font=font,
                bold=_is_bold(weight, flags.value, font),
            ))
        return out
    finally:
        textpage.close()


def _same_line(line: _Line, char: Char) -> bool:
    """The character sits on the line's row, abutting it rather than across a gutter."""
    box = line.chars[-1].box
    overlap = min(box.y1, char.box.y1) - max(box.y0, char.box.y0)
    shortest = min(box.height, char.box.height)
    return overlap > 0.5 * shortest and box.x1 - char.size <= char.box.x0 <= box.x1 + _MAX_GAP * char.size


#: How many earlier lines a stray fragment may rejoin; content order seldom
#: strays further than the previous few text objects.
_REJOIN_WINDOW = 8
#: A gap wider than this many font sizes is a column gutter, not a word gap.
_MAX_GAP = 1.5


def _join(left: _Line, right: _Line) -> _Line:
    """*right*, which sits beyond *left*'s end, appended with a space if none shows."""
    a, b = left.chars[-1], right.chars[0]
    chars = left.chars + right.chars
    if a.text != " " and b.text != " " and b.box.x0 - a.box.x1 > 0.2 * a.size:
        gap = Rect(a.box.x1, a.box.y0, b.box.x0, a.box.y1)
        chars = [*left.chars, Char(" ", gap, a.size, a.font, a.bold), *right.chars]
    return _Line(chars)


def _rejoin(lines: list[_Line], line: _Line) -> bool:
    """Merge a fragment into the recent line on its row that it abuts."""
    box, size = line.box, line.chars[0].size
    for i in range(len(lines) - 1, max(len(lines) - 1 - _REJOIN_WINDOW, -1), -1):
        other = lines[i].box
        overlap = min(box.y1, other.y1) - max(box.y0, other.y0)
        if overlap <= 0.5 * min(box.height, other.height):
            continue
        if -0.3 * size <= box.x0 - other.x1 <= _MAX_GAP * size:
            lines[i] = _join(lines[i], line)
            return True
        if -0.3 * size <= other.x0 - box.x1 <= _MAX_GAP * size:
            lines[i] = _join(line, lines[i])
            return True
    return False


def _lines(chars: list[Char]) -> list[_Line]:
    runs: list[_Line] = []
    for char in chars:
        if runs and _same_line(runs[-1], char):
            runs[-1].chars.append(char)
        else:
            runs.append(_Line([char]))
    lines: list[_Line] = []
    for run in runs:
        if not _rejoin(lines, run):
            lines.append(run)
    for line in lines:
        # Edge spaces are separators between layout runs, not content.
        while line.chars and line.chars[-1].text == " ":
            line.chars.pop()
        while line.chars and line.chars[0].text == " ":
            line.chars.pop(0)
    return [line for line in lines if line.chars]


def _same_block(prev: _Line, line: _Line) -> bool:
    a, b = prev.box, line.box
    height = max(a.height, b.height)
    return -0.5 * height <= b.y0 - a.y1 <= 0.7 * height and b.x0 < a.x1


def segment(chars: list[Char]) -> list[list[_Line]]:
    """Characters into blocks of lines."""
    blocks: list[list[_Line]] = []
    for line in _lines(chars):
        if blocks and _same_block(blocks[-1][-1], line):
            blocks[-1].append(line)
        else:
            blocks.append([line])
    return blocks


def _hyphenated(line: _Line, following: _Line) -> bool:
    text = line.text
    return (
        len(text) > 1 and text.endswith(_HYPHENS) and text[-2].isalpha()
        and following.chars[0].text.isalpha()
    )


def _dehyphenated(block: list[_Line]) -> list[_Line]:
    """Join each hyphenated line break; the joined line keeps the hyphen's row."""
    out: list[_Line] = []
    carry: list[Char] | None = None
    for i, line in enumerate(block):
        chars = (carry or []) + line.chars
        carry = None
        if i + 1 < len(block) and _hyphenated(_Line(chars), block[i + 1]):
            carry = chars[:-1]
            continue
        out.append(_Line(chars))
    return out


def _blocks(chars: list[Char], dehyphenate: bool) -> list[list[_Line]]:
    blocks = segment(chars)
    return [_dehyphenated(b) for b in blocks] if dehyphenate else blocks


def plain_text(chars: list[Char], *, dehyphenate: bool) -> str:
    return "".join(f"{line.text}\n" for block in _blocks(chars, dehyphenate) for line in block)


def text_blocks(chars: list[Char], *, dehyphenate: bool) -> list[Block]:
    return [
        Block(
            bbox=_union([line.box for line in block]), number=number,
            text="".join(f"{line.text}\n" for line in block),
        )
        for number, block in enumerate(_blocks(chars, dehyphenate))
    ]


def _spans(line: _Line) -> tuple[Span, ...]:
    runs: list[list[Char]] = []
    for char in line.chars:
        key = (char.font, char.size, char.bold)
        if runs and (runs[-1][0].font, runs[-1][0].size, runs[-1][0].bold) == key:
            runs[-1].append(char)
        else:
            runs.append([char])
    return tuple(
        Span(
            text="".join(c.text for c in run), bbox=_union([c.box for c in run]),
            size=run[0].size, font=run[0].font, bold=run[0].bold,
        )
        for run in runs
    )


def text_dict(chars: list[Char]) -> list[Block]:
    """Blocks with spans. Never dehyphenated: the dict is the raw layout."""
    return [
        Block(
            bbox=_union([line.box for line in block]), number=number,
            lines=tuple(Line(bbox=line.box, spans=_spans(line)) for line in block),
        )
        for number, block in enumerate(segment(chars))
    ]


def words(chars: list[Char], *, dehyphenate: bool) -> list[Word]:
    out: list[Word] = []
    for block_no, block in enumerate(_blocks(chars, dehyphenate)):
        for line_no, line in enumerate(block):
            word_no = 0
            run: list[Char] = []
            for char in [*line.chars, None]:
                if char is not None and not char.text.isspace():
                    run.append(char)
                    continue
                if run:
                    box = _union([c.box for c in run])
                    out.append(Word(
                        box.x0, box.y0, box.x1, box.y1,
                        "".join(c.text for c in run), block_no, line_no, word_no,
                    ))
                    word_no += 1
                    run = []
    return out
