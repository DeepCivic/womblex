"""The pdfium text engine: characters in content order, rebuilt into the seam's shapes.

pdfium reports a page's text as characters with boxes and fonts, not as lines or
blocks, so this module does the segmentation MuPDF does natively. Phase 0
(`docs/decisions.md`) chose this over pdfminer's layout analysis on speed. The
segmentation here (`segment`) is pure — it takes `Char`s — so it is tested
without a PDF; `read_chars` is the only part that touches pdfium.

Reading order is content-stream order, as in MuPDF; a column jump starts a new
block but no reordering is attempted, so multi-column pages differ from MuPDF
exactly where Phase 0 measured the divergence tail.

Dehyphenation follows MuPDF's measured behaviour, which is to join nothing: the
locked MuPDF (1.27.2) keeps every line-end hyphen and its break under
`TEXT_DEHYPHENATE`, on synthetic pages and in the vendored corpus alike, so the
`dehyphenate` flag is accepted and changes nothing here either.

Font size is the effective size, the font's size scaled by the character's
matrix as MuPDF reports it: pdfium's own figure is the `Tf` operand, which is 1
wherever a producer scales text through the text matrix.
"""

from __future__ import annotations

import ctypes
import math
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
_LTR = (1.0, 0.0)


@dataclass(frozen=True)
class Char:
    """One character, its box in top-left page space, its font, and its writing
    direction there as a unit vector (left to right is ``(1, 0)``)."""

    text: str
    box: Rect
    size: float
    font: str
    bold: bool
    direction: tuple[float, float] = _LTR


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


def read_chars(
    page: pdfium.PdfPage, to_rect: Callable[[tuple[float, float, float, float]], Rect], clip: Rect,
) -> list[Char]:
    """Every printable character of *page* that touches *clip*, in content order.

    *to_rect* turns a pdfium user-space box into top-left page space, where
    *clip* is the page: MuPDF drops each character lying off it. Control
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
        matrix = pdfium_c.FS_MATRIX()
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
            box = to_rect(textpage.get_charbox(i, loose=True))
            if not (box.x1 > clip.x0 and box.x0 < clip.x1 and box.y1 > clip.y0 and box.y0 < clip.y1):
                continue
            length = pdfium_c.FPDFText_GetFontInfo(raw, i, name, len(name), ctypes.byref(flags))
            font = _SUBSET_PREFIX.sub("", name.value.decode("latin-1")) if length > 0 else ""
            weight = pdfium_c.FPDFText_GetFontWeight(raw, i)
            pdfium_c.FPDFText_GetMatrix(raw, i, ctypes.byref(matrix))
            scale = math.sqrt(abs(matrix.a * matrix.d - matrix.b * matrix.c))
            # User space is y-up, so the direction's y flips into page space.
            norm = math.hypot(matrix.a, matrix.b) or 1.0
            direction = (round(matrix.a / norm, 2) + 0.0, round(-matrix.b / norm, 2) + 0.0)
            out.append(Char(
                text=" " if text == "\t" else text,
                box=box,
                size=float(pdfium_c.FPDFText_GetFontSize(raw, i)) * scale,
                font=font,
                bold=_is_bold(weight, flags.value, font),
                direction=direction,
            ))
        return out
    finally:
        textpage.close()


def _extent(box: Rect, axis: tuple[float, float]) -> tuple[float, float]:
    """The interval *box* covers along *axis*."""
    ends = [x * axis[0] + y * axis[1] for x in (box.x0, box.x1) for y in (box.y0, box.y1)]
    return min(ends), max(ends)


def _same_line(line: _Line, char: Char) -> bool:
    """The character sits on the line's row, abutting it rather than across a gutter.

    Row and gap are measured in the writing direction, so rotated text forms lines too.
    """
    prev = line.chars[-1]
    if prev.direction != char.direction:
        return False
    along = char.direction
    across = (-along[1], along[0])
    (a0, a1), (b0, b1) = _extent(prev.box, across), _extent(char.box, across)
    overlap = min(a1, b1) - max(a0, b0)
    end, start = _extent(prev.box, along)[1], _extent(char.box, along)[0]
    return overlap > 0.5 * min(a1 - a0, b1 - b0) and end - char.size <= start <= end + _MAX_GAP * char.size


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


def _rejoin(lines: list[_Line], boxes: list[Rect], line: _Line, box: Rect) -> bool:
    """Merge a fragment into the recent line on its row that it abuts.

    *boxes* holds each line's box, kept in step so no line is re-measured.
    """
    size = line.chars[0].size
    if line.chars[0].direction != _LTR:
        return False
    for i in range(len(lines) - 1, max(len(lines) - 1 - _REJOIN_WINDOW, -1), -1):
        if lines[i].chars[0].direction != _LTR:
            continue
        other = boxes[i]
        overlap = min(box.y1, other.y1) - max(box.y0, other.y0)
        if overlap <= 0.5 * min(box.height, other.height):
            continue
        if -0.3 * size <= box.x0 - other.x1 <= _MAX_GAP * size:
            lines[i] = _join(lines[i], line)
        elif -0.3 * size <= other.x0 - box.x1 <= _MAX_GAP * size:
            lines[i] = _join(line, lines[i])
        else:
            continue
        boxes[i] = _union([other, box])
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
    boxes: list[Rect] = []
    for run in runs:
        box = run.box
        if not _rejoin(lines, boxes, run, box):
            lines.append(run)
            boxes.append(box)
    for line in lines:
        # Edge spaces are separators between layout runs, not content.
        while line.chars and line.chars[-1].text == " ":
            line.chars.pop()
        while line.chars and line.chars[0].text == " ":
            line.chars.pop(0)
    return [line for line in lines if line.chars]


def _same_block(prev: _Line, line: _Line) -> bool:
    if prev.chars[0].direction != _LTR or line.chars[0].direction != _LTR:
        return False
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


def plain_text(chars: list[Char]) -> str:
    return "".join(f"{line.text}\n" for block in segment(chars) for line in block)


def text_blocks(chars: list[Char]) -> list[Block]:
    return [
        Block(
            bbox=_union([line.box for line in block]), number=number,
            text="".join(f"{line.text}\n" for line in block),
        )
        for number, block in enumerate(segment(chars))
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
    """Blocks with lines and spans."""
    return [
        Block(
            bbox=_union([line.box for line in block]), number=number,
            lines=tuple(Line(bbox=line.box, spans=_spans(line)) for line in block),
        )
        for number, block in enumerate(segment(chars))
    ]


def words(chars: list[Char]) -> list[Word]:
    out: list[Word] = []
    for block_no, block in enumerate(segment(chars)):
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
