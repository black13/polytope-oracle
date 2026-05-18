#!/usr/bin/env python3
"""
Pure standard-library PDF generator for fine line drawings on A4/A3.

No dependencies beyond Python 3.8+.  Generates valid PDF-1.4 files
that open correctly in any viewer on Windows, Linux, or macOS.

Paper sizes (PDF points, 72 pts/inch):
  A4 portrait : 595 x 842
  A3 portrait : 842 x 1191

PDF operators used:
  m   = moveto
  l   = lineto
  c   = cubic bezier (6 args: x1 y1 x2 y2 x3 y3)
  v   = cubic bezier to (4 args: x2 y2 x3 y3, start=current point)
  y   = cubic bezier to (4 args: x1 y1 x3 y3, start=current point)
  h   = closepath
  w   = linewidth
  G   = stroking gray level (0=black, 1=white)
  RG  = stroking RGB color
  S   = stroke path
  s   = close + stroke
"""

from __future__ import annotations

import math
import struct
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import BinaryIO, Sequence
import os

# ── paper sizes (PDF points = 1/72 inch) ──────────────────────────────────

POINTS_PER_MM = 72.0 / 25.4

A4_PORTRAIT  = (595.0, 842.0)
A4_LANDSCAPE = (842.0, 595.0)
A3_PORTRAIT  = (842.0, 1191.0)
A3_LANDSCAPE = (1191.0, 842.0)

PAPER = {
    "a4p": A4_PORTRAIT,
    "a4l": A4_LANDSCAPE,
    "a3p": A3_PORTRAIT,
    "a3l": A3_LANDSCAPE,
}


def mm_to_pts(mm: float) -> float:
    return mm * POINTS_PER_MM


# ── content stream builder ────────────────────────────────────────────────


@dataclass
class ContentStream:
    """Accumulate PDF content stream operators as text."""

    ops: list[str]

    def __init__(self) -> None:
        self.ops = []

    def emit(self, *tokens: str) -> None:
        self.ops.append(" ".join(tokens))

    def _fmt(self, value: float) -> str:
        return f"{value:.4f}"

    # ── path construction ──
    def move_to(self, x: float, y: float) -> None:
        self.emit(self._fmt(x), self._fmt(y), "m")

    def line_to(self, x: float, y: float) -> None:
        self.emit(self._fmt(x), self._fmt(y), "l")

    def curve_to(self, x1: float, y1: float,
                 x2: float, y2: float,
                 x3: float, y3: float) -> None:
        self.emit(self._fmt(x1), self._fmt(y1),
                  self._fmt(x2), self._fmt(y2),
                  self._fmt(x3), self._fmt(y3), "c")

    def close_path(self) -> None:
        self.emit("h")

    def circle(self, cx: float, cy: float, r: float) -> None:
        """Approximate a full circle with four cubic bezier segments.

        Uses the magic constant 0.5522847498 for a near-perfect circle.
        """
        k = 0.5522847498 * r
        self.move_to(cx + r, cy)
        self.curve_to(cx + r, cy + k, cx + k, cy + r, cx, cy + r)
        self.curve_to(cx - k, cy + r, cx - r, cy + k, cx - r, cy)
        self.curve_to(cx - r, cy - k, cx - k, cy - r, cx, cy - r)
        self.curve_to(cx + k, cy - r, cx + r, cy - k, cx + r, cy)
        self.close_path()

    def rect(self, x: float, y: float, w: float, h: float) -> None:
        """Append a rectangle to the current path."""
        self.emit(self._fmt(x), self._fmt(y),
                  self._fmt(w), self._fmt(h), "re")

    def clip(self) -> None:
        """Intersect current clip region with the current path (non-zero rule)."""
        self.emit("W")

    def clip_even_odd(self) -> None:
        """Intersect current clip region with the current path (even-odd rule)."""
        self.emit("W*")

    def end_path_no_op(self) -> None:
        """End path without painting (used after clip)."""
        self.emit("n")

    def clip_to_rect(self, x: float, y: float, w: float, h: float) -> None:
        """Convenience: set clip region to a rectangle."""
        self.rect(x, y, w, h)
        self.clip()
        self.end_path_no_op()

    # ── text ──
    def begin_text(self) -> None:
        self.emit("BT")

    def end_text(self) -> None:
        self.emit("ET")

    def set_font(self, name: str, size: float) -> None:
        """Select a standard PDF font by name and point size."""
        self.emit(f"/{name}", self._fmt(size), "Tf")

    def text_move_to(self, x: float, y: float) -> None:
        """Position the text cursor at (x, y)."""
        self.emit("1 0 0 1", self._fmt(x), self._fmt(y), "Tm")

    def show_text(self, text: str) -> None:
        """Show a text string, using PDF hex encoding for non-ASCII chars."""
        if all(32 <= ord(c) < 127 or c in "\n\r\t" for c in text):
            escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            self.emit(f"({escaped})", "Tj")
        else:
            # UTF-16BE with BOM as PDF hex string
            utf16 = text.encode("utf-16-be")
            hex_str = "<FEFF" + "".join(f"{b:02X}" for b in utf16) + ">"
            self.emit(hex_str, "Tj")

    def text_at(
        self, x: float, y: float, text: str,
        font: str = "F1", size: float = 12.0, gray: float = 0.0,
    ) -> None:
        """Convenience: draw text at (x, y)."""
        self.save_state()
        self.begin_text()
        self.set_font(font, size)
        self.text_move_to(x, y)
        self.set_fill_gray(gray)
        self.show_text(text)
        self.end_text()
        self.restore_state()

    # ── stroke control ──
    def set_line_width(self, pt: float) -> None:
        self.emit(self._fmt(pt), "w")

    def set_dash(self, pattern: Sequence[float], phase: float = 0.0) -> None:
        arr = "[" + " ".join(self._fmt(p) for p in pattern) + "]"
        self.emit(arr, self._fmt(phase), "d")

    def clear_dash(self) -> None:
        self.emit("[] 0 d")

    def set_line_cap(self, style: int = 0) -> None:
        """0=butt, 1=round, 2=projecting"""
        self.emit(str(style), "J")

    def set_line_join(self, style: int = 0) -> None:
        """0=miter, 1=round, 2=bevel"""
        self.emit(str(style), "j")

    # ── colour ──
    def set_stroke_gray(self, gray: float) -> None:
        self.emit(self._fmt(gray), "G")

    def set_fill_gray(self, gray: float) -> None:
        self.emit(self._fmt(gray), "g")

    def set_stroke_rgb(self, r: float, g: float, b: float) -> None:
        self.emit(self._fmt(r), self._fmt(g), self._fmt(b), "RG")

    def set_fill_rgb(self, r: float, g: float, b: float) -> None:
        self.emit(self._fmt(r), self._fmt(g), self._fmt(b), "rg")

    # ── painting ──
    def stroke(self) -> None:
        self.emit("S")

    def close_and_stroke(self) -> None:
        self.emit("s")

    def fill(self) -> None:
        self.emit("f")

    def fill_stroke(self) -> None:
        self.emit("B")

    # ── state ──
    def save_state(self) -> None:
        self.emit("q")

    def restore_state(self) -> None:
        self.emit("Q")

    # ── convenience ──
    def polyline(self, points: list[tuple[float, float]]) -> None:
        if not points:
            return
        self.move_to(*points[0])
        for p in points[1:]:
            self.line_to(*p)

    def stroked_polyline(
        self,
        points: list[tuple[float, float]],
        width: float = 0.5,
        gray: float = 0.0,
    ) -> None:
        self.save_state()
        self.set_line_width(width)
        self.set_stroke_gray(gray)
        self.polyline(points)
        self.stroke()
        self.restore_state()

    def parametric_bezier(
        self,
        f,
        t_start: float,
        t_end: float,
        segments: int = 12,
    ) -> None:
        """Emit a cubic bezier spline approximating a parametric curve.

        ``f(t)`` must return ``(x, y, dx_dt, dy_dt)`` — position and first
        derivative with respect to the parameter ``t``.  Control points are
        computed so each segment matches the exact tangent at both ends,
        giving C¹-continuous curvature with far fewer segments than a
        polyline approximation.

        The first segment emits ``m``; subsequent segments use ``c``.
        """
        if segments < 1:
            return
        dt = (t_end - t_start) / segments
        # first knot
        t = t_start
        x, y, dx, dy = f(t)
        self.move_to(x, y)
        # segment loop
        for _ in range(segments):
            # derive tangent vector in (x,y) space: derivative * param step
            tx0 = dx * dt / 3.0
            ty0 = dy * dt / 3.0
            t += dt
            x1, y1, dx1, dy1 = f(t)
            tx1 = dx1 * dt / 3.0
            ty1 = dy1 * dt / 3.0
            self.curve_to(
                x + tx0, y + ty0,        # control point 1
                x1 - tx1, y1 - ty1,      # control point 2
                x1, y1,                   # end point
            )
            x, y, dx, dy = x1, y1, dx1, dy1

    def grid_lines(
        self,
        x0: float, y0: float, x1: float, y1: float,
        nx: int, ny: int,
        width: float = 0.3,
        gray: float = 0.6,
    ) -> None:
        """Draw a rectangular grid of nx x ny cells."""
        dx = (x1 - x0) / nx
        dy = (y1 - y0) / ny
        self.save_state()
        self.set_line_width(width)
        self.set_stroke_gray(gray)
        for i in range(nx + 1):
            x = x0 + i * dx
            self.move_to(x, y0)
            self.line_to(x, y1)
            self.stroke()
        for j in range(ny + 1):
            y = y0 + j * dy
            self.move_to(x0, y)
            self.line_to(x1, y)
            self.stroke()
        self.restore_state()

    def build(self) -> str:
        return "\n".join(self.ops) + "\n"


# ── PDF document ───────────────────────────────────────────────────────────


@dataclass
class _ObjRef:
    num: int
    gen: int = 0

    def __str__(self) -> str:
        return f"{self.num} {self.gen} R"


@dataclass
class _Page:
    obj_num: int
    media_box: tuple[float, float, float, float]
    content_ref: _ObjRef
    font_refs: dict[str, _ObjRef]


class PurePDF:
    """Pure-Python PDF generator for line art on pre-sized paper.

    Usage::

        pdf = PurePDF("a4p")
        pdf.line_width(0.3)
        pdf.stroke_gray(0.0)
        pdf.polyline([(100, 100), (200, 300), (400, 200)])
        pdf.save("drawing.pdf")
    """

    objects: list[bytes]          # raw object body bytes
    obj_offsets: list[int]        # byte offset of each object into the file
    pages: list[_Page]
    content: ContentStream
    _fonts_used: dict[str, str]   # font-key -> base-font-name

    _page_width: float
    _page_height: float

    # map human-friendly names to PDF base font names
    STANDARD_FONTS: dict[str, str] = {
        "Times-Roman":      "Times-Roman",
        "Times-Bold":       "Times-Bold",
        "Times-Italic":     "Times-Italic",
        "Times-BoldItalic": "Times-BoldItalic",
        "Helvetica":        "Helvetica",
        "Helvetica-Bold":   "Helvetica-Bold",
        "Courier":          "Courier",
        "Courier-Bold":     "Courier-Bold",
        "Symbol":           "Symbol",
    }

    def __init__(self, paper: str = "a4p") -> None:
        w, h = PAPER[paper]
        self._page_width = w
        self._page_height = h
        self.objects = []
        self.obj_offsets = []
        self.pages = []
        self._fonts_used = {}
        self.content = ContentStream()

    # ── helpers for the content stream ──

    def line_width(self, pt: float) -> None:
        self.content.set_line_width(pt)

    def stroke_gray(self, gray: float) -> None:
        self.content.set_stroke_gray(gray)

    def stroke_rgb(self, r: float, g: float, b: float) -> None:
        self.content.set_stroke_rgb(r, g, b)

    def polyline(self, points: list[tuple[float, float]]) -> None:
        self.content.stroked_polyline(points)

    def raw_polyline(
        self,
        points: list[tuple[float, float]],
        width: float = 0.5,
        gray: float = 0.0,
    ) -> None:
        self.content.stroked_polyline(points, width=width, gray=gray)

    def move_to(self, x: float, y: float) -> None:
        self.content.move_to(x, y)

    def line_to(self, x: float, y: float) -> None:
        self.content.line_to(x, y)

    def stroke(self) -> None:
        self.content.stroke()

    def save_state(self) -> None:
        self.content.save_state()

    def restore_state(self) -> None:
        self.content.restore_state()

    def use_font(self, name: str) -> str:
        """Register a standard font and return its resource key (F1, F2, ...)."""
        base = self.STANDARD_FONTS.get(name, "Times-Roman")
        for key, bf in self._fonts_used.items():
            if bf == base:
                return key
        key = f"F{len(self._fonts_used) + 1}"
        self._fonts_used[key] = base
        return key

    def text(
        self, x: float, y: float, text_str: str,
        font: str = "Times-Roman", size: float = 12.0, gray: float = 0.0,
    ) -> None:
        key = self.use_font(font)
        self.content.text_at(x, y, text_str, font=key, size=size, gray=gray)

    def text_width_estimate(self, text_str: str, font_size: float = 12.0) -> float:
        """Rough width estimate (~0.5 * font_size per glyph)."""
        return 0.52 * font_size * len(text_str)

    def centered_text_x(self, text_str: str, font_size: float = 12.0) -> float:
        """Return x coordinate to center text on the page."""
        return 0.5 * (self._page_width - self.text_width_estimate(text_str, font_size))

    def draw_circle(
        self, cx: float, cy: float, r: float,
        fill_gray: float = 0.0, stroke: bool = False,
    ) -> None:
        self.content.save_state()
        self.content.set_fill_gray(fill_gray)
        if not stroke:
            self.content.set_stroke_gray(1.0)
        self.content.circle(cx, cy, r)
        self.content.fill_stroke() if stroke else self.content.fill()
        self.content.restore_state()

    def clip_rect(self, x: float, y: float, w: float, h: float) -> None:
        self.content.clip_to_rect(x, y, w, h)

    def draw_arrow(
        self, x0: float, y0: float, x1: float, y1: float,
        label: str = "", font: str = "Times-Roman", font_size: float = 10.0,
    ) -> None:
        dx = x1 - x0
        dy = y1 - y0
        length = math.hypot(dx, dy)
        if length < 1e-9:
            return
        ux, uy = dx / length, dy / length
        px, py = -uy, ux  # perpendicular

        head_len = 8.0
        head_wid = 3.5
        # shaft
        self.raw_polyline([(x0, y0), (x1, y1)], width=1.1)
        # arrowhead (filled triangle)
        tip = (x1, y1)
        base = (x1 - head_len * ux, y1 - head_len * uy)
        left = (base[0] + head_wid * px, base[1] + head_wid * py)
        right = (base[0] - head_wid * px, base[1] - head_wid * py)
        self.content.save_state()
        self.content.set_fill_gray(0.0)
        self.content.move_to(*tip)
        self.content.line_to(*left)
        self.content.line_to(*right)
        self.content.close_path()
        self.content.fill_stroke()
        self.content.restore_state()
        # label
        if label:
            lx = x1 + 5.0 * ux + 4.0 * py
            ly = y1 + 5.0 * uy + 4.0 * py
            self.text(lx, ly, label, font=font, size=font_size)

    def _add_object(self, body: bytes) -> _ObjRef:
        num = len(self.objects) + 1
        self.objects.append(body)
        return _ObjRef(num)

    # ── finalize and write ──

    def save(self, path: str | os.PathLike) -> None:
        self._finalize()
        with open(path, "wb") as f:
            self._write_header(f)
            for num, body in enumerate(self.objects, start=1):
                self.obj_offsets.append(f.tell())
                f.write(f"{num} 0 obj\n".encode())
                f.write(body)
                f.write(b"\nendobj\n")
            xref_offset = f.tell()
            self._write_xref(f)
            self._write_trailer(f, xref_offset)

    def _finalize(self) -> None:
        """Build the object list: catalog, pages, page, content stream."""
        if self.objects:
            return  # already finalized

        content_body = self.content.build()
        content_data = content_body.encode("latin-1")
        compressed = zlib.compress(content_data)
        content_obj = (
            b"<< /Length %d /Filter /FlateDecode >>\n"
            b"stream\n%s\nendstream"
        ) % (len(compressed), compressed)

        # build font resource dict
        if self._fonts_used:
            font_entries = " ".join(
                f"/{key} << /Type /Font /Subtype /Type1 /BaseFont /{base} >>"
                for key, base in self._fonts_used.items()
            )
            font_dict = f"<</Font << {font_entries} >>>>" 
        else:
            font_dict = "<</Font << >>>>"

        page_obj = (
            b"<< /Type /Page\n"
            b"   /Parent 2 0 R\n"
            b"   /MediaBox [0 0 %.1f %.1f]\n"
            b"   /Contents 4 0 R\n"
            b"   /Resources %s\n"
            b">>"
        ) % (self._page_width, self._page_height, font_dict.encode("ascii"))

        pages_obj = b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>"
        catalog_obj = b"<< /Type /Catalog /Pages 2 0 R >>"

        self.objects = [catalog_obj, pages_obj, page_obj, content_obj]

    def _write_header(self, f: BinaryIO) -> None:
        f.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")

    def _write_xref(self, f: BinaryIO) -> None:
        count = len(self.objects) + 1  # +1 for the free entry
        f.write(b"xref\n")
        f.write(f"0 {count}\n".encode())
        f.write(b"0000000000 65535 f \n")  # free object 0
        for offset in self.obj_offsets:
            f.write(f"{offset:010d} 00000 n \n".encode())

    def _write_trailer(self, f: BinaryIO, xref_offset: int) -> None:
        dt = (datetime.now(timezone.utc)
              .strftime("D:%Y%m%d%H%M%SZ")
              .encode())
        f.write(
            b"trailer\n"
            b"<< /Size %d /Root 1 0 R /Info << /CreationDate (%s) >> >>\n"
            b"startxref\n"
            b"%d\n"
            b"%%%%EOF"
            % (len(self.objects) + 1, dt, xref_offset)
        )


# ── utility: map from data space to page space ────────────────────────────


@dataclass
class Viewport:
    """Map data coordinates to page points with preserved aspect ratio."""

    page_x: float
    page_y: float
    page_w: float
    page_h: float

    data_x_min: float
    data_x_max: float
    data_y_min: float
    data_y_max: float

    scale: float
    offset_x: float
    offset_y: float

    @classmethod
    def fit(
        cls,
        page_x: float, page_y: float,
        page_w: float, page_h: float,
        data_points: list[tuple[float, float]],
        margin: float = 20.0,
    ) -> "Viewport":
        xs = [p[0] for p in data_points]
        ys = [p[1] for p in data_points]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        d_span = max(x_max - x_min, y_max - y_min, 1e-9)
        view_w = page_w - 2.0 * margin
        view_h = page_h - 2.0 * margin
        scale = min(view_w / d_span, view_h / d_span)

        offset_x = margin + 0.5 * (view_w - d_span * scale)
        offset_y = margin + 0.5 * (view_h - d_span * scale)

        return cls(
            page_x=page_x, page_y=page_y,
            page_w=page_w, page_h=page_h,
            data_x_min=x_min, data_x_max=x_max,
            data_y_min=y_min, data_y_max=y_max,
            scale=scale, offset_x=offset_x, offset_y=offset_y,
        )

    def map(self, x: float, y: float) -> tuple[float, float]:
        px = self.page_x + self.offset_x + (x - self.data_x_min) * self.scale
        py = self.page_y + self.offset_y + (y - self.data_y_min) * self.scale
        return (px, py)

    def map_points(self, points) -> list[tuple[float, float]]:
        return [self.map(p[0], p[1]) for p in points]


# ── rotation helpers (pure Python, no numpy needed for simple forms) ──────


def rot_x_matrix(angle_deg: float) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]:
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    return ((1, 0, 0), (0, c, -s), (0, s, c))


def rot_z_matrix(angle_deg: float) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]:
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    return ((c, -s, 0), (s, c, 0), (0, 0, 1))


def mat_vec_mul(
    m: tuple[tuple[float, float, float],
             tuple[float, float, float],
             tuple[float, float, float]],
    v: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    )


def mat_mul(
    a: tuple[tuple[float, float, float],
             tuple[float, float, float],
             tuple[float, float, float]],
    b: tuple[tuple[float, float, float],
             tuple[float, float, float],
             tuple[float, float, float]],
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]:
    return (
        (
            a[0][0] * b[0][0] + a[0][1] * b[1][0] + a[0][2] * b[2][0],
            a[0][0] * b[0][1] + a[0][1] * b[1][1] + a[0][2] * b[2][1],
            a[0][0] * b[0][2] + a[0][1] * b[1][2] + a[0][2] * b[2][2],
        ),
        (
            a[1][0] * b[0][0] + a[1][1] * b[1][0] + a[1][2] * b[2][0],
            a[1][0] * b[0][1] + a[1][1] * b[1][1] + a[1][2] * b[2][1],
            a[1][0] * b[0][2] + a[1][1] * b[1][2] + a[1][2] * b[2][2],
        ),
        (
            a[2][0] * b[0][0] + a[2][1] * b[1][0] + a[2][2] * b[2][0],
            a[2][0] * b[0][1] + a[2][1] * b[1][1] + a[2][2] * b[2][1],
            a[2][0] * b[0][2] + a[2][1] * b[1][2] + a[2][2] * b[2][2],
        ),
    )


# ── demo ──────────────────────────────────────────────────────────────────


def _demo() -> None:
    """Generate a demonstration PDF showing off fine line drawing capabilities."""
    pdf = PurePDF("a4p")

    w, h = A4_PORTRAIT

    # ── border ──
    pdf.content.save_state()
    pdf.content.set_line_width(0.8)
    pdf.content.set_stroke_gray(0.0)
    pdf.content.move_to(20, 20)
    pdf.content.line_to(w - 20, 20)
    pdf.content.line_to(w - 20, h - 20)
    pdf.content.line_to(20, h - 20)
    pdf.content.close_path()
    pdf.content.stroke()
    pdf.content.restore_state()

    # ── varied line widths ──
    y = h - 50
    pdf.content.save_state()
    for i, lw in enumerate([0.2, 0.4, 0.6, 0.8, 1.2, 1.8, 2.5]):
        pdf.content.set_line_width(lw)
        pdf.content.set_stroke_gray(0.0)
        px = 50 + i * 75
        pdf.content.move_to(px, y)
        pdf.content.line_to(px, y - 60)
        pdf.content.stroke()
    pdf.content.restore_state()

    # ── Lissajous curve as 12-segment cubic bezier spline ──
    def lissajous(t: float) -> tuple[float, float, float, float]:
        """x = A·sin(a·t+δ), y = B·cos(b·t)  with analytic derivatives."""
        return (
            300 + 120 * math.sin(3.0 * t + 0.4),   # x
            500 + 100 * math.cos(5.0 * t),           # y
            360 * math.cos(3.0 * t + 0.4),           # dx/dt
            -500 * math.sin(5.0 * t),                 # dy/dt
        )

    pdf.content.save_state()
    pdf.content.set_line_width(0.25)
    pdf.content.set_stroke_gray(0.15)
    pdf.content.parametric_bezier(lissajous, 0.0, 6.0 * math.pi, segments=24)
    pdf.content.stroke()
    pdf.content.restore_state()

    # ── bezier curve loop ──
    pdf.content.save_state()
    pdf.content.set_line_width(0.4)
    pdf.content.set_stroke_gray(0.0)
    pdf.content.move_to(400, 250)
    pdf.content.curve_to(450, 180, 520, 300, 440, 340)
    pdf.content.curve_to(360, 380, 320, 300, 400, 250)
    pdf.content.stroke()
    pdf.content.restore_state()

    # ── grid ──
    pdf.content.grid_lines(50, 120, 250, 220, 8, 8, width=0.2, gray=0.7)

    # ── dashed line ── style demonstration ──
    pdf.content.save_state()
    pdf.content.set_line_width(0.6)
    pdf.content.set_stroke_gray(0.3)
    pdf.content.set_dash([6.0, 3.0])
    pdf.content.move_to(400, 120)
    pdf.content.line_to(550, 170)
    pdf.content.stroke()
    pdf.content.clear_dash()
    pdf.content.restore_state()

    pdf.save(os.path.join(os.path.dirname(__file__),
                          "..", "output", "pure_pdf_demo.pdf"))
    print("Written output/pure_pdf_demo.pdf")


if __name__ == "__main__":
    _demo()
