#!/usr/bin/env python
"""Build a Docling-derived multimodal context pack.

The pack keeps text and visual evidence separate: normal text is written to a
Markdown document, while code blocks, formulas, pictures, and tables are linked
as visual artifacts and arranged into contact sheets. Page sheets are built from
Docling's rendered page images to preserve document structure without parsing the
source PDF directly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any, Literal, TypeAlias
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

from PIL import Image, ImageDraw, ImageFont
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,  # type: ignore[attr-defined]
    field_validator,  # type: ignore[attr-defined]
)
from rectpack import (  # type: ignore[import-not-found]
    SORT_AREA,
    MaxRectsBaf,
    MaxRectsBl,
    MaxRectsBlsf,
    MaxRectsBssf,
    PackingBin,
    PackingMode,
    newPacker,
)

from interactive_understanding.crop_docling_bboxes import (
    CropRecord,
    DoclingCropper,
    find_docling_json,
    resolve_image_uri,
)

VisualKind: TypeAlias = Literal["code", "formula", "picture", "table"]
SheetKind: TypeAlias = Literal["crop", "page"]

_log = logging.getLogger(__name__)


class FrozenModel(BaseModel):
    """Immutable base class for context-pack value objects."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class ImageSheetBudget(FrozenModel):
    """Pixel/patch budget for sheets sent through high-detail image intake."""

    max_dimension: int = Field(default=2048, gt=0)
    patch_size: int = Field(default=32, gt=0)
    max_patches: int = Field(default=2500, gt=0)

    def patch_count(self, width: int, height: int) -> int:
        return math.ceil(width / self.patch_size) * math.ceil(height / self.patch_size)

    def fits(self, width: int, height: int) -> bool:
        return (
            width <= self.max_dimension
            and height <= self.max_dimension
            and self.patch_count(width, height) <= self.max_patches
        )

    def constrain_dimensions(self, width: int, height: int) -> tuple[int, int]:
        """Return the largest same-aspect-ish dimensions inside the budget."""
        if self.fits(width, height):
            return width, height

        requested_aspect = width / height
        max_patch_dimension = math.ceil(self.max_dimension / self.patch_size)
        best_preferred: tuple[int, float, int, int] | None = None
        best_fallback: tuple[float, int, int, int] | None = None
        for patch_width in range(1, max_patch_dimension + 1):
            for patch_height in range(1, max_patch_dimension + 1):
                if patch_width * patch_height <= self.max_patches:
                    candidate_width = min(
                        width, self.max_dimension, patch_width * self.patch_size
                    )
                    candidate_height = min(
                        height, self.max_dimension, patch_height * self.patch_size
                    )
                    if self.fits(candidate_width, candidate_height):
                        candidate_aspect = candidate_width / candidate_height
                        aspect_delta = abs(candidate_aspect - requested_aspect)
                        relative_delta = aspect_delta / requested_aspect
                        area = candidate_width * candidate_height
                        fallback_score = (
                            aspect_delta,
                            -area,
                            candidate_width,
                            candidate_height,
                        )
                        if best_fallback is None or fallback_score < best_fallback:
                            best_fallback = fallback_score
                        if relative_delta <= 0.05:
                            preferred_score = (
                                -area,
                                aspect_delta,
                                candidate_width,
                                candidate_height,
                            )
                            if (
                                best_preferred is None
                                or preferred_score < best_preferred
                            ):
                                best_preferred = preferred_score
        if best_preferred is not None:
            return best_preferred[2], best_preferred[3]
        if best_fallback is not None:
            return best_fallback[2], best_fallback[3]
        message = "could not find sheet dimensions inside the image budget"
        raise ValueError(message)


class AcquiredPdf(FrozenModel):
    """Validated source PDF bytes copied into a context-pack directory."""

    pdf_path: Path
    source_name: str
    source_uri: str
    content_sha256: str
    byte_count: int


class PdfSource(FrozenModel):
    """Read a local path or URL and validate that the bytes are a PDF."""

    source: str | Path

    def write_pdf(self, output_dir: Path) -> AcquiredPdf:
        output_dir.mkdir(parents=True, exist_ok=True)
        pdf_bytes = self.read_bytes()
        validate_pdf_bytes(pdf_bytes)
        pdf_path = output_dir / "source.pdf"
        pdf_path.write_bytes(pdf_bytes)
        return AcquiredPdf(
            pdf_path=pdf_path,
            source_name=self.source_name,
            source_uri=str(self.source),
            content_sha256=hashlib.sha256(pdf_bytes).hexdigest(),
            byte_count=len(pdf_bytes),
        )

    def read_bytes(self) -> bytes:
        parsed = urlparse(str(self.source))
        if parsed.scheme in {"http", "https"}:
            request = Request(
                str(self.source), headers={"User-Agent": "iu-context-pack/0.1"}
            )
            with urlopen(request, timeout=60) as response:  # noqa: S310 - user-supplied public PDF URL.
                return response.read()
        if parsed.scheme == "file":
            return Path(unquote(parsed.path)).read_bytes()
        if parsed.scheme:
            message = f"unsupported PDF source scheme: {parsed.scheme}"
            raise ValueError(message)
        return Path(self.source).expanduser().read_bytes()

    @computed_field
    def source_name(self) -> str:
        parsed = urlparse(str(self.source))
        if parsed.scheme in {"http", "https"}:
            name = Path(unquote(parsed.path)).name
        elif parsed.scheme == "file":
            name = Path(unquote(parsed.path)).name
        else:
            name = Path(self.source).name
        if name:
            return name
        return "source.pdf"


def validate_pdf_bytes(pdf_bytes: bytes) -> None:
    if not pdf_bytes.startswith(b"%PDF-"):
        raise ValueError("input is not a PDF")


class DoclingReference(FrozenModel):
    """Parsed Docling JSON reference such as ``#/texts/12``."""

    ref: str
    collection: str
    index: int

    @classmethod
    def parse(cls, ref: str) -> DoclingReference:
        parts = ref.removeprefix("#/").split("/")
        if len(parts) != 2 or not parts[1].isdigit():
            message = f"Unsupported Docling reference: {ref}"
            raise ValueError(message)
        return cls(ref=ref, collection=parts[0], index=int(parts[1]))


class SheetPlacement(FrozenModel):
    """Location of an item inside a generated contact sheet."""

    self_ref: str
    label: str
    kind: str
    page_no: int | None = None
    source_path: Path
    bbox_px: tuple[int, int, int, int]
    label_bbox_px: tuple[int, int, int, int] | None = None
    border_bbox_px: tuple[int, int, int, int] | None = None

    @computed_field
    def width_px(self) -> int:
        return self.bbox_px[2] - self.bbox_px[0]

    @computed_field
    def height_px(self) -> int:
        return self.bbox_px[3] - self.bbox_px[1]

    @computed_field
    def right_edge(self) -> int:
        edges = [self.bbox_px[2]]
        if self.border_bbox_px is not None:
            edges.append(self.border_bbox_px[2])
        if self.label_bbox_px is not None:
            edges.append(self.label_bbox_px[2])
        return max(edges)

    @computed_field
    def lower_edge(self) -> int:
        edges = [self.bbox_px[3]]
        if self.border_bbox_px is not None:
            edges.append(self.border_bbox_px[3])
        if self.label_bbox_px is not None:
            edges.append(self.label_bbox_px[3])
        return max(edges)

    def manifest_record(self, *, output_dir: Path) -> dict[str, Any]:
        return {
            "self_ref": self.self_ref,
            "label": self.label,
            "kind": self.kind,
            "page_no": self.page_no,
            "source": relative_path(self.source_path, output_dir),
            "bbox_px": list(self.bbox_px),
            "label_bbox_px": maybe_bbox(self.label_bbox_px),
            "border_bbox_px": maybe_bbox(self.border_bbox_px),
            "width_px": self.width_px,
            "height_px": self.height_px,
        }


class ImageSheet(FrozenModel):
    """One generated image sheet and its item locations."""

    kind: SheetKind
    sheet_path: Path
    width_px: int
    height_px: int
    items: list[SheetPlacement]
    packing_algorithm: str | None = None

    def manifest_record(self, *, output_dir: Path) -> dict[str, Any]:
        record: dict[str, Any] = {
            "kind": self.kind,
            "sheet": relative_path(self.sheet_path, output_dir),
            "width_px": self.width_px,
            "height_px": self.height_px,
            "items": [
                item.manifest_record(output_dir=output_dir) for item in self.items
            ],
        }
        if self.packing_algorithm is not None:
            record["packing_algorithm"] = self.packing_algorithm
        return record


class CropSheetSource(FrozenModel):
    """Input crop image for the visual-artifact contact sheet."""

    self_ref: str
    label: str
    kind: VisualKind
    page_no: int
    source_path: Path

    @classmethod
    def from_crop_record(cls, record: CropRecord, *, crop_dir: Path) -> CropSheetSource:
        return cls(
            self_ref=record.target.self_ref,
            label=record.target.label,
            kind=record.target.kind,  # type: ignore[arg-type]
            page_no=record.target.page_no,
            source_path=crop_dir / record.output_file,
        )

    @computed_field
    def overlay_label(self) -> str:
        ref_index = DoclingReference.parse(self.self_ref).index
        return f"p{self.page_no:03d} {self.kind} {ref_index}"


class PageSheetSource(FrozenModel):
    """Input page image for the low-resolution page-structure sheet."""

    page_no: int
    source_path: Path

    @computed_field
    def self_ref(self) -> str:
        return f"#/pages/{self.page_no}"

    @computed_field
    def overlay_label(self) -> str:
        return f"p{self.page_no:03d}"


class PageImageSource(FrozenModel):
    """Input page image plus PDF-point dimensions for full-page export."""

    page_no: int
    source_path: Path
    width_points: float
    height_points: float

    def output_pixel_size(self, *, dpi: int) -> tuple[int, int]:
        width = max(1, round(self.width_points * dpi / 72))
        height = max(1, round(self.height_points * dpi / 72))
        return width, height


class PageImageRecord(FrozenModel):
    """Manifest entry for one exported full-page image."""

    page_no: int
    image_path: Path
    source_path: Path
    dpi: int
    width_px: int
    height_px: int

    def manifest_record(self, *, output_dir: Path) -> dict[str, Any]:
        return {
            "page_no": self.page_no,
            "image": relative_path(self.image_path, output_dir),
            "source": relative_path(self.source_path, output_dir),
            "dpi": self.dpi,
            "width_px": self.width_px,
            "height_px": self.height_px,
        }


class CropSheetSettings(FrozenModel):
    """Layout settings for densely packed visual crop sheets."""

    width: int = Field(default=2200, gt=0)
    height: int = Field(default=2600, gt=0)
    gutter: int = Field(default=4, ge=0)
    thumb_max_edge: int = Field(default=1800, gt=0)
    label_gap: int = Field(default=2, ge=0)
    label_padding_x: int = Field(default=6, ge=0)
    label_padding_y: int = Field(default=6, ge=0)
    label_font_size: int = Field(default=24, gt=0)
    border_px: int = Field(default=2, ge=0)
    sheet_budget: ImageSheetBudget = Field(default_factory=ImageSheetBudget)

    @computed_field
    def packing_width(self) -> int:
        return self.sheet_budget.constrain_dimensions(self.width, self.height)[0]

    @computed_field
    def packing_height(self) -> int:
        return self.sheet_budget.constrain_dimensions(self.width, self.height)[1]


class PageSheetSettings(FrozenModel):
    """Layout settings for regular page overview sheets."""

    columns: int = Field(default=4, gt=0)
    rows: int = Field(default=5, gt=0)
    gutter: int = Field(default=10, ge=0)
    thumb_height: int = Field(default=360, gt=0)
    border_px: int = Field(default=2, ge=0)
    label_gap: int = Field(default=2, ge=0)
    label_padding_x: int = Field(default=4, ge=0)
    label_padding_y: int = Field(default=3, ge=0)
    label_font_size: int = Field(default=18, gt=0)
    sheet_budget: ImageSheetBudget = Field(default_factory=ImageSheetBudget)

    @computed_field
    def pages_per_sheet(self) -> int:
        return self.columns * self.rows


class PageSheetGrid(FrozenModel):
    """Chosen grid for one page sheet under the image budget."""

    columns: int
    rows: int
    width_px: int
    height_px: int


class PackedRect(FrozenModel):
    """Top-left pixel rectangle selected by the sheet packing algorithm."""

    x: int
    y: int
    width: int
    height: int


class CropSheetItem(FrozenModel):
    """Prepared crop image plus the label strip that travels with it."""

    source: CropSheetSource
    image: Image.Image
    label_height: int
    label_gap: int
    border_px: int

    @computed_field
    def image_width(self) -> int:
        return self.image.width

    @computed_field
    def image_height(self) -> int:
        return self.image.height

    @computed_field
    def item_width(self) -> int:
        return self.image_width + 2 * self.border_px

    @computed_field
    def item_height(self) -> int:
        return (
            self.image_height + 2 * self.border_px + self.label_gap + self.label_height
        )

    @computed_field
    def area(self) -> int:
        return self.item_width * self.item_height


class CropSheetState:
    """Mutable packing state for one crop sheet."""

    def __init__(
        self,
        *,
        number: int,
        width: int,
        height: int,
        packing_algorithm: str,
    ) -> None:
        self.number = number
        self.packing_algorithm = packing_algorithm
        self.canvas = Image.new("RGB", (width, height), "#f2f2f2")
        self.placements: list[SheetPlacement] = []


class CropSheetBuilder(FrozenModel):
    """Write max-rects-packed sheets for bbox crop artifacts."""

    output_dir: Path
    settings: CropSheetSettings
    sources: list[CropSheetSource]

    def write(self) -> list[ImageSheet]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        items = self.sorted_items()
        self.validate_items_fit(items)
        states = self.pack_items(items)
        return [self._save_sheet(state) for state in states]

    def sorted_items(self) -> list[CropSheetItem]:
        items = [self.crop_sheet_item(source) for source in self.sources]
        return sorted(
            items,
            key=lambda item: (item.area, item.item_height, item.item_width),
            reverse=True,
        )

    def crop_sheet_item(self, source: CropSheetSource) -> CropSheetItem:
        label_height = self.label_height(source.overlay_label)
        image = self._thumbnail(source.source_path, label_height=label_height)
        return CropSheetItem(
            source=source,
            image=image,
            label_height=label_height,
            label_gap=self.settings.label_gap,
            border_px=self.settings.border_px,
        )

    def validate_items_fit(self, items: list[CropSheetItem]) -> None:
        inner_width = self.settings.packing_width - 2 * self.settings.gutter
        inner_height = self.settings.packing_height - 2 * self.settings.gutter
        for item in items:
            if item.item_width > inner_width or item.item_height > inner_height:
                message = f"crop item does not fit sheet: {item.source.source_path}"
                raise ValueError(message)

    def pack_items(self, items: list[CropSheetItem]) -> list[CropSheetState]:
        best_states: list[CropSheetState] | None = None
        best_score: tuple[int, int] | None = None
        for algorithm_name, pack_algorithm in self.pack_algorithms():
            states = self.pack_items_with_algorithm(
                items, algorithm_name=algorithm_name, pack_algorithm=pack_algorithm
            )
            score = self.layout_score(states)
            if best_score is None or score < best_score:
                best_score = score
                best_states = states
        if best_states is None:
            message = "rectpack failed to produce a crop-sheet layout"
            raise ValueError(message)
        return best_states

    def pack_algorithms(self) -> list[tuple[str, Any]]:
        return [
            ("rectpack-maxrects-baf", MaxRectsBaf),
            ("rectpack-maxrects-bssf", MaxRectsBssf),
            ("rectpack-maxrects-blsf", MaxRectsBlsf),
            ("rectpack-maxrects-bl", MaxRectsBl),
        ]

    def pack_items_with_algorithm(
        self, items: list[CropSheetItem], *, algorithm_name: str, pack_algorithm: Any
    ) -> list[CropSheetState]:
        states: dict[int, CropSheetState] = {}
        packer = newPacker(
            mode=PackingMode.Offline,
            bin_algo=PackingBin.Global,
            pack_algo=pack_algorithm,
            sort_algo=SORT_AREA,
            rotation=False,
        )
        for index, item in enumerate(items):
            packer.add_rect(item.item_width, item.item_height, rid=index)
        packer.add_bin(
            self.settings.packing_width - 2 * self.settings.gutter,
            self.settings.packing_height - 2 * self.settings.gutter,
            count=len(items),
        )
        packer.pack()
        for bin_index, x, y, width, height, item_index in packer.rect_list():
            state = states.get(bin_index)
            if state is None:
                state = self.new_state(bin_index + 1, algorithm_name=algorithm_name)
                states[bin_index] = state
            rect = PackedRect(
                x=x + self.settings.gutter,
                y=y + self.settings.gutter,
                width=width,
                height=height,
            )
            self.place_item(state, items[item_index], rect)
        if sum(len(state.placements) for state in states.values()) != len(items):
            message = "rectpack failed to place every crop item"
            raise ValueError(message)
        return [states[index] for index in sorted(states)]

    def layout_score(self, states: list[CropSheetState]) -> tuple[int, int]:
        return (len(states), sum(self.trimmed_area(state) for state in states))

    def trimmed_area(self, state: CropSheetState) -> int:
        width = max(item.right_edge for item in state.placements)
        width += self.settings.gutter
        height = max(item.lower_edge for item in state.placements)
        height += self.settings.gutter
        return width * height

    def new_state(self, number: int, *, algorithm_name: str) -> CropSheetState:
        return CropSheetState(
            number=number,
            width=self.settings.packing_width,
            height=self.settings.packing_height,
            packing_algorithm=algorithm_name,
        )

    def place_item(
        self, state: CropSheetState, item: CropSheetItem, rect: PackedRect
    ) -> None:
        x = rect.x
        y = rect.y
        border_bbox = (
            x,
            y,
            x + item.item_width,
            y + item.image_height + 2 * item.border_px,
        )
        ImageDraw.Draw(state.canvas).rectangle(border_bbox, fill="black")
        image_x = x + item.border_px
        image_y = y + item.border_px
        state.canvas.paste(item.image, (image_x, image_y))
        image_bbox = (
            image_x,
            image_y,
            image_x + item.image_width,
            image_y + item.image_height,
        )
        label_top = border_bbox[3] + item.label_gap
        label_bbox = (x, label_top, x + item.item_width, label_top + item.label_height)
        self._draw_label_below(
            state.canvas,
            label_bbox=label_bbox,
            label=item.source.overlay_label,
        )
        placement = SheetPlacement(
            self_ref=item.source.self_ref,
            label=item.source.label,
            kind=item.source.kind,
            page_no=item.source.page_no,
            source_path=item.source.source_path,
            bbox_px=image_bbox,
            label_bbox_px=label_bbox,
            border_bbox_px=border_bbox,
        )
        state.placements.append(placement)

    def _thumbnail(self, path: Path, *, label_height: int) -> Image.Image:
        with Image.open(path) as image:
            thumbnail = image.convert("RGB")
        max_width = (
            self.settings.packing_width
            - 2 * self.settings.gutter
            - 2 * self.settings.border_px
        )
        max_height = self.settings.packing_height - 2 * self.settings.gutter
        max_image_height = (
            max_height
            - 2 * self.settings.border_px
            - self.settings.label_gap
            - label_height
        )
        scale = min(
            1.0,
            self.settings.thumb_max_edge / thumbnail.width,
            self.settings.thumb_max_edge / thumbnail.height,
            max_width / thumbnail.width,
            max_image_height / thumbnail.height,
        )
        size = (
            max(1, math.floor(thumbnail.width * scale)),
            max(1, math.floor(thumbnail.height * scale)),
        )
        if size != thumbnail.size:
            thumbnail = thumbnail.resize(size, Image.Resampling.LANCZOS)
        return thumbnail

    def label_height(self, label: str) -> int:
        measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        text_bbox = measure.textbbox((0, 0), label, font=self.label_font())
        return text_bbox[3] - text_bbox[1] + 2 * self.settings.label_padding_y

    def _save_sheet(self, state: CropSheetState) -> ImageSheet:
        used_right = max(item.right_edge for item in state.placements)
        used_lower = max(item.lower_edge for item in state.placements)
        trimmed = state.canvas.crop(
            (
                0,
                0,
                min(state.canvas.width, used_right + self.settings.gutter),
                min(state.canvas.height, used_lower + self.settings.gutter),
            )
        )
        if not self.settings.sheet_budget.fits(trimmed.width, trimmed.height):
            message = (
                f"crop sheet exceeds image budget: {trimmed.width}x{trimmed.height}"
            )
            raise ValueError(message)
        sheet_path = self.output_dir / f"crop-sheet-{state.number:03d}.webp"
        save_webp(trimmed, sheet_path)
        return ImageSheet(
            kind="crop",
            sheet_path=sheet_path,
            width_px=trimmed.width,
            height_px=trimmed.height,
            items=state.placements,
            packing_algorithm=state.packing_algorithm,
        )

    def _draw_label_below(
        self, canvas: Image.Image, *, label_bbox: tuple[int, int, int, int], label: str
    ) -> None:
        draw = ImageDraw.Draw(canvas)
        draw.rectangle(label_bbox, fill="black")
        font = self.label_font()
        fitted_label = self.fitted_label(
            draw, label, width=label_bbox[2] - label_bbox[0], font=font
        )
        draw.text(
            (
                label_bbox[0] + self.settings.label_padding_x,
                label_bbox[1] + self.settings.label_padding_y,
            ),
            fitted_label,
            fill="white",
            font=font,
        )

    def fitted_label(
        self,
        draw: ImageDraw.ImageDraw,
        label: str,
        *,
        width: int,
        font: ImageFont.FreeTypeFont,
    ) -> str:
        available_width = max(1, width - 2 * self.settings.label_padding_x)
        fitted = label
        if self.text_width(draw, fitted, font=font) > available_width:
            fitted = self.truncated_label(draw, label, available_width, font=font)
        return fitted

    def truncated_label(
        self,
        draw: ImageDraw.ImageDraw,
        label: str,
        available_width: int,
        *,
        font: ImageFont.FreeTypeFont,
    ) -> str:
        ellipsis = "..."
        candidate = label
        while (
            candidate
            and self.text_width(draw, f"{candidate}{ellipsis}", font=font)
            > available_width
        ):
            candidate = candidate[:-1]
        if candidate:
            return f"{candidate}{ellipsis}"
        return ellipsis

    def text_width(
        self, draw: ImageDraw.ImageDraw, text: str, *, font: ImageFont.FreeTypeFont
    ) -> int:
        text_bbox = draw.textbbox((0, 0), text, font=font)
        return text_bbox[2] - text_bbox[0]

    def label_font(self) -> ImageFont.FreeTypeFont:
        return ImageFont.truetype("DejaVuSans.ttf", self.settings.label_font_size)


class PageSheetBuilder(FrozenModel):
    """Write regular low-resolution sheets of Docling-rendered pages."""

    output_dir: Path
    settings: PageSheetSettings
    sources: list[PageSheetSource]

    def write(self) -> list[ImageSheet]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        thumbnails = [
            (source, self._thumbnail(source.source_path)) for source in self.sources
        ]
        sheets: list[ImageSheet] = []
        start = 0
        sheet_number = 1
        while start < len(thumbnails):
            chunk, grid = self.next_sheet(thumbnails[start:])
            sheets.append(self._write_sheet(chunk, sheet_number, grid=grid))
            start += len(chunk)
            sheet_number += 1
        return sheets

    def next_sheet(
        self, thumbnails: list[tuple[PageSheetSource, Image.Image]]
    ) -> tuple[list[tuple[PageSheetSource, Image.Image]], PageSheetGrid]:
        limit = min(self.settings.pages_per_sheet, len(thumbnails))
        best_count = 0
        best_grid: PageSheetGrid | None = None
        for count in range(1, limit + 1):
            grid = self.best_grid(thumbnails[:count])
            if grid is None:
                break
            best_count = count
            best_grid = grid
        if best_grid is None:
            source_path = thumbnails[0][0].source_path if thumbnails else "unknown"
            message = f"page thumbnail does not fit image budget: {source_path}"
            raise ValueError(message)
        return thumbnails[:best_count], best_grid

    def best_grid(
        self, thumbnails: list[tuple[PageSheetSource, Image.Image]]
    ) -> PageSheetGrid | None:
        best_score: tuple[int, int, int, int] | None = None
        best_grid: PageSheetGrid | None = None
        max_columns = min(self.settings.columns, len(thumbnails))
        for columns in range(1, max_columns + 1):
            rows = math.ceil(len(thumbnails) / columns)
            if rows <= self.settings.rows:
                grid = self.grid_for(thumbnails, columns=columns, rows=rows)
                if self.settings.sheet_budget.fits(grid.width_px, grid.height_px):
                    score = (
                        grid.width_px * grid.height_px,
                        columns,
                        grid.width_px,
                        grid.height_px,
                    )
                    if best_score is None or score < best_score:
                        best_score = score
                        best_grid = grid
        return best_grid

    def grid_for(
        self,
        thumbnails: list[tuple[PageSheetSource, Image.Image]],
        *,
        columns: int,
        rows: int,
    ) -> PageSheetGrid:
        item_sizes = [
            self.item_size(source=source, image=image) for source, image in thumbnails
        ]
        cell_width = max(width for width, _ in item_sizes)
        cell_height = max(height for _, height in item_sizes)
        sheet_width = self.settings.gutter * (columns + 1)
        sheet_width += cell_width * columns
        sheet_height = self.settings.gutter * (rows + 1)
        sheet_height += cell_height * rows
        return PageSheetGrid(
            columns=columns,
            rows=rows,
            width_px=sheet_width,
            height_px=sheet_height,
        )

    def _write_sheet(
        self,
        thumbnails: list[tuple[PageSheetSource, Image.Image]],
        sheet_number: int,
        *,
        grid: PageSheetGrid,
    ) -> ImageSheet:
        item_sizes = [
            self.item_size(source=source, image=image) for source, image in thumbnails
        ]
        cell_width = max(width for width, _ in item_sizes)
        cell_height = max(height for _, height in item_sizes)
        canvas = Image.new("RGB", (grid.width_px, grid.height_px), "#e8e8e8")
        placements: list[SheetPlacement] = []

        for index, (source, image) in enumerate(thumbnails):
            item_width, _ = item_sizes[index]
            row = index // grid.columns
            column = index % grid.columns
            cell_x = self.settings.gutter + column * (cell_width + self.settings.gutter)
            cell_y = self.settings.gutter + row * (cell_height + self.settings.gutter)
            item_x = cell_x + (cell_width - item_width) // 2
            border_width = image.width + 2 * self.settings.border_px
            border_x = item_x + (item_width - border_width) // 2
            border_y = cell_y
            border_bbox = (
                border_x,
                border_y,
                border_x + border_width,
                border_y + image.height + 2 * self.settings.border_px,
            )
            ImageDraw.Draw(canvas).rectangle(border_bbox, fill="black")
            image_x = border_x + self.settings.border_px
            image_y = border_y + self.settings.border_px
            canvas.paste(image, (image_x, image_y))
            image_bbox = (
                image_x,
                image_y,
                image_x + image.width,
                image_y + image.height,
            )
            label_top = border_bbox[3] + self.settings.label_gap
            label_bbox = (
                item_x,
                label_top,
                item_x + item_width,
                label_top + self.label_height(source.overlay_label),
            )
            self._draw_label_below(
                canvas,
                label_bbox=label_bbox,
                label=source.overlay_label,
            )
            placements.append(
                SheetPlacement(
                    self_ref=source.self_ref,
                    label=source.overlay_label,
                    kind="page",
                    page_no=source.page_no,
                    source_path=source.source_path,
                    bbox_px=image_bbox,
                    label_bbox_px=label_bbox,
                    border_bbox_px=border_bbox,
                )
            )

        used_right = max(item.right_edge for item in placements)
        used_lower = max(item.lower_edge for item in placements)
        trimmed = canvas.crop(
            (
                0,
                0,
                min(canvas.width, used_right + self.settings.gutter),
                min(canvas.height, used_lower + self.settings.gutter),
            )
        )
        if not self.settings.sheet_budget.fits(trimmed.width, trimmed.height):
            message = (
                f"page sheet exceeds image budget: {trimmed.width}x{trimmed.height}"
            )
            raise ValueError(message)
        sheet_path = self.output_dir / f"page-sheet-{sheet_number:03d}.webp"
        save_webp(trimmed, sheet_path)
        return ImageSheet(
            kind="page",
            sheet_path=sheet_path,
            width_px=trimmed.width,
            height_px=trimmed.height,
            items=placements,
        )

    def item_size(
        self, *, source: PageSheetSource, image: Image.Image
    ) -> tuple[int, int]:
        border_width = image.width + 2 * self.settings.border_px
        label_width = (
            self.text_width(source.overlay_label) + 2 * self.settings.label_padding_x
        )
        item_width = max(border_width, label_width)
        item_height = (
            image.height
            + 2 * self.settings.border_px
            + self.settings.label_gap
            + self.label_height(source.overlay_label)
        )
        return item_width, item_height

    def _thumbnail(self, path: Path) -> Image.Image:
        with Image.open(path) as image:
            thumbnail = image.convert("RGB")
        width = max(
            1, round(thumbnail.width * self.settings.thumb_height / thumbnail.height)
        )
        size = (width, self.settings.thumb_height)
        if size != thumbnail.size:
            thumbnail = thumbnail.resize(size, Image.Resampling.LANCZOS)
        return thumbnail

    def label_height(self, label: str) -> int:
        text_bbox = self.label_font().getbbox(label)
        return text_bbox[3] - text_bbox[1] + 2 * self.settings.label_padding_y

    def _draw_label_below(
        self, canvas: Image.Image, *, label_bbox: tuple[int, int, int, int], label: str
    ) -> None:
        draw = ImageDraw.Draw(canvas)
        draw.rectangle(label_bbox, fill="black")
        font = self.label_font()
        x = label_bbox[0] + self.settings.label_padding_x
        y = label_bbox[1] + self.settings.label_padding_y
        draw.text((x, y), label, fill="white", font=font)

    def text_width(self, text: str) -> int:
        text_bbox = self.label_font().getbbox(text)
        return text_bbox[2] - text_bbox[0]

    def label_font(self) -> ImageFont.FreeTypeFont:
        return ImageFont.truetype("DejaVuSans.ttf", self.settings.label_font_size)


class PageImageExporter(FrozenModel):
    """Write per-page full-page images at a target DPI."""

    output_dir: Path
    dpi: int = Field(default=130, gt=0)
    sources: list[PageImageSource]

    def write(self) -> list[PageImageRecord]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return [self.write_page(source) for source in self.sources]

    def write_page(self, source: PageImageSource) -> PageImageRecord:
        output_size = source.output_pixel_size(dpi=self.dpi)
        image_path = self.output_dir / f"page-{source.page_no:03d}.webp"
        with Image.open(source.source_path) as image:
            page_image = image.convert("RGB")
        if page_image.size != output_size:
            page_image = page_image.resize(output_size, Image.Resampling.LANCZOS)
        save_webp(page_image, image_path)
        return PageImageRecord(
            page_no=source.page_no,
            image_path=image_path,
            source_path=source.source_path,
            dpi=self.dpi,
            width_px=output_size[0],
            height_px=output_size[1],
        )


class VisualLink(FrozenModel):
    """Text-side link to a visual crop and its sheet locations."""

    self_ref: str
    kind: VisualKind
    label: str
    page_no: int
    crop_path: Path
    crop_sheet_path: Path | None
    crop_sheet_bbox_px: tuple[int, int, int, int] | None
    page_sheet_path: Path | None
    page_sheet_bbox_px: tuple[int, int, int, int] | None

    def markdown_line(self, *, output_dir: Path) -> str:
        crop_link = relative_path(self.crop_path, output_dir)
        details = [
            f"page {self.page_no}",
            f"crop: `{crop_link}`",
        ]
        if self.crop_sheet_path is not None:
            details.append(
                f"crop sheet: `{relative_path(self.crop_sheet_path, output_dir)}`"
            )
        if self.crop_sheet_bbox_px is not None:
            details.append(f"crop sheet bbox px: {list(self.crop_sheet_bbox_px)}")
        if self.page_sheet_path is not None:
            details.append(
                f"page sheet: `{relative_path(self.page_sheet_path, output_dir)}`"
            )
        if self.page_sheet_bbox_px is not None:
            details.append(f"page sheet bbox px: {list(self.page_sheet_bbox_px)}")
        detail_text = "; ".join(details)
        return f"[visual: {self.kind} {self.self_ref}]({crop_link}) -- {detail_text}"

    def manifest_record(self, *, output_dir: Path) -> dict[str, Any]:
        return {
            "self_ref": self.self_ref,
            "kind": self.kind,
            "label": self.label,
            "page_no": self.page_no,
            "crop": relative_path(self.crop_path, output_dir),
            "crop_sheet": maybe_relative_path(self.crop_sheet_path, output_dir),
            "crop_sheet_bbox_px": maybe_bbox(self.crop_sheet_bbox_px),
            "page_sheet": maybe_relative_path(self.page_sheet_path, output_dir),
            "page_sheet_bbox_px": maybe_bbox(self.page_sheet_bbox_px),
        }


class DoclingContextSource(FrozenModel):
    """Docling JSON fields needed for text ordering and page sheets."""

    body: dict[str, Any] = Field(default_factory=dict)
    texts: list[dict[str, Any]] = Field(default_factory=list)
    pictures: list[dict[str, Any]] = Field(default_factory=list)
    tables: list[dict[str, Any]] = Field(default_factory=list)
    groups: list[dict[str, Any]] = Field(default_factory=list)
    pages: dict[str, dict[str, Any]] | list[dict[str, Any]] = Field(
        default_factory=dict
    )

    @classmethod
    def read(cls, json_path: Path) -> DoclingContextSource:
        return cls.model_validate_json(json_path.read_text(encoding="utf-8"))  # type: ignore[attr-defined]

    def visual_refs_in_body_order(self) -> list[str]:
        refs: list[str] = []
        for ref in self.iter_body_refs():
            if self.visual_kind(ref) is not None:
                refs.append(ref)
        return refs

    def iter_body_refs(self) -> list[str]:
        refs: list[str] = []
        self._append_child_refs(self.body.get("children", []), refs)
        return refs

    def _append_child_refs(
        self, children: list[dict[str, Any]], refs: list[str]
    ) -> None:
        for child in children:
            ref = child.get("$ref")
            if isinstance(ref, str):
                parsed = DoclingReference.parse(ref)
                if parsed.collection == "groups":
                    self._append_child_refs(self.children_for_ref(ref), refs)
                else:
                    refs.append(ref)

    def children_for_ref(self, ref: str) -> list[dict[str, Any]]:
        item = self.item_for_ref(ref)
        children = item.get("children", [])
        if isinstance(children, list):
            return children
        return []

    def item_for_ref(self, ref: str) -> dict[str, Any]:
        parsed = DoclingReference.parse(ref)
        collection = self.collection(parsed.collection)
        if parsed.index >= len(collection):
            message = f"Docling reference is out of range: {ref}"
            raise IndexError(message)
        return collection[parsed.index]

    def collection(self, name: str) -> list[dict[str, Any]]:
        if name == "texts":
            return self.texts
        if name == "pictures":
            return self.pictures
        if name == "tables":
            return self.tables
        if name == "groups":
            return self.groups
        return []

    def visual_kind(self, ref: str) -> VisualKind | None:
        parsed = DoclingReference.parse(ref)
        if parsed.collection == "pictures":
            return "picture"
        if parsed.collection == "tables":
            return "table"
        if parsed.collection == "texts":
            label = str(self.item_for_ref(ref).get("label", ""))
            if label in {"code", "formula"}:
                return label  # type: ignore[return-value]
        return None

    def page_sources(
        self, *, json_path: Path, image_root: Path | None
    ) -> list[PageSheetSource]:
        sources: list[PageSheetSource] = []
        for page_no, page in self.iter_pages():
            image_path = self.page_image_path(
                page_no=page_no, page=page, json_path=json_path, image_root=image_root
            )
            sources.append(PageSheetSource(page_no=page_no, source_path=image_path))
        return sources

    def page_image_sources(
        self, *, json_path: Path, image_root: Path | None
    ) -> list[PageImageSource]:
        sources: list[PageImageSource] = []
        for page_no, page in self.iter_pages():
            image_path = self.page_image_path(
                page_no=page_no, page=page, json_path=json_path, image_root=image_root
            )
            width_points, height_points = self.page_size_points(page, page_no=page_no)
            sources.append(
                PageImageSource(
                    page_no=page_no,
                    source_path=image_path,
                    width_points=width_points,
                    height_points=height_points,
                )
            )
        return sources

    def page_image_path(
        self,
        *,
        page_no: int,
        page: dict[str, Any],
        json_path: Path,
        image_root: Path | None,
    ) -> Path:
        image = page.get("image")
        if not isinstance(image, dict) or not image.get("uri"):
            message = f"page {page_no} has no Docling-rendered image"
            raise ValueError(message)
        image_path = resolve_image_uri(
            str(image["uri"]), json_path=json_path, image_root=image_root
        )
        if not image_path.exists():
            message = f"page {page_no} image not found: {image_path}"
            raise FileNotFoundError(message)
        return image_path

    def page_size_points(
        self, page: dict[str, Any], *, page_no: int
    ) -> tuple[float, float]:
        size = page.get("size")
        if not isinstance(size, dict):
            message = f"page {page_no} has no page size"
            raise ValueError(message)
        width = float(size.get("width") or size.get("w") or 0)
        height = float(size.get("height") or size.get("h") or 0)
        if width <= 0 or height <= 0:
            message = f"page {page_no} has invalid page size: {size}"
            raise ValueError(message)
        return width, height

    def iter_pages(self) -> list[tuple[int, dict[str, Any]]]:
        pages: list[tuple[int, dict[str, Any]]] = []
        if isinstance(self.pages, dict):
            for key, page in self.pages.items():
                page_no = int(page.get("page_no") or key)
                pages.append((page_no, page))
        else:
            for index, page in enumerate(self.pages, start=1):
                page_no = int(page.get("page_no") or index)
                pages.append((page_no, page))
        return sorted(pages, key=lambda item: item[0])


class MarkdownTextWriter(FrozenModel):
    """Write text-only Markdown with visual object links in reading order."""

    source: DoclingContextSource
    links_by_ref: dict[str, VisualLink]
    output_dir: Path

    def write(self, path: Path) -> None:
        lines = [
            "# Extracted text with visual links",
            "",
            "Code blocks, formulas, figures, and tables are linked as images rather than transcribed as text.",
            "",
        ]
        for ref in self.source.iter_body_refs():
            lines.extend(self.lines_for_ref(ref))
        path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    def lines_for_ref(self, ref: str) -> list[str]:
        visual_kind = self.source.visual_kind(ref)
        if visual_kind is not None:
            return self.visual_lines(ref, visual_kind)
        item = self.source.item_for_ref(ref)
        text = str(item.get("text") or item.get("orig") or "").strip()
        if not text:
            return []
        label = str(item.get("label", ""))
        if label == "page_footer":
            return ["", self.page_marker_for_item(item, text), "", "---", ""]
        if label == "section_header":
            return ["", self.heading_for_item(item, text), ""]
        if label == "list_item":
            return [f"- {text}"]
        return ["", text, ""]

    def visual_lines(self, ref: str, visual_kind: VisualKind) -> list[str]:
        link = self.links_by_ref.get(ref)
        if link is None:
            return ["", f"[visual missing: {visual_kind} {ref}]", ""]
        return ["", link.markdown_line(output_dir=self.output_dir), ""]

    def heading_for_item(self, item: dict[str, Any], text: str) -> str:
        raw_level = item.get("level") or 2
        level = min(6, max(2, int(raw_level) + 1))
        return f"{'#' * level} {text}"

    def page_marker_for_item(self, item: dict[str, Any], text: str) -> str:
        page_label = text or self.page_number_for_item(item)
        return f"< Page {page_label} >"

    def page_number_for_item(self, item: dict[str, Any]) -> str:
        provenance = item.get("prov")
        if isinstance(provenance, list) and provenance:
            page_no = provenance[0].get("page_no")
            if page_no is not None:
                return str(page_no)
        return "unknown"


class ContextPackResult(FrozenModel):
    """Paths and records produced by one context-pack run."""

    output_dir: Path
    text_path: Path
    manifest_path: Path
    visual_links: list[VisualLink]
    crop_sheets: list[ImageSheet]
    page_sheets: list[ImageSheet]
    page_images: list[PageImageRecord]

    @computed_field
    def visual_count(self) -> int:
        return len(self.visual_links)

    @computed_field
    def crop_sheet_paths(self) -> list[Path]:
        return [sheet.sheet_path for sheet in self.crop_sheets]

    @computed_field
    def page_sheet_paths(self) -> list[Path]:
        return [sheet.sheet_path for sheet in self.page_sheets]

    @computed_field
    def page_image_paths(self) -> list[Path]:
        return [record.image_path for record in self.page_images]


class ContextPackPaths(FrozenModel):
    """Output paths for one context-pack run."""

    output_dir: Path
    docling_output_dir: Path
    crop_dir: Path
    crop_sheet_dir: Path
    page_sheet_dir: Path
    page_image_dir: Path
    text_path: Path
    manifest_path: Path


def context_pack_paths(output_dir: Path, *, page_image_dpi: int) -> ContextPackPaths:
    return ContextPackPaths(
        output_dir=output_dir,
        docling_output_dir=output_dir / "docling",
        crop_dir=output_dir / "visual-crops",
        crop_sheet_dir=output_dir / "crop-sheets",
        page_sheet_dir=output_dir / "page-sheets",
        page_image_dir=output_dir / f"page-images-{page_image_dpi}dpi",
        text_path=output_dir / "text.md",
        manifest_path=output_dir / "manifest.json",
    )


class DoclingContextPack(FrozenModel):
    """Convert a PDF with Docling, then build visual sheets and text links."""

    pdf_path: Path
    output_dir: Path
    docling_command: str = "docling"
    run_docling_parse: bool = True
    docling_ocr_engine: str = "auto"
    docling_device: str = "auto"
    crop_dpi: int = Field(default=150, gt=0)
    crop_padding_points: float = Field(default=1, ge=0)
    image_root: Path | None = None
    crop_sheet_width: int = Field(default=2200, gt=0)
    crop_sheet_height: int = Field(default=2600, gt=0)
    crop_sheet_gutter: int = Field(default=4, ge=0)
    crop_thumb_max_edge: int = Field(default=1800, gt=0)
    page_sheet_columns: int = Field(default=4, gt=0)
    page_sheet_rows: int = Field(default=5, gt=0)
    page_sheet_gutter: int = Field(default=10, ge=0)
    page_thumb_height: int = Field(default=360, gt=0)
    page_image_dpi: int = Field(default=130, gt=0)
    sheet_max_dimension: int = Field(default=2048, gt=0)
    sheet_patch_size: int = Field(default=32, gt=0)
    sheet_max_patches: int = Field(default=2500, gt=0)

    @field_validator("pdf_path")
    @classmethod
    def require_pdf_input(cls, value: Path) -> Path:
        if value.suffix.lower() != ".pdf":
            message = f"context pack input must be a PDF: {value}"
            raise ValueError(message)
        return value

    def run(self) -> ContextPackResult:
        paths = context_pack_paths(self.output_dir, page_image_dpi=self.page_image_dpi)
        sheet_budget = ImageSheetBudget(
            max_dimension=self.sheet_max_dimension,
            patch_size=self.sheet_patch_size,
            max_patches=self.sheet_max_patches,
        )
        crop_sheet_settings = CropSheetSettings(
            width=self.crop_sheet_width,
            height=self.crop_sheet_height,
            gutter=self.crop_sheet_gutter,
            thumb_max_edge=self.crop_thumb_max_edge,
            sheet_budget=sheet_budget,
        )
        page_sheet_settings = PageSheetSettings(
            columns=self.page_sheet_columns,
            rows=self.page_sheet_rows,
            gutter=self.page_sheet_gutter,
            thumb_height=self.page_thumb_height,
            sheet_budget=sheet_budget,
        )

        self.prepare_output_dirs(paths)
        if self.run_docling_parse:
            self.run_docling(paths.docling_output_dir)
        json_path = find_docling_json(paths.docling_output_dir)
        self.convert_docling_images_to_webp(paths=paths, json_path=json_path)
        source = DoclingContextSource.read(json_path)
        crop_records = self.write_visual_crops(source, paths=paths, json_path=json_path)
        crop_sheets = self.write_crop_sheets(
            crop_records, paths=paths, settings=crop_sheet_settings
        )
        page_sheets = self.write_page_sheets(
            source, paths=paths, json_path=json_path, settings=page_sheet_settings
        )
        page_images = self.write_page_images(source, paths=paths, json_path=json_path)
        visual_links = self.visual_links(
            crop_records, crop_sheets, page_sheets, paths=paths
        )
        links_by_ref = {link.self_ref: link for link in visual_links}
        MarkdownTextWriter(
            source=source, links_by_ref=links_by_ref, output_dir=paths.output_dir
        ).write(paths.text_path)
        self.write_manifest(
            visual_links,
            crop_sheets,
            page_sheets,
            page_images,
            paths=paths,
            json_path=json_path,
        )
        return ContextPackResult(
            output_dir=paths.output_dir,
            text_path=paths.text_path,
            manifest_path=paths.manifest_path,
            visual_links=visual_links,
            crop_sheets=crop_sheets,
            page_sheets=page_sheets,
            page_images=page_images,
        )

    def prepare_output_dirs(self, paths: ContextPackPaths) -> None:
        paths.output_dir.mkdir(parents=True, exist_ok=True)
        generated_dirs = [
            paths.crop_dir,
            paths.crop_sheet_dir,
            paths.page_sheet_dir,
            paths.page_image_dir,
        ]
        if self.run_docling_parse:
            generated_dirs.insert(0, paths.docling_output_dir)
        for generated_dir in generated_dirs:
            if generated_dir.exists():
                shutil.rmtree(generated_dir)

    def run_docling(self, docling_output_dir: Path) -> None:
        docling_output_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(self.docling_args(docling_output_dir), check=True)

    def convert_docling_images_to_webp(
        self, *, paths: ContextPackPaths, json_path: Path
    ) -> None:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        self.rewrite_image_uris(payload, paths=paths, json_path=json_path)
        json_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def rewrite_image_uris(
        self, value: Any, *, paths: ContextPackPaths, json_path: Path
    ) -> None:
        if isinstance(value, dict):
            self.rewrite_image_uri(value, paths=paths, json_path=json_path)
            for child_value in value.values():
                self.rewrite_image_uris(child_value, paths=paths, json_path=json_path)
        if isinstance(value, list):
            for item in value:
                self.rewrite_image_uris(item, paths=paths, json_path=json_path)

    def rewrite_image_uri(
        self, value: dict[str, Any], *, paths: ContextPackPaths, json_path: Path
    ) -> None:
        uri = value.get("uri")
        if not isinstance(uri, str):
            return
        source_path = resolve_image_uri(uri, json_path=json_path, image_root=None)
        if source_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            return
        if not source_path.exists() or source_path.suffix.lower() == ".webp":
            return
        webp_path = source_path.with_suffix(".webp")
        with Image.open(source_path) as image:
            save_webp(image.convert("RGB"), webp_path)
        value["uri"] = self.rewritten_uri(uri, webp_path, json_path=json_path)
        if source_path.is_relative_to(paths.docling_output_dir):
            source_path.unlink()

    def rewritten_uri(
        self, original_uri: str, webp_path: Path, *, json_path: Path
    ) -> str:
        if Path(original_uri).is_absolute():
            return str(webp_path)
        return webp_path.relative_to(json_path.parent).as_posix()

    def docling_args(self, docling_output_dir: Path) -> list[str]:
        args = [
            self.docling_command,
            "convert",
            "--pipeline",
            "standard",
            "--device",
            self.docling_device,
            "--to",
            "md",
            "--to",
            "json",
            "--image-export-mode",
            "referenced",
            "--show-layout",
            "--save-profiling",
            "--output",
            str(docling_output_dir),
        ]
        args.extend(["--ocr", "--ocr-engine", self.docling_ocr_engine])
        args.append(str(self.pdf_path))
        return args

    def write_visual_crops(
        self, source: DoclingContextSource, *, paths: ContextPackPaths, json_path: Path
    ) -> list[CropRecord]:
        records = DoclingCropper(
            input_path=json_path,
            output_dir=paths.crop_dir,
            dpi=self.crop_dpi,
            padding_points=self.crop_padding_points,
            image_root=self.image_root,
            image_format="WEBP",
        ).run()
        return self.order_crop_records(records, source)

    def order_crop_records(
        self, records: list[CropRecord], source: DoclingContextSource
    ) -> list[CropRecord]:
        records_by_ref = {record.target.self_ref: record for record in records}
        ordered_refs = source.visual_refs_in_body_order()
        ordered_records = [
            records_by_ref[ref] for ref in ordered_refs if ref in records_by_ref
        ]
        used_refs = {record.target.self_ref for record in ordered_records}
        ordered_records.extend(
            record for record in records if record.target.self_ref not in used_refs
        )
        return ordered_records

    def write_crop_sheets(
        self,
        records: list[CropRecord],
        *,
        paths: ContextPackPaths,
        settings: CropSheetSettings,
    ) -> list[ImageSheet]:
        if not records:
            return []
        sources = [
            CropSheetSource.from_crop_record(record, crop_dir=paths.crop_dir)
            for record in records
        ]
        return CropSheetBuilder(
            output_dir=paths.crop_sheet_dir,
            settings=settings,
            sources=sources,
        ).write()

    def write_page_sheets(
        self,
        source: DoclingContextSource,
        *,
        paths: ContextPackPaths,
        json_path: Path,
        settings: PageSheetSettings,
    ) -> list[ImageSheet]:
        return PageSheetBuilder(
            output_dir=paths.page_sheet_dir,
            settings=settings,
            sources=source.page_sources(
                json_path=json_path, image_root=self.image_root
            ),
        ).write()

    def write_page_images(
        self, source: DoclingContextSource, *, paths: ContextPackPaths, json_path: Path
    ) -> list[PageImageRecord]:
        return PageImageExporter(
            output_dir=paths.page_image_dir,
            dpi=self.page_image_dpi,
            sources=source.page_image_sources(
                json_path=json_path, image_root=self.image_root
            ),
        ).write()

    def visual_links(
        self,
        records: list[CropRecord],
        crop_sheets: list[ImageSheet],
        page_sheets: list[ImageSheet],
        *,
        paths: ContextPackPaths,
    ) -> list[VisualLink]:
        crop_placements = placements_by_ref(crop_sheets)
        page_placements = placements_by_page(page_sheets)
        links: list[VisualLink] = []
        for record in records:
            crop_placement = crop_placements.get(record.target.self_ref)
            page_placement = page_placements.get(record.target.page_no)
            links.append(
                VisualLink(
                    self_ref=record.target.self_ref,
                    kind=record.target.kind,  # type: ignore[arg-type]
                    label=record.target.label,
                    page_no=record.target.page_no,
                    crop_path=paths.crop_dir / record.output_file,
                    crop_sheet_path=sheet_path_for(crop_sheets, crop_placement),
                    crop_sheet_bbox_px=crop_placement.bbox_px
                    if crop_placement
                    else None,
                    page_sheet_path=sheet_path_for(page_sheets, page_placement),
                    page_sheet_bbox_px=page_placement.bbox_px
                    if page_placement
                    else None,
                )
            )
        return links

    def write_manifest(
        self,
        visual_links: list[VisualLink],
        crop_sheets: list[ImageSheet],
        page_sheets: list[ImageSheet],
        page_images: list[PageImageRecord],
        *,
        paths: ContextPackPaths,
        json_path: Path,
    ) -> None:
        payload = {
            "source_pdf": relative_path(self.pdf_path, paths.output_dir),
            "source_docling_json": relative_path(json_path, paths.output_dir),
            "text": relative_path(paths.text_path, paths.output_dir),
            "visual_count": len(visual_links),
            "crop_sheets": [
                sheet.manifest_record(output_dir=paths.output_dir)
                for sheet in crop_sheets
            ],
            "page_sheets": [
                sheet.manifest_record(output_dir=paths.output_dir)
                for sheet in page_sheets
            ],
            "page_images": [
                record.manifest_record(output_dir=paths.output_dir)
                for record in page_images
            ],
            "visual_links": [
                link.manifest_record(output_dir=paths.output_dir)
                for link in visual_links
            ],
        }
        paths.manifest_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


def save_webp(image: Image.Image, path: Path) -> None:
    image.save(path, format="WEBP", lossless=True, quality=100, method=3)


def placements_by_ref(sheets: list[ImageSheet]) -> dict[str, SheetPlacement]:
    placements: dict[str, SheetPlacement] = {}
    for sheet in sheets:
        for item in sheet.items:
            placements[item.self_ref] = item
    return placements


def placements_by_page(sheets: list[ImageSheet]) -> dict[int, SheetPlacement]:
    placements: dict[int, SheetPlacement] = {}
    for sheet in sheets:
        for item in sheet.items:
            if item.page_no is not None:
                placements[item.page_no] = item
    return placements


def sheet_path_for(
    sheets: list[ImageSheet], placement: SheetPlacement | None
) -> Path | None:
    if placement is None:
        return None
    for sheet in sheets:
        if placement in sheet.items:
            return sheet.sheet_path
    return None


def relative_path(path: Path, base_dir: Path) -> str:
    try:
        return path.relative_to(base_dir).as_posix()
    except ValueError:
        return str(path)


def maybe_relative_path(path: Path | None, base_dir: Path) -> str | None:
    if path is None:
        return None
    return relative_path(path, base_dir)


def maybe_bbox(bbox: tuple[int, int, int, int] | None) -> list[int] | None:
    if bbox is None:
        return None
    return list(bbox)


class ContextPackSettings(FrozenModel):
    """Docling and sheet-generation options shared by all pack entrypoints."""

    docling_command: str = "docling"
    docling_ocr_engine: str = "auto"
    docling_device: str = "auto"
    crop_dpi: int = Field(default=150, gt=0)
    crop_padding_points: float = Field(default=1, ge=0)
    crop_sheet_width: int = Field(default=2200, gt=0)
    crop_sheet_height: int = Field(default=2600, gt=0)
    crop_sheet_gutter: int = Field(default=4, ge=0)
    crop_thumb_max_edge: int = Field(default=1800, gt=0)
    page_sheet_columns: int = Field(default=4, gt=0)
    page_sheet_rows: int = Field(default=5, gt=0)
    page_sheet_gutter: int = Field(default=10, ge=0)
    page_thumb_height: int = Field(default=360, gt=0)
    page_image_dpi: int = Field(default=130, gt=0)
    sheet_max_dimension: int = Field(default=2048, gt=0)
    sheet_patch_size: int = Field(default=32, gt=0)
    sheet_max_patches: int = Field(default=2500, gt=0)

    def context_pack(self, *, pdf_path: Path, output_dir: Path) -> DoclingContextPack:
        return DoclingContextPack(
            pdf_path=pdf_path,
            output_dir=output_dir,
            **self.model_dump(),
        )


class ContextPackWorkflow(FrozenModel):
    """Acquire a PDF source, run Docling, and build a context pack."""

    source: str | Path
    output_dir: Path
    settings: ContextPackSettings = Field(default_factory=ContextPackSettings)

    def run(self) -> ContextPackResult:
        acquired = PdfSource(source=self.source).write_pdf(self.output_dir)
        result = self.settings.context_pack(
            pdf_path=acquired.pdf_path,
            output_dir=self.output_dir,
        ).run()
        write_source_metadata(acquired, self.output_dir)
        return result


def write_source_metadata(acquired: AcquiredPdf, output_dir: Path) -> None:
    payload = {
        "source_name": acquired.source_name,
        "source_pdf": relative_path(acquired.pdf_path, output_dir),
        "content_sha256": acquired.content_sha256,
        "byte_count": acquired.byte_count,
    }
    (output_dir / "source.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("source", help="local PDF path, file URL, or direct PDF URL")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--docling-command", default="docling")
    parser.add_argument(
        "--docling-device",
        default="auto",
        help="Docling accelerator device: auto, cpu, cuda, mps, or xpu",
    )
    parser.add_argument(
        "--docling-ocr-engine",
        default="auto",
        help="Docling OCR engine for bitmap fallback",
    )
    parser.add_argument("--crop-dpi", type=int, default=150)
    parser.add_argument("--crop-padding-points", type=float, default=1)
    parser.add_argument("--crop-sheet-width", type=int, default=2200)
    parser.add_argument("--crop-sheet-height", type=int, default=2600)
    parser.add_argument("--crop-sheet-gutter", type=int, default=4)
    parser.add_argument("--crop-thumb-max-edge", type=int, default=1800)
    parser.add_argument("--page-sheet-columns", type=int, default=4)
    parser.add_argument("--page-sheet-rows", type=int, default=5)
    parser.add_argument("--page-sheet-gutter", type=int, default=10)
    parser.add_argument("--page-thumb-height", type=int, default=360)
    parser.add_argument(
        "--page-image-dpi",
        type=int,
        default=130,
        help="DPI for exported full-page images in page-images-<dpi>dpi/",
    )
    parser.add_argument(
        "--sheet-max-dimension",
        type=int,
        default=2048,
        help="maximum output sheet width or height before model-side resizing",
    )
    parser.add_argument(
        "--sheet-patch-size",
        type=int,
        default=32,
        help="image-intake patch size used for sheet budget checks",
    )
    parser.add_argument(
        "--sheet-max-patches",
        type=int,
        default=2500,
        help="maximum image-intake patches allowed per output sheet",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON result")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def settings_from_args(args: argparse.Namespace) -> ContextPackSettings:
    return ContextPackSettings.model_validate(vars(args))  # type: ignore[attr-defined]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        settings = settings_from_args(args)
        result = ContextPackWorkflow(
            source=args.source,
            output_dir=args.output_dir,
            settings=settings,
        ).run()
    except Exception as exc:  # noqa: BLE001 - CLI boundary normalizes errors.
        if args.json:
            print(json.dumps({"status": "error", "error": str(exc)}, indent=2))
        else:
            print(f"iu-context-pack failed: {exc}")
        return 1
    if args.json:
        print(json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True))
    else:
        print(
            "Wrote "
            f"{result.visual_count} visual links, "
            f"{len(result.crop_sheets)} crop sheets, "
            f"{len(result.page_sheets)} page sheets, "
            f"{len(result.page_images)} page images to {result.output_dir}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
