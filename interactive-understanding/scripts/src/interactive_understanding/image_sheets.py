"""Image budgets, contact sheets, and page-image exports."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal, TypeAlias

from PIL import Image, ImageDraw, ImageFont
from pydantic import Field, computed_field  # type: ignore[attr-defined]
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

from interactive_understanding.docling_helpers import VisualKind, reference_location
from interactive_understanding.models import ContextPackModel as FrozenModel

SheetKind: TypeAlias = Literal["crop", "page"]


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

    @computed_field
    def overlay_label(self) -> str:
        _collection, ref_index = reference_location(self.self_ref)
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


class SheetLabelStyle(FrozenModel):
    """Measure and draw the label strip shared by both sheet layouts."""

    padding_x: int = Field(ge=0)
    padding_y: int = Field(ge=0)
    font_size: int = Field(gt=0)

    @property
    def font(self) -> ImageFont.FreeTypeFont:
        return ImageFont.truetype("DejaVuSans.ttf", self.font_size)

    def height(self, label: str) -> int:
        text_bbox = self.font.getbbox(label)
        return text_bbox[3] - text_bbox[1] + 2 * self.padding_y

    def width(self, text: str) -> int:
        text_bbox = self.font.getbbox(text)
        return text_bbox[2] - text_bbox[0]

    def draw(
        self,
        canvas: Image.Image,
        *,
        label_bbox: tuple[int, int, int, int],
        label: str,
    ) -> None:
        self._draw(canvas, label_bbox=label_bbox, label=label, fit=False)

    def draw_fitted(
        self,
        canvas: Image.Image,
        *,
        label_bbox: tuple[int, int, int, int],
        label: str,
    ) -> None:
        self._draw(canvas, label_bbox=label_bbox, label=label, fit=True)

    def _draw(
        self,
        canvas: Image.Image,
        *,
        label_bbox: tuple[int, int, int, int],
        label: str,
        fit: bool,
    ) -> None:
        draw = ImageDraw.Draw(canvas)
        draw.rectangle(label_bbox, fill="black")
        rendered_label = label
        if fit:
            rendered_label = self.fitted(label, width=label_bbox[2] - label_bbox[0])
        draw.text(
            (label_bbox[0] + self.padding_x, label_bbox[1] + self.padding_y),
            rendered_label,
            fill="white",
            font=self.font,
        )

    def fitted(self, label: str, *, width: int) -> str:
        available_width = max(1, width - 2 * self.padding_x)
        if self.width(label) <= available_width:
            return label
        return self.truncated(label, available_width=available_width)

    def truncated(self, label: str, *, available_width: int) -> str:
        ellipsis = "..."
        candidate = label
        while candidate and self.width(f"{candidate}{ellipsis}") > available_width:
            candidate = candidate[:-1]
        if candidate:
            return f"{candidate}{ellipsis}"
        return ellipsis


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

    @property
    def label_style(self) -> SheetLabelStyle:
        return SheetLabelStyle(
            padding_x=self.label_padding_x,
            padding_y=self.label_padding_y,
            font_size=self.label_font_size,
        )

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

    @property
    def label_style(self) -> SheetLabelStyle:
        return SheetLabelStyle(
            padding_x=self.label_padding_x,
            padding_y=self.label_padding_y,
            font_size=self.label_font_size,
        )

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
        label_height = self.settings.label_style.height(source.overlay_label)
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
        self.settings.label_style.draw_fitted(
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
            source_path: Path | str = "unknown"
            if thumbnails:
                source_path = thumbnails[0][0].source_path
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
                label_top + self.settings.label_style.height(source.overlay_label),
            )
            self.settings.label_style.draw(
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
            self.settings.label_style.width(source.overlay_label)
            + 2 * self.settings.label_padding_x
        )
        item_width = max(border_width, label_width)
        item_height = (
            image.height
            + 2 * self.settings.border_px
            + self.settings.label_gap
            + self.settings.label_style.height(source.overlay_label)
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


def save_webp(image: Image.Image, path: Path) -> None:
    image.save(path, format="WEBP", lossless=True, quality=100, method=3)


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
