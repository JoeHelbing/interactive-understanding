"""Write context-pack crops from Docling document items."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal, TypeAlias
from urllib.parse import unquote

from docling_core.types.doc import (
    BoundingBox,
    CodeItem,
    CoordOrigin,
    DocItem,
    DoclingDocument,
    FormulaItem,
    PictureItem,
    Size,
    TableItem,
)
from PIL import Image
from pydantic import Field, field_validator  # type: ignore[attr-defined]
from pydantic.networks import AnyUrl

from interactive_understanding.docling_helpers import (
    VisualKind,
    reference_location,
    visual_kind,
)
from interactive_understanding.models import ContextPackModel

ImageFormat: TypeAlias = Literal["PNG", "WEBP"]
PixelBox: TypeAlias = tuple[int, int, int, int]
SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")
VISUAL_TYPES = (CodeItem, FormulaItem, PictureItem, TableItem)


class CropTarget(ContextPackModel):
    """A Docling item plus the naming metadata required by crop outputs."""

    kind: VisualKind
    label: str
    self_ref: str
    collection: str
    index: int
    page_no: int
    bbox: BoundingBox

    @classmethod
    def from_item(cls, item: DocItem) -> CropTarget | None:
        kind = visual_kind(item)
        if kind is None or not item.prov:
            return None
        collection, index = reference_location(item.self_ref)
        return cls(
            kind=kind,
            label=item.label.value,
            self_ref=item.self_ref,
            collection=collection,
            index=index,
            page_no=item.prov[0].page_no,
            bbox=item.prov[0].bbox,
        )

    def output_file(self, sequence: int, *, extension: str) -> str:
        ref_token = self.self_ref.removeprefix("#/").replace("/", "_")
        safe_ref = SAFE_FILENAME_RE.sub("_", ref_token).strip("_")
        token = safe_ref or f"{self.collection}_{self.index}"
        return f"p{self.page_no:03d}_{sequence:04d}_{self.kind}_{token}.{extension}"


class CropRecord(ContextPackModel):
    """Manifest entry for one written crop image."""

    sequence: int
    target: CropTarget
    page_box: BoundingBox
    pixel_box: PixelBox
    output_size: tuple[int, int]
    source_page_image_path: Path
    image_extension: str = "png"

    @property
    def output_file(self) -> str:
        return self.target.output_file(self.sequence, extension=self.image_extension)

    def manifest_record(self) -> dict[str, Any]:
        left, upper, right, lower = self.pixel_box
        return {
            "output_file": self.output_file,
            "kind": self.target.kind,
            "label": self.target.label,
            "self_ref": self.target.self_ref,
            "collection": self.target.collection,
            "index": self.target.index,
            "page_no": self.target.page_no,
            "bbox_l": self.target.bbox.l,
            "bbox_t": self.target.bbox.t,
            "bbox_r": self.target.bbox.r,
            "bbox_b": self.target.bbox.b,
            "bbox_coord_origin": self.target.bbox.coord_origin.value,
            "crop_left_px": left,
            "crop_upper_px": upper,
            "crop_right_px": right,
            "crop_lower_px": lower,
            "output_width_px": self.output_size[0],
            "output_height_px": self.output_size[1],
            "source_page_image": str(self.source_page_image_path),
        }


class DoclingCropper(ContextPackModel):
    """Apply context-pack crop policy to public Docling document objects."""

    output_dir: Path
    dpi: int = Field(default=150, gt=0)
    padding_points: float = Field(default=0, ge=0)
    image_format: ImageFormat = "PNG"

    @field_validator("image_format", mode="before")
    @classmethod
    def normalize_image_format(cls, value: Any) -> ImageFormat:
        normalized = str(value or "PNG").upper()
        if normalized not in {"PNG", "WEBP"}:
            raise ValueError(f"unsupported crop image format: {value!r}")
        return normalized  # type: ignore[return-value]

    @property
    def image_extension(self) -> str:
        return self.image_format.lower()

    def write(
        self, document: DoclingDocument, *, source_json: Path
    ) -> list[CropRecord]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        records = [
            self.write_crop(document, target, sequence)
            for sequence, target in enumerate(crop_targets(document), start=1)
        ]
        payload = {
            "source_json": str(source_json),
            "dpi": self.dpi,
            "image_format": self.image_format,
            "count": len(records),
            "crops": [record.manifest_record() for record in records],
        }
        (self.output_dir / "manifest.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
        )
        return records

    def write_crop(
        self,
        document: DoclingDocument,
        target: CropTarget,
        sequence: int,
    ) -> CropRecord:
        page = document.pages.get(target.page_no)
        if page is None or page.image is None:
            raise ValueError(f"page {target.page_no} has no image")
        source_path = image_path(page.image.uri)
        page_box = padded_page_box(
            target.bbox,
            page_size=page.size,
            padding_points=self.padding_points,
        )
        with Image.open(source_path) as page_image:
            source_image = page_image.convert("RGB")
            crop_box = pixel_box(
                page_box,
                page_size=page.size,
                image_size=source_image.size,
            )
            requested_size = output_pixel_size(page_box, dpi=self.dpi)
            crop_image = source_image.crop(crop_box)
            if crop_image.size != requested_size:
                crop_image = crop_image.resize(requested_size, Image.Resampling.LANCZOS)
            record = CropRecord(
                sequence=sequence,
                target=target,
                page_box=page_box,
                pixel_box=crop_box,
                output_size=requested_size,
                source_page_image_path=source_path,
                image_extension=self.image_extension,
            )
            self.save_crop(crop_image, self.output_dir / record.output_file)
        return record

    def save_crop(self, image: Image.Image, output_path: Path) -> None:
        if self.image_format == "WEBP":
            image.save(output_path, format="WEBP", lossless=True, quality=100, method=3)
        else:
            image.save(output_path, format="PNG", dpi=(self.dpi, self.dpi))


def crop_targets(document: DoclingDocument) -> list[CropTarget]:
    """Keep legacy collection order for stable crop filenames and manifests."""
    items: Iterable[DocItem] = (*document.texts, *document.pictures, *document.tables)
    return [
        target
        for item in items
        if isinstance(item, VISUAL_TYPES)
        if (target := CropTarget.from_item(item)) is not None
    ]


def padded_page_box(
    bbox: BoundingBox,
    *,
    page_size: Size,
    padding_points: float = 0,
) -> BoundingBox:
    top_left = bbox.to_top_left_origin(page_height=page_size.height)
    left, right = sorted((top_left.l, top_left.r))
    upper, lower = sorted((top_left.t, top_left.b))
    values = (
        max(0, left - padding_points),
        max(0, upper - padding_points),
        min(page_size.width, right + padding_points),
        min(page_size.height, lower + padding_points),
    )
    if values[2] <= values[0] or values[3] <= values[1]:
        raise ValueError(f"bbox produces an empty crop: {bbox}")
    return BoundingBox(
        l=values[0],
        t=values[1],
        r=values[2],
        b=values[3],
        coord_origin=CoordOrigin.TOPLEFT,
    )


def pixel_box(
    page_box: BoundingBox,
    *,
    page_size: Size,
    image_size: tuple[int, int],
) -> PixelBox:
    left, upper, right, lower = page_box.as_tuple()
    image_width, image_height = image_size
    result = (
        max(0, math.floor(left * image_width / page_size.width)),
        max(0, math.floor(upper * image_height / page_size.height)),
        min(image_width, math.ceil(right * image_width / page_size.width)),
        min(image_height, math.ceil(lower * image_height / page_size.height)),
    )
    if result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError(f"pixel box produces an empty crop: {result}")
    return result


def output_pixel_size(page_box: BoundingBox, *, dpi: int) -> tuple[int, int]:
    left, upper, right, lower = page_box.as_tuple()
    return (
        max(1, round((right - left) * dpi / 72)),
        max(1, round((lower - upper) * dpi / 72)),
    )


def image_path(uri: Path | AnyUrl) -> Path:
    if isinstance(uri, Path):
        return uri
    if uri.scheme == "file":
        return Path(unquote(uri.path))
    raise ValueError(f"Docling page image is not a local path: {uri}")
