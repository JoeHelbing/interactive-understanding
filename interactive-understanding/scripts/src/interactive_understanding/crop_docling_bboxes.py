#!/usr/bin/env python
"""Crop bbox-bearing Docling objects from rendered page images.

The script reads a Docling JSON export, finds code blocks, formulas, pictures,
and tables, crops each object's provenance bbox from the referenced page image,
and writes image crops with 150 DPI metadata by default.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any, ClassVar, Literal, TypeAlias
from urllib.parse import urlparse

from PIL import Image
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,  # type: ignore[attr-defined]
    model_validator,  # type: ignore[attr-defined]
)
from typing_extensions import Self

TargetKind: TypeAlias = Literal["code", "formula", "picture", "table"]
CoordinateOrigin: TypeAlias = Literal["BOTTOMLEFT", "TOPLEFT"]
ImageFormat: TypeAlias = Literal["PNG", "WEBP"]

SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_log = logging.getLogger(__name__)


class FrozenModel(BaseModel):
    """Immutable base class for the cropper's small value objects."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class DoclingBBox(FrozenModel):
    """Docling bbox in page points with its coordinate origin preserved."""

    left: float = Field(alias="l")
    top: float = Field(alias="t")
    right: float = Field(alias="r")
    bottom: float = Field(alias="b")
    coord_origin: CoordinateOrigin = "BOTTOMLEFT"

    @field_validator("coord_origin", mode="before")
    @classmethod
    def normalize_coord_origin(cls, value: Any) -> CoordinateOrigin:
        normalized = str(value or "BOTTOMLEFT").upper()
        if normalized not in {"BOTTOMLEFT", "TOPLEFT"}:
            message = f"Unsupported bbox coord_origin: {value!r}"
            raise ValueError(message)
        return normalized  # type: ignore[return-value]

    @property
    def left_points(self) -> float:
        return min(self.left, self.right)

    @property
    def right_points(self) -> float:
        return max(self.left, self.right)

    @property
    def lower_source_points(self) -> float:
        return min(self.top, self.bottom)

    @property
    def upper_source_points(self) -> float:
        return max(self.top, self.bottom)

    def to_page_box(
        self, *, page_width: float, page_height: float, padding_points: float = 0
    ) -> PageBox:
        """Return the bbox as a top-left-origin page box in PDF points."""
        if self.coord_origin == "BOTTOMLEFT":
            upper_points = page_height - self.upper_source_points
            lower_points = page_height - self.lower_source_points
        else:
            upper_points = self.lower_source_points
            lower_points = self.upper_source_points
        return PageBox(
            left=clamp(self.left_points - padding_points, 0, page_width),
            upper=clamp(upper_points - padding_points, 0, page_height),
            right=clamp(self.right_points + padding_points, 0, page_width),
            lower=clamp(lower_points + padding_points, 0, page_height),
        )


class PageBox(FrozenModel):
    """Top-left-origin crop rectangle in PDF page points."""

    left: float
    upper: float
    right: float
    lower: float

    @model_validator(mode="after")
    def require_positive_area(self) -> Self:
        if self.right <= self.left or self.lower <= self.upper:
            message = f"bbox produces an empty crop: {self}"
            raise ValueError(message)
        return self

    @property
    def width_points(self) -> float:
        return self.right - self.left

    @property
    def height_points(self) -> float:
        return self.lower - self.upper

    def to_pixel_box(
        self,
        *,
        image_width: int,
        image_height: int,
        page_width: float,
        page_height: float,
    ) -> PixelBox:
        """Scale the page-point box into source page-image pixels."""
        scale_x = image_width / page_width
        scale_y = image_height / page_height
        return PixelBox(
            left=max(0, math.floor(self.left * scale_x)),
            upper=max(0, math.floor(self.upper * scale_y)),
            right=min(image_width, math.ceil(self.right * scale_x)),
            lower=min(image_height, math.ceil(self.lower * scale_y)),
        )

    def output_pixel_size(self, *, dpi: int) -> tuple[int, int]:
        """Return crop dimensions for the requested output DPI."""
        width = max(1, round(self.width_points * dpi / 72))
        height = max(1, round(self.height_points * dpi / 72))
        return width, height


class PixelBox(FrozenModel):
    """Pixel rectangle used with Pillow's crop API."""

    left: int
    upper: int
    right: int
    lower: int

    @model_validator(mode="after")
    def require_positive_area(self) -> Self:
        if self.right <= self.left or self.lower <= self.upper:
            message = f"pixel box produces an empty crop: {self}"
            raise ValueError(message)
        return self

    @property
    def pillow_box(self) -> tuple[int, int, int, int]:
        return (self.left, self.upper, self.right, self.lower)


class DoclingProvenance(FrozenModel):
    """First-class model for Docling provenance entries used by crops."""

    page_no: int
    bbox: DoclingBBox


class DoclingItem(FrozenModel):
    """Docling object with optional provenance and arbitrary extra fields."""

    self_ref: str | None = None
    label: str = ""
    prov: list[DoclingProvenance] = Field(default_factory=list)

    @property
    def first_provenance(self) -> DoclingProvenance | None:
        if not self.prov:
            return None
        return self.prov[0]

    def crop_target(
        self, *, kind: TargetKind, collection: str, index: int
    ) -> CropTarget | None:
        provenance = self.first_provenance
        if provenance is None:
            return None
        return CropTarget(
            kind=kind,
            label=self.label,
            self_ref=self.self_ref or f"#/{collection}/{index}",
            collection=collection,
            index=index,
            page_no=provenance.page_no,
            bbox=provenance.bbox,
        )


class PageSize(FrozenModel):
    """PDF page dimensions in points."""

    width: float
    height: float


class PageImageRef(FrozenModel):
    """Rendered page image reference from the Docling page map."""

    uri: str | None = None


class DoclingPage(FrozenModel):
    """Docling page metadata needed to crop page bboxes."""

    size: PageSize
    image: PageImageRef | None = None
    page_no: int | None = None

    def image_path(self, *, json_path: Path, image_root: Path | None) -> Path:
        if self.image is None or not self.image.uri:
            page_label = self.page_no or "unknown"
            message = f"page {page_label} has no image"
            raise ValueError(message)
        return resolve_image_uri(
            self.image.uri, json_path=json_path, image_root=image_root
        )


class DoclingDocument(FrozenModel):
    """Parsed Docling document with only fields needed by the cropper."""

    text_target_labels: ClassVar[frozenset[str]] = frozenset(("code", "formula"))
    texts: list[DoclingItem] = Field(default_factory=list)
    pictures: list[DoclingItem] = Field(default_factory=list)
    tables: list[DoclingItem] = Field(default_factory=list)
    pages: dict[str, DoclingPage] | list[DoclingPage] = Field(default_factory=dict)

    def crop_targets(self) -> list[CropTarget]:
        """Return code, formula, picture, and table targets in document order."""
        targets: list[CropTarget] = []
        targets.extend(self._targets_from_texts())
        targets.extend(
            self._targets_from_items(
                self.pictures, collection="pictures", kind="picture"
            )
        )
        targets.extend(
            self._targets_from_items(self.tables, collection="tables", kind="table")
        )
        return targets

    def page_geometry(
        self, *, page_no: int, json_path: Path, image_root: Path | None
    ) -> PageGeometry:
        page = self._page(page_no)
        image_path = page.image_path(json_path=json_path, image_root=image_root)
        if not image_path.exists():
            message = f"page {page_no} image not found: {image_path}"
            raise FileNotFoundError(message)
        return PageGeometry(
            page_no=page_no,
            width=page.size.width,
            height=page.size.height,
            image_path=image_path,
        )

    def _targets_from_texts(self) -> list[CropTarget]:
        targets: list[CropTarget] = []
        for index, item in enumerate(self.texts):
            if item.label in self.text_target_labels:
                target = item.crop_target(
                    kind=item.label,  # type: ignore[arg-type]
                    collection="texts",
                    index=index,
                )
                if target is not None:
                    targets.append(target)
        return targets

    def _targets_from_items(
        self, items: list[DoclingItem], *, collection: str, kind: TargetKind
    ) -> list[CropTarget]:
        targets: list[CropTarget] = []
        for index, item in enumerate(items):
            target = item.crop_target(kind=kind, collection=collection, index=index)
            if target is not None:
                targets.append(target)
        return targets

    def _page(self, page_no: int) -> DoclingPage:
        page = None
        if isinstance(self.pages, dict):
            page = self.pages.get(str(page_no))
        else:
            for candidate_page in self.pages:
                if candidate_page.page_no == page_no:
                    page = candidate_page
        if page is None:
            message = f"page {page_no} is not present in Docling JSON pages"
            raise ValueError(message)
        return DoclingPage(size=page.size, image=page.image, page_no=page_no)


class PageGeometry(FrozenModel):
    """Resolved page dimensions and source image path."""

    page_no: int
    width: float
    height: float
    image_path: Path


class CropTarget(FrozenModel):
    """A Docling object selected for bbox cropping."""

    kind: TargetKind
    label: str
    self_ref: str
    collection: str
    index: int
    page_no: int
    bbox: DoclingBBox

    @property
    def filename_token(self) -> str:
        ref_token = self.self_ref.removeprefix("#/").replace("/", "_")
        safe_ref = SAFE_FILENAME_RE.sub("_", ref_token).strip("_")
        if safe_ref:
            return safe_ref
        return f"{self.collection}_{self.index}"

    def output_file(self, sequence: int, *, extension: str = "png") -> str:
        return f"p{self.page_no:03d}_{sequence:04d}_{self.kind}_{self.filename_token}.{extension}"


class CropRecord(FrozenModel):
    """Manifest entry for one written crop image."""

    sequence: int
    target: CropTarget
    page_box: PageBox
    pixel_box: PixelBox
    output_size: tuple[int, int]
    source_page_image_path: Path
    image_extension: str = "png"

    @property
    def output_file(self) -> str:
        return self.target.output_file(self.sequence, extension=self.image_extension)

    def manifest_record(self) -> dict[str, Any]:
        return {
            "output_file": self.output_file,
            "kind": self.target.kind,
            "label": self.target.label,
            "self_ref": self.target.self_ref,
            "collection": self.target.collection,
            "index": self.target.index,
            "page_no": self.target.page_no,
            "bbox_l": self.target.bbox.left,
            "bbox_t": self.target.bbox.top,
            "bbox_r": self.target.bbox.right,
            "bbox_b": self.target.bbox.bottom,
            "bbox_coord_origin": self.target.bbox.coord_origin,
            "crop_left_px": self.pixel_box.left,
            "crop_upper_px": self.pixel_box.upper,
            "crop_right_px": self.pixel_box.right,
            "crop_lower_px": self.pixel_box.lower,
            "output_width_px": self.output_size[0],
            "output_height_px": self.output_size[1],
            "source_page_image": str(self.source_page_image_path),
        }


class CropManifest(FrozenModel):
    """JSON manifest written beside crop images."""

    source_json: Path
    dpi: int
    image_format: ImageFormat = "PNG"
    crops: list[CropRecord]

    @property
    def count(self) -> int:
        return len(self.crops)

    def write(self, output_dir: Path) -> None:
        payload = {
            "source_json": str(self.source_json),
            "dpi": self.dpi,
            "image_format": self.image_format,
            "count": self.count,
            "crops": [crop.manifest_record() for crop in self.crops],
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
        )


class DoclingCropper(FrozenModel):
    """Crop selected Docling objects from the page images referenced by JSON."""

    input_path: Path
    output_dir: Path | None = None
    dpi: int = Field(default=150, gt=0)
    padding_points: float = Field(default=0, ge=0)
    image_root: Path | None = None
    image_format: ImageFormat = "PNG"

    @field_validator("image_format", mode="before")
    @classmethod
    def normalize_image_format(cls, value: Any) -> ImageFormat:
        normalized = str(value or "PNG").upper()
        if normalized not in {"PNG", "WEBP"}:
            message = f"Unsupported crop image format: {value!r}"
            raise ValueError(message)
        return normalized  # type: ignore[return-value]

    @property
    def image_extension(self) -> str:
        return self.image_format.lower()

    def run(self) -> list[CropRecord]:
        """Write crop images and manifest JSON, returning manifest records."""
        json_path = find_docling_json(self.input_path)
        output_dir = self.crop_output_dir()
        docling_document = self.read_document(json_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        records = self.write_crops(
            docling_document,
            docling_document.crop_targets(),
            json_path,
            output_dir,
        )
        CropManifest(
            source_json=json_path,
            dpi=self.dpi,
            image_format=self.image_format,
            crops=records,
        ).write(output_dir)
        return records

    def crop_output_dir(self) -> Path:
        if self.output_dir is not None:
            return self.output_dir
        if self.input_path.is_dir():
            return self.input_path / "bbox-crops"
        return self.input_path.parent / "bbox-crops"

    def read_document(self, json_path: Path) -> DoclingDocument:
        return DoclingDocument.model_validate_json(json_path.read_text())  # type: ignore[attr-defined]

    def write_crops(
        self,
        docling_document: DoclingDocument,
        targets: Iterable[CropTarget],
        json_path: Path,
        output_dir: Path,
    ) -> list[CropRecord]:
        records: list[CropRecord] = []
        for sequence, target in enumerate(targets, start=1):
            geometry = docling_document.page_geometry(
                page_no=target.page_no,
                json_path=json_path,
                image_root=self.image_root,
            )
            record = self.write_crop(target, geometry, sequence, output_dir)
            records.append(record)
        return records

    def write_crop(
        self,
        target: CropTarget,
        geometry: PageGeometry,
        sequence: int,
        output_dir: Path,
    ) -> CropRecord:
        page_box = target.bbox.to_page_box(
            page_width=geometry.width,
            page_height=geometry.height,
            padding_points=self.padding_points,
        )
        with Image.open(geometry.image_path) as page_image:
            source_image = page_image.convert("RGB")
            pixel_box = page_box.to_pixel_box(
                image_width=source_image.width,
                image_height=source_image.height,
                page_width=geometry.width,
                page_height=geometry.height,
            )
            output_size = page_box.output_pixel_size(dpi=self.dpi)
            crop_image = source_image.crop(pixel_box.pillow_box)
            if crop_image.size != output_size:
                crop_image = crop_image.resize(output_size, Image.Resampling.LANCZOS)
            record = CropRecord(
                sequence=sequence,
                target=target,
                page_box=page_box,
                pixel_box=pixel_box,
                output_size=output_size,
                source_page_image_path=geometry.image_path,
                image_extension=self.image_extension,
            )
            self.save_crop(crop_image, output_dir / record.output_file)
        return record

    def save_crop(self, image: Image.Image, output_path: Path) -> None:
        if self.image_format == "WEBP":
            image.save(output_path, format="WEBP", lossless=True, quality=100, method=3)
        else:
            image.save(output_path, format="PNG", dpi=(self.dpi, self.dpi))


def find_docling_json(path: Path) -> Path:
    if path.is_file():
        return path
    candidates = sorted(
        candidate
        for candidate in path.glob("*.json")
        if is_probable_docling_json(candidate)
    )
    if candidates:
        return candidates[0]
    recursive = sorted(
        candidate
        for candidate in path.rglob("*.json")
        if is_probable_docling_json(candidate)
    )
    if recursive:
        return recursive[0]
    message = f"No Docling JSON found under {path}"
    raise FileNotFoundError(message)


def is_probable_docling_json(path: Path) -> bool:
    name = path.name.lower()
    excluded_prefixes = ("object-", "gold-match-", "scored-")
    excluded = name.startswith(excluded_prefixes) or "timings" in name
    return not excluded


def resolve_image_uri(uri: str, *, json_path: Path, image_root: Path | None) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        return Path(parsed.path)
    uri_path = Path(uri)
    if uri_path.is_absolute():
        return uri_path
    return first_existing_path(uri_path, image_roots(json_path, image_root))


def image_roots(json_path: Path, image_root: Path | None) -> list[Path]:
    roots: list[Path] = []
    if image_root is not None:
        roots.append(image_root)
    roots.append(json_path.parent)
    roots.append(Path.cwd())
    return roots


def first_existing_path(relative_path: Path, roots: list[Path]) -> Path:
    for root in roots:
        candidate_path = root / relative_path
        if candidate_path.exists():
            return candidate_path
    return roots[0] / relative_path


def clamp(value: float, minimum: float, maximum: float) -> float:
    return min(max(value, minimum), maximum)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Docling JSON file or run directory")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for crop images; defaults to <input-dir>/bbox-crops",
    )
    parser.add_argument("--dpi", type=int, default=150, help="Output crop DPI")
    parser.add_argument(
        "--padding-points", type=float, default=0, help="Padding around each bbox"
    )
    parser.add_argument(
        "--image-root", type=Path, help="Base directory for relative page image URIs"
    )
    parser.add_argument(
        "--image-format",
        choices=["PNG", "WEBP", "png", "webp"],
        default="PNG",
        help="Output image format for crops. Default: PNG",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = DoclingCropper(
        input_path=args.input,
        output_dir=args.output_dir,
        dpi=args.dpi,
        padding_points=args.padding_points,
        image_root=args.image_root,
        image_format=args.image_format,
    ).run()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _log.info("Wrote %s crops", len(records))


if __name__ == "__main__":
    main()
