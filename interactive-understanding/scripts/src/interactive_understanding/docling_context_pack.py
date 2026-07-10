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
import gc
from enum import Enum
import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

from docling.datamodel.accelerator_options import (
    AcceleratorDevice,
    AcceleratorOptions,
)
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.models.factories import get_ocr_factory
from docling_core.types.doc import (
    DocItem,
    DocItemLabel,
    DoclingDocument,
    ImageRef,
    TextItem,
)
from pydantic import (
    Field,
    computed_field,  # type: ignore[attr-defined]
    field_validator,  # type: ignore[attr-defined]
)

from interactive_understanding.crop_docling_bboxes import (
    CropRecord,
    DoclingCropper,
    image_path,
)
from interactive_understanding.docling_helpers import (
    VisualKind,
    iter_body_items,
    visual_items_in_reading_order,
    visual_kind,
)
from interactive_understanding.image_sheets import (
    CropSheetBuilder,
    CropSheetSettings,
    CropSheetSource,
    ImageSheet,
    ImageSheetBudget,
    PageImageExporter,
    PageImageRecord,
    PageImageSource,
    PageSheetBuilder,
    PageSheetSettings,
    PageSheetSource,
    maybe_bbox,
    maybe_relative_path,
    relative_path,
    save_webp,
)
from interactive_understanding.models import ContextPackModel as FrozenModel

_log = logging.getLogger(__name__)


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


class MarkdownTextWriter(FrozenModel):
    """Write text-only Markdown with visual object links in reading order."""

    source: DoclingDocument
    links_by_ref: dict[str, VisualLink]
    output_dir: Path

    def write(self, path: Path) -> None:
        lines = [
            "# Extracted text with visual links",
            "",
            "Code blocks, formulas, figures, and tables are linked as images rather than transcribed as text.",
            "",
        ]
        for item in iter_body_items(self.source):
            lines.extend(self.lines_for_item(item))
        path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    def lines_for_item(self, item: DocItem) -> list[str]:
        kind = visual_kind(item)
        if kind is not None:
            return self.visual_lines(item.self_ref, kind)
        if not isinstance(item, TextItem):
            return []
        text = (item.text or item.orig).strip()
        if not text:
            return []
        if item.label == DocItemLabel.PAGE_FOOTER:
            return ["", f"< Page {text or self.page_number(item)} >", "", "---", ""]
        if item.label == DocItemLabel.SECTION_HEADER:
            raw_level = getattr(item, "level", 1)
            level = min(6, max(2, int(raw_level) + 1))
            return ["", f"{'#' * level} {text}", ""]
        if item.label == DocItemLabel.LIST_ITEM:
            return [f"- {text}"]
        return ["", text, ""]

    def visual_lines(self, self_ref: str, kind: VisualKind) -> list[str]:
        link = self.links_by_ref.get(self_ref)
        if link is None:
            return ["", f"[visual missing: {kind} {self_ref}]", ""]
        return ["", link.markdown_line(output_dir=self.output_dir), ""]

    def page_number(self, item: DocItem) -> str:
        if not item.prov:
            return "unknown"
        return str(item.prov[0].page_no)


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


class ContextPackSettings(FrozenModel):
    """Validated Docling and sheet options shared by the workflow and CLI."""

    docling_ocr_engine: str = Field(
        default="auto", description="Docling OCR engine for bitmap fallback"
    )
    docling_device: AcceleratorDevice = Field(
        default=AcceleratorDevice.AUTO,
        description="Docling accelerator device: auto, cpu, cuda, mps, or xpu",
    )
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
    page_image_dpi: int = Field(
        default=130,
        gt=0,
        description="DPI for exported full-page images in page-images-<dpi>dpi/",
    )
    sheet_max_dimension: int = Field(
        default=2048,
        gt=0,
        description="maximum output sheet width or height before model-side resizing",
    )
    sheet_patch_size: int = Field(
        default=32,
        gt=0,
        description="image-intake patch size used for sheet budget checks",
    )
    sheet_max_patches: int = Field(
        default=2500,
        gt=0,
        description="maximum image-intake patches allowed per output sheet",
    )

    def sheet_budget(self) -> ImageSheetBudget:
        return ImageSheetBudget(
            max_dimension=self.sheet_max_dimension,
            patch_size=self.sheet_patch_size,
            max_patches=self.sheet_max_patches,
        )

    def crop_sheet_settings(self, budget: ImageSheetBudget) -> CropSheetSettings:
        return CropSheetSettings(
            width=self.crop_sheet_width,
            height=self.crop_sheet_height,
            gutter=self.crop_sheet_gutter,
            thumb_max_edge=self.crop_thumb_max_edge,
            sheet_budget=budget,
        )

    def page_sheet_settings(self, budget: ImageSheetBudget) -> PageSheetSettings:
        return PageSheetSettings(
            columns=self.page_sheet_columns,
            rows=self.page_sheet_rows,
            gutter=self.page_sheet_gutter,
            thumb_height=self.page_thumb_height,
            sheet_budget=budget,
        )

    def context_pack(self, *, pdf_path: Path, output_dir: Path) -> DoclingContextPack:
        return DoclingContextPack(
            pdf_path=pdf_path,
            output_dir=output_dir,
            **self.model_dump(),
        )


class DoclingContextPack(ContextPackSettings):
    """Convert a PDF with Docling, then build its multimodal context pack."""

    pdf_path: Path
    output_dir: Path
    run_docling_parse: bool = True

    @field_validator("pdf_path")
    @classmethod
    def require_pdf_input(cls, value: Path) -> Path:
        if value.suffix.lower() != ".pdf":
            raise ValueError(f"context pack input must be a PDF: {value}")
        return value

    def run(self) -> ContextPackResult:
        paths = context_pack_paths(self.output_dir, page_image_dpi=self.page_image_dpi)
        json_path = paths.docling_output_dir / "source.json"
        budget = self.sheet_budget()

        self.prepare_output_dirs(paths)
        if self.run_docling_parse:
            self.convert_pdf(paths.docling_output_dir, json_path=json_path)
        document = DoclingDocument.load_from_json(json_path)

        crop_records = self.write_visual_crops(
            document,
            paths=paths,
            json_path=json_path,
        )
        crop_sheets = self.write_crop_sheets(
            crop_records,
            paths=paths,
            settings=self.crop_sheet_settings(budget),
        )
        page_sources = self.page_sources(document)
        page_sheets = self.write_page_sheets(
            page_sources,
            paths=paths,
            settings=self.page_sheet_settings(budget),
        )
        page_images = self.write_page_images(page_sources, paths=paths)
        visual_links = self.visual_links(
            crop_records,
            crop_sheets,
            page_sheets,
            paths=paths,
        )
        MarkdownTextWriter(
            source=document,
            links_by_ref={link.self_ref: link for link in visual_links},
            output_dir=paths.output_dir,
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

    def convert_pdf(self, output_dir: Path, *, json_path: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        converter = self.document_converter()
        conversion = converter.convert(self.pdf_path, raises_on_error=False)
        if conversion.status != ConversionStatus.SUCCESS:
            raise RuntimeError(f"Docling conversion failed: {conversion.status}")
        self.export_docling_document(
            conversion.document,
            output_dir=output_dir,
            json_path=json_path,
        )
        del conversion, converter
        gc.collect()

    def document_converter(self) -> DocumentConverter:
        ocr_options = get_ocr_factory(allow_external_plugins=False).create_options(
            kind=self.docling_ocr_engine,
            force_full_page_ocr=False,
        )
        pipeline_options = PdfPipelineOptions(
            allow_external_plugins=False,
            enable_remote_services=False,
            accelerator_options=AcceleratorOptions(device=self.docling_device),
            do_ocr=True,
            ocr_options=ocr_options,
            do_table_structure=True,
            images_scale=2,
            generate_page_images=True,
            generate_picture_images=True,
        )
        return DocumentConverter(
            allowed_formats=[InputFormat.PDF],
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            },
        )

    def export_docling_document(
        self,
        document: DoclingDocument,
        *,
        output_dir: Path,
        json_path: Path,
    ) -> None:
        artifact_dir = output_dir / "source_artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        for page_no, page in sorted(document.pages.items()):
            if page.image is None:
                raise ValueError(f"Docling page {page_no} has no rendered image")
            page.image = self.export_image_ref(
                page.image,
                artifact_dir / f"page_{page_no:06d}.webp",
            )
        for index, picture in enumerate(document.pictures, start=1):
            if picture.image is not None:
                picture.image = self.export_image_ref(
                    picture.image,
                    artifact_dir / f"picture_{index:06d}.webp",
                )
        json_path.write_text(
            json.dumps(document.export_to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def export_image_ref(self, image_ref: ImageRef, output_path: Path) -> ImageRef:
        image = image_ref.pil_image
        if image is None:
            raise ValueError(f"Docling image cannot be read: {image_ref.uri}")
        output_path = output_path.resolve()
        save_webp(image.convert("RGB"), output_path)
        return ImageRef(
            mimetype="image/webp",
            dpi=image_ref.dpi,
            size=image_ref.size,
            uri=output_path,
        )

    def write_visual_crops(
        self,
        document: DoclingDocument,
        *,
        paths: ContextPackPaths,
        json_path: Path,
    ) -> list[CropRecord]:
        records = DoclingCropper(
            output_dir=paths.crop_dir,
            dpi=self.crop_dpi,
            padding_points=self.crop_padding_points,
            image_format="WEBP",
        ).write(document, source_json=json_path)
        records_by_ref = {record.target.self_ref: record for record in records}
        ordered_refs = [
            item.self_ref for item in visual_items_in_reading_order(document)
        ]
        ordered = [records_by_ref[ref] for ref in ordered_refs if ref in records_by_ref]
        used_refs = {record.target.self_ref for record in ordered}
        ordered.extend(
            record for record in records if record.target.self_ref not in used_refs
        )
        return ordered

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
            CropSheetSource(
                self_ref=record.target.self_ref,
                label=record.target.label,
                kind=record.target.kind,
                page_no=record.target.page_no,
                source_path=paths.crop_dir / record.output_file,
            )
            for record in records
        ]
        return CropSheetBuilder(
            output_dir=paths.crop_sheet_dir,
            settings=settings,
            sources=sources,
        ).write()

    def page_sources(self, document: DoclingDocument) -> list[PageImageSource]:
        sources: list[PageImageSource] = []
        for page_no, page in sorted(document.pages.items()):
            if page.image is None:
                raise ValueError(f"Docling page {page_no} has no rendered image")
            source_path = image_path(page.image.uri)
            if not source_path.exists():
                raise FileNotFoundError(
                    f"page {page_no} image not found: {source_path}"
                )
            sources.append(
                PageImageSource(
                    page_no=page_no,
                    source_path=source_path,
                    width_points=page.size.width,
                    height_points=page.size.height,
                )
            )
        return sources

    def write_page_sheets(
        self,
        page_sources: list[PageImageSource],
        *,
        paths: ContextPackPaths,
        settings: PageSheetSettings,
    ) -> list[ImageSheet]:
        return PageSheetBuilder(
            output_dir=paths.page_sheet_dir,
            settings=settings,
            sources=[
                PageSheetSource(
                    page_no=source.page_no,
                    source_path=source.source_path,
                )
                for source in page_sources
            ],
        ).write()

    def write_page_images(
        self,
        page_sources: list[PageImageSource],
        *,
        paths: ContextPackPaths,
    ) -> list[PageImageRecord]:
        return PageImageExporter(
            output_dir=paths.page_image_dir,
            dpi=self.page_image_dpi,
            sources=page_sources,
        ).write()

    def visual_links(
        self,
        records: list[CropRecord],
        crop_sheets: list[ImageSheet],
        page_sheets: list[ImageSheet],
        *,
        paths: ContextPackPaths,
    ) -> list[VisualLink]:
        crop_details = sheet_details_by_ref(crop_sheets)
        page_details = sheet_details_by_page(page_sheets)
        return [
            VisualLink(
                self_ref=record.target.self_ref,
                kind=record.target.kind,
                label=record.target.label,
                page_no=record.target.page_no,
                crop_path=paths.crop_dir / record.output_file,
                crop_sheet_path=crop_details.get(record.target.self_ref, (None, None))[
                    0
                ],
                crop_sheet_bbox_px=crop_details.get(
                    record.target.self_ref, (None, None)
                )[1],
                page_sheet_path=page_details.get(record.target.page_no, (None, None))[
                    0
                ],
                page_sheet_bbox_px=page_details.get(
                    record.target.page_no, (None, None)
                )[1],
            )
            for record in records
        ]

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


def sheet_details_by_ref(
    sheets: list[ImageSheet],
) -> dict[str, tuple[Path, tuple[int, int, int, int]]]:
    return {
        item.self_ref: (sheet.sheet_path, item.bbox_px)
        for sheet in sheets
        for item in sheet.items
    }


def sheet_details_by_page(
    sheets: list[ImageSheet],
) -> dict[int, tuple[Path, tuple[int, int, int, int]]]:
    return {
        item.page_no: (sheet.sheet_path, item.bbox_px)
        for sheet in sheets
        for item in sheet.items
        if item.page_no is not None
    }


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
    for name, field in ContextPackSettings.model_fields.items():
        default = field.default
        choices = None
        argument_type = field.annotation
        if isinstance(default, Enum):
            choices = [member.value for member in type(default)]
            default = default.value
            argument_type = str
        parser.add_argument(
            f"--{name.replace('_', '-')}",
            type=argument_type,
            choices=choices,
            default=default,
            help=field.description,
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
        acquired = PdfSource(source=args.source).write_pdf(args.output_dir)
        result = settings.context_pack(
            pdf_path=acquired.pdf_path,
            output_dir=args.output_dir,
        ).run()
        write_source_metadata(acquired, args.output_dir)
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
