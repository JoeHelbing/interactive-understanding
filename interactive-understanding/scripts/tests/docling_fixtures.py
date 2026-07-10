from pathlib import Path

from docling_core.types.doc import (
    BoundingBox,
    CoordOrigin,
    DocItemLabel,
    DoclingDocument,
    GroupLabel,
    ImageRef,
    ProvenanceItem,
    Size,
)
from PIL import Image


def write_image(path: Path, size: tuple[int, int], color: str = "white") -> None:
    Image.new("RGB", size, color).save(path)


def provenance(
    left: int,
    upper: int,
    right: int,
    lower: int,
) -> ProvenanceItem:
    return ProvenanceItem(
        page_no=1,
        bbox=BoundingBox(
            l=left,
            t=upper,
            r=right,
            b=lower,
            coord_origin=CoordOrigin.TOPLEFT,
        ),
        charspan=(0, 0),
    )


def add_page(document: DoclingDocument, image_path: Path) -> None:
    document.add_page(
        page_no=1,
        size=Size(width=720, height=720),
        image=ImageRef(
            mimetype="image/webp",
            dpi=100,
            size=Size(width=1000, height=1000),
            uri=image_path.resolve(),
        ),
    )


def write_document(document: DoclingDocument, path: Path) -> None:
    path.write_text(
        document.model_dump_json(by_alias=True, indent=2) + "\n",
        encoding="utf-8",
    )


def write_minimal_docling_export(docling_dir: Path) -> None:
    docling_dir.mkdir()
    page_image = docling_dir / "page-1.webp"
    write_image(page_image, (1000, 1000))
    document = DoclingDocument(name="document")
    add_page(document, page_image)
    document.add_picture(
        prov=provenance(72, 72, 216, 216),
        parent=document.body,
    )
    write_document(document, docling_dir / "source.json")


def write_mixed_docling_export(docling_dir: Path) -> None:
    docling_dir.mkdir()
    page_image = docling_dir / "page-1.webp"
    write_image(page_image, (1000, 1000))
    document = DoclingDocument(name="document")
    add_page(document, page_image)
    group = document.add_group(label=GroupLabel.UNSPECIFIED, parent=document.body)
    document.add_text(
        label=DocItemLabel.TEXT,
        text="Mixed visual document.",
        parent=group,
    )
    document.add_table(
        data={"num_rows": 0, "num_cols": 0, "table_cells": []},
        prov=provenance(140, 140, 240, 240),
        parent=group,
    )
    document.add_formula(
        "",
        prov=provenance(20, 20, 120, 120),
        parent=group,
    )
    document.add_picture(
        prov=provenance(20, 140, 120, 240),
        parent=group,
    )
    document.add_code(
        "",
        prov=provenance(140, 20, 240, 120),
        parent=group,
    )
    write_document(document, docling_dir / "source.json")


def markdown_document() -> DoclingDocument:
    document = DoclingDocument(name="markdown")
    group = document.add_group(label=GroupLabel.UNSPECIFIED, parent=document.body)
    document.add_heading("Introduction", level=1, parent=group)
    document.add_text(
        label=DocItemLabel.TEXT,
        text="A paragraph.",
        parent=group,
    )
    list_group = document.add_list_group(parent=group)
    document.add_list_item("First item", parent=list_group)
    document.add_picture(parent=group)
    document.add_table(
        data={"num_rows": 0, "num_cols": 0, "table_cells": []},
        parent=group,
    )
    document.add_text(
        label=DocItemLabel.PAGE_FOOTER,
        text="1",
        prov=provenance(0, 0, 1, 1),
        parent=document.body,
    )
    return document
