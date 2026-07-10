from docling_core.types.doc import (
    BoundingBox,
    DocItemLabel,
    DoclingDocument,
    GroupLabel,
    ProvenanceItem,
)

from interactive_understanding.docling_helpers import (
    iter_body_items,
    reference_location,
    visual_items_in_reading_order,
    visual_kind,
)


def provenance(page_no: int = 1) -> ProvenanceItem:
    return ProvenanceItem(
        page_no=page_no,
        bbox=BoundingBox(l=10, t=40, r=30, b=20),
        charspan=(0, 0),
    )


def test_helpers_use_docling_items_for_reading_order_and_visual_kinds() -> None:
    document = DoclingDocument(name="fixture")
    group = document.add_group(label=GroupLabel.UNSPECIFIED, parent=document.body)
    heading = document.add_heading("Introduction", parent=group)
    picture = document.add_picture(prov=provenance(), parent=group)
    table = document.add_table(
        data={"num_rows": 0, "num_cols": 0, "table_cells": []},
        prov=provenance(),
        parent=group,
    )
    formula = document.add_formula("x", prov=provenance(), parent=group)
    code = document.add_code("print(1)", prov=provenance(), parent=group)
    paragraph = document.add_text(
        label=DocItemLabel.TEXT,
        text="Done.",
        parent=document.body,
    )

    assert list(iter_body_items(document)) == [
        heading,
        picture,
        table,
        formula,
        code,
        paragraph,
    ]
    assert list(visual_items_in_reading_order(document)) == [
        picture,
        table,
        formula,
        code,
    ]
    assert [visual_kind(item) for item in (picture, table, formula, code)] == [
        "picture",
        "table",
        "formula",
        "code",
    ]
    assert reference_location(picture.self_ref) == ("pictures", 0)
