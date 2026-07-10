"""Small context-pack policies over Docling's public document model."""

from collections.abc import Iterable, Iterator
from typing import Literal, TypeAlias

from docling_core.types.doc import (
    CodeItem,
    DocItem,
    DoclingDocument,
    FormulaItem,
    GroupItem,
    PictureItem,
    RefItem,
    TableItem,
)

VisualKind: TypeAlias = Literal["code", "formula", "picture", "table"]
VisualItem: TypeAlias = CodeItem | FormulaItem | PictureItem | TableItem
VISUAL_ITEM_TYPES = (CodeItem, FormulaItem, PictureItem, TableItem)


def reference_location(self_ref: str) -> tuple[str, int]:
    """Return the collection and index encoded in a Docling self-reference."""
    parts = self_ref.removeprefix("#/").split("/")
    if len(parts) != 2 or not parts[1].isdigit():
        raise ValueError(f"unsupported Docling reference: {self_ref}")
    return parts[0], int(parts[1])


def iter_body_items(document: DoclingDocument) -> Iterator[DocItem]:
    """Yield body items in the legacy context-pack order, flattening groups."""
    yield from _resolve_children(document.body.children, document)


def _resolve_children(
    references: Iterable[RefItem], document: DoclingDocument
) -> Iterator[DocItem]:
    for reference in references:
        item = reference.resolve(document)
        if isinstance(item, GroupItem):
            yield from _resolve_children(item.children, document)
        elif isinstance(item, DocItem):
            yield item


def visual_kind(item: DocItem) -> VisualKind | None:
    """Classify the Docling visual item types used by context packs."""
    if isinstance(item, CodeItem):
        return "code"
    if isinstance(item, FormulaItem):
        return "formula"
    if isinstance(item, PictureItem):
        return "picture"
    if isinstance(item, TableItem):
        return "table"
    return None


def visual_items_in_reading_order(
    document: DoclingDocument,
) -> Iterator[VisualItem]:
    for item, _level in document.iterate_items():
        if isinstance(item, VISUAL_ITEM_TYPES):
            yield item
