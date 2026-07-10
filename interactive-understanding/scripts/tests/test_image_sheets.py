from pathlib import Path

from interactive_understanding.docling_context_pack import (
    DoclingContextPack,
    context_pack_paths,
)
from interactive_understanding.image_sheets import (
    CropSheetBuilder,
    CropSheetSettings,
    PageSheetBuilder,
    PageSheetSettings,
)


def test_sheet_builders_define_empty_sources_as_empty_output(tmp_path: Path) -> None:
    # Arrange
    crop_builder = CropSheetBuilder(
        output_dir=tmp_path / "crop-sheets",
        settings=CropSheetSettings(),
        sources=[],
    )
    page_builder = PageSheetBuilder(
        output_dir=tmp_path / "page-sheets",
        settings=PageSheetSettings(),
        sources=[],
    )

    # Act
    crop_sheets = crop_builder.write()
    page_sheets = page_builder.write()

    # Assert
    assert crop_sheets == []
    assert page_sheets == []


def test_context_pack_does_not_create_crop_sheet_directory_without_crops(
    tmp_path: Path,
) -> None:
    # Arrange
    context_pack = DoclingContextPack(
        pdf_path=tmp_path / "paper.pdf",
        output_dir=tmp_path,
        run_docling_parse=False,
    )
    paths = context_pack_paths(tmp_path, page_image_dpi=130)

    # Act
    sheets = context_pack.write_crop_sheets(
        [],
        paths=paths,
        settings=CropSheetSettings(),
    )

    # Assert
    assert sheets == []
    assert not paths.crop_sheet_dir.exists()
