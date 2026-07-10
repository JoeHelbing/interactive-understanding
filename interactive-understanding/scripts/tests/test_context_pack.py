import json
from pathlib import Path

from PIL import Image

from interactive_understanding.docling_context_pack import (
    ContextPackSettings,
    DoclingContextPack,
    MarkdownTextWriter,
    VisualLink,
    parse_args,
    settings_from_args,
)
from docling_fixtures import (
    markdown_document,
    write_image,
    write_minimal_docling_export,
    write_mixed_docling_export,
)
from interactive_understanding.image_sheets import (
    CropSheetBuilder,
    CropSheetSettings,
    CropSheetSource,
    PageSheetBuilder,
    PageSheetSettings,
    PageSheetSource,
)

PATCH_SIZE = 32
MAX_PATCHES = 2500
MAX_DIMENSION = 2048


def assert_codex_high_safe(width: int, height: int) -> None:
    patches = ((width + PATCH_SIZE - 1) // PATCH_SIZE) * (
        (height + PATCH_SIZE - 1) // PATCH_SIZE
    )
    assert width <= MAX_DIMENSION
    assert height <= MAX_DIMENSION
    assert patches <= MAX_PATCHES


def test_context_pack_builds_primary_text_visual_and_page_outputs(
    tmp_path: Path,
) -> None:
    source_pdf = tmp_path / "source.pdf"
    source_pdf.write_bytes(b"%PDF- fake test input")
    write_minimal_docling_export(tmp_path / "docling")

    result = DoclingContextPack(
        pdf_path=source_pdf,
        output_dir=tmp_path,
        run_docling_parse=False,
    ).run()

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert result.visual_count == 1
    assert result.text_path.exists()
    assert len(result.crop_sheets) == 1
    assert len(result.page_sheets) == 1
    assert len(result.page_images) == 1
    assert (tmp_path / manifest["visual_links"][0]["crop"]).exists()
    assert manifest["page_images"] == [
        {
            "page_no": 1,
            "image": "page-images-130dpi/page-001.webp",
            "source": "docling/page-1.webp",
            "dpi": 130,
            "width_px": 1300,
            "height_px": 1300,
        }
    ]
    with Image.open(tmp_path / manifest["page_images"][0]["image"]) as image:
        assert image.size == (1300, 1300)


def test_context_pack_preserves_mixed_visual_order_and_manifest_contract(
    tmp_path: Path,
) -> None:
    # Arrange
    source_pdf = tmp_path / "source.pdf"
    source_pdf.write_bytes(b"%PDF- fake test input")
    write_mixed_docling_export(tmp_path / "docling")

    # Act
    result = DoclingContextPack(
        pdf_path=source_pdf,
        output_dir=tmp_path,
        run_docling_parse=False,
    ).run()

    # Assert
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    crop_manifest = json.loads(
        (tmp_path / "visual-crops" / "manifest.json").read_text(encoding="utf-8")
    )
    expected_body_order = [
        "#/tables/0",
        "#/texts/1",
        "#/pictures/0",
        "#/texts/2",
    ]
    assert set(manifest) == {
        "source_pdf",
        "source_docling_json",
        "text",
        "visual_count",
        "crop_sheets",
        "page_sheets",
        "page_images",
        "visual_links",
    }
    assert manifest["visual_count"] == 4
    assert [link["self_ref"] for link in manifest["visual_links"]] == (
        expected_body_order
    )
    assert [crop["self_ref"] for crop in crop_manifest["crops"]] == [
        "#/texts/1",
        "#/texts/2",
        "#/pictures/0",
        "#/tables/0",
    ]
    assert [link.self_ref for link in result.visual_links] == expected_body_order
    assert len(result.crop_sheets) == 1
    assert len(result.page_sheets) == 1
    assert len(result.page_images) == 1
    for link in manifest["visual_links"]:
        assert (tmp_path / link["crop"]).exists()
        assert (tmp_path / link["crop_sheet"]).exists()
        assert (tmp_path / link["page_sheet"]).exists()
    text = result.text_path.read_text(encoding="utf-8")
    assert [text.index(ref) for ref in expected_body_order] == sorted(
        text.index(ref) for ref in expected_body_order
    )


def test_codex_sheet_budget_splits_sheets_without_reducing_item_fidelity(
    tmp_path: Path,
) -> None:
    crop_sources = []
    for index in range(9):
        image_path = tmp_path / f"crop-{index}.webp"
        write_image(image_path, (600, 600))
        crop_sources.append(
            CropSheetSource(
                self_ref=f"#/pictures/{index}",
                label="picture",
                kind="picture",
                page_no=index + 1,
                source_path=image_path,
            )
        )

    crop_sheets = CropSheetBuilder(
        output_dir=tmp_path / "crop-sheets",
        settings=CropSheetSettings(),
        sources=crop_sources,
    ).write()

    assert len(crop_sheets) > 1
    for sheet in crop_sheets:
        assert_codex_high_safe(sheet.width_px, sheet.height_px)
        assert {item.width_px for item in sheet.items} == {600}
        assert {item.height_px for item in sheet.items} == {600}


def test_page_sheet_budget_splits_pages_without_reducing_thumbnail_height(
    tmp_path: Path,
) -> None:
    page_sources = []
    for page_no in range(1, 21):
        image_path = tmp_path / f"page-{page_no}.webp"
        write_image(image_path, (640, 360))
        page_sources.append(PageSheetSource(page_no=page_no, source_path=image_path))

    page_sheets = PageSheetBuilder(
        output_dir=tmp_path / "page-sheets",
        settings=PageSheetSettings(),
        sources=page_sources,
    ).write()

    assert len(page_sheets) > 1
    assert sum(len(sheet.items) for sheet in page_sheets) == 20
    for sheet in page_sheets:
        assert_codex_high_safe(sheet.width_px, sheet.height_px)
        assert {item.height_px for item in sheet.items} == {360}


def test_settings_from_args_keeps_cli_field_mapping_in_one_place() -> None:
    args = parse_args(
        [
            "paper.pdf",
            "--output-dir",
            "pack",
            "--docling-device",
            "cpu",
            "--crop-dpi",
            "144",
            "--page-image-dpi",
            "96",
            "--sheet-max-patches",
            "1024",
        ]
    )

    settings = settings_from_args(args)

    assert isinstance(settings, ContextPackSettings)
    assert settings.docling_device == "cpu"
    assert settings.crop_dpi == 144
    assert settings.page_image_dpi == 96
    assert settings.sheet_max_patches == 1024


def test_markdown_writer_preserves_nested_reading_order_and_visual_fallbacks(
    tmp_path: Path,
) -> None:
    # Arrange
    source = markdown_document()
    picture_link = VisualLink(
        self_ref="#/pictures/0",
        kind="picture",
        label="picture",
        page_no=1,
        crop_path=tmp_path / "visual-crops" / "picture.webp",
        crop_sheet_path=None,
        crop_sheet_bbox_px=None,
        page_sheet_path=None,
        page_sheet_bbox_px=None,
    )
    output_path = tmp_path / "text.md"

    # Act
    MarkdownTextWriter(
        source=source,
        links_by_ref={picture_link.self_ref: picture_link},
        output_dir=tmp_path,
    ).write(output_path)

    # Assert
    assert output_path.read_text(encoding="utf-8") == (
        "# Extracted text with visual links\n"
        "\n"
        "Code blocks, formulas, figures, and tables are linked as images rather than transcribed as text.\n"
        "\n"
        "\n"
        "## Introduction\n"
        "\n"
        "\n"
        "A paragraph.\n"
        "\n"
        "- First item\n"
        "\n"
        "[visual: picture #/pictures/0](visual-crops/picture.webp) -- page 1; crop: `visual-crops/picture.webp`\n"
        "\n"
        "\n"
        "[visual missing: table #/tables/0]\n"
        "\n"
        "\n"
        "< Page 1 >\n"
        "\n"
        "---\n"
    )
