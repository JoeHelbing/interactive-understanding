import argparse
from pathlib import Path

import pytest
from pydantic import ValidationError

from interactive_understanding.docling_context_pack import (
    ContextPackSettings,
    DoclingContextPack,
    build_parser,
    main,
    parse_args,
    settings_from_args,
)


def expected_defaults() -> dict[str, str | int | float]:
    return {
        "docling_ocr_engine": "auto",
        "docling_device": "auto",
        "crop_dpi": 150,
        "crop_padding_points": 1.0,
        "crop_sheet_width": 2200,
        "crop_sheet_height": 2600,
        "crop_sheet_gutter": 4,
        "crop_thumb_max_edge": 1800,
        "page_sheet_columns": 4,
        "page_sheet_rows": 5,
        "page_sheet_gutter": 10,
        "page_thumb_height": 360,
        "page_image_dpi": 130,
        "sheet_max_dimension": 2048,
        "sheet_patch_size": 32,
        "sheet_max_patches": 2500,
    }


def test_cli_defaults_match_context_pack_settings() -> None:
    # Arrange
    args = parse_args(["paper.pdf", "--output-dir", "pack"])

    # Act
    settings = settings_from_args(args)

    # Assert
    assert settings.model_dump() == expected_defaults()


def test_cli_maps_every_context_pack_setting() -> None:
    # Arrange
    args = parse_args(
        [
            "paper.pdf",
            "--output-dir",
            "pack",
            "--docling-ocr-engine",
            "rapidocr",
            "--docling-device",
            "cpu",
            "--crop-dpi",
            "144",
            "--crop-padding-points",
            "2.5",
            "--crop-sheet-width",
            "1800",
            "--crop-sheet-height",
            "1900",
            "--crop-sheet-gutter",
            "8",
            "--crop-thumb-max-edge",
            "1200",
            "--page-sheet-columns",
            "3",
            "--page-sheet-rows",
            "4",
            "--page-sheet-gutter",
            "12",
            "--page-thumb-height",
            "300",
            "--page-image-dpi",
            "96",
            "--sheet-max-dimension",
            "1600",
            "--sheet-patch-size",
            "16",
            "--sheet-max-patches",
            "1024",
        ]
    )

    # Act
    settings = settings_from_args(args)

    # Assert
    assert settings.model_dump() == {
        "docling_ocr_engine": "rapidocr",
        "docling_device": "cpu",
        "crop_dpi": 144,
        "crop_padding_points": 2.5,
        "crop_sheet_width": 1800,
        "crop_sheet_height": 1900,
        "crop_sheet_gutter": 8,
        "crop_thumb_max_edge": 1200,
        "page_sheet_columns": 3,
        "page_sheet_rows": 4,
        "page_sheet_gutter": 12,
        "page_thumb_height": 300,
        "page_image_dpi": 96,
        "sheet_max_dimension": 1600,
        "sheet_patch_size": 16,
        "sheet_max_patches": 1024,
    }


def test_settings_from_partial_namespace_uses_model_defaults() -> None:
    # Arrange
    args = argparse.Namespace(crop_dpi=144)

    # Act
    settings = settings_from_args(args)

    # Assert
    assert settings.crop_dpi == 144
    assert settings.docling_device == "auto"
    assert settings.page_image_dpi == 130


def test_cli_rejects_removed_docling_command() -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "paper.pdf",
                "--output-dir",
                "pack",
                "--docling-command",
                "custom-docling",
            ]
        )


def test_help_displays_an_accepted_docling_device_default() -> None:
    help_text = build_parser().format_help()

    assert "(default: auto)" in help_text
    assert "AcceleratorDevice.AUTO" not in help_text
    assert parse_args(["paper.pdf", "--output-dir", "pack"]).docling_device == "auto"


def test_main_validates_settings_before_touching_source(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "pack"

    status = main(
        [
            str(tmp_path / "missing.pdf"),
            "--output-dir",
            str(output_dir),
            "--crop-dpi",
            "0",
            "--json",
        ]
    )

    assert status == 1
    assert "greater than 0" in capsys.readouterr().out
    assert not output_dir.exists()


def test_context_pack_accepts_every_settings_field() -> None:
    # Arrange
    settings_fields = set(ContextPackSettings.model_fields)

    # Act
    context_pack_fields = set(DoclingContextPack.model_fields)

    # Assert
    assert settings_fields <= context_pack_fields
    for field_name in settings_fields:
        settings_default = ContextPackSettings.model_fields[field_name].default
        context_pack_default = DoclingContextPack.model_fields[field_name].default
        assert context_pack_default == settings_default


@pytest.mark.parametrize("field_name", ["crop_dpi", "sheet_max_patches"])
def test_shared_numeric_constraints_reject_zero(field_name: str) -> None:
    # Arrange
    invalid_setting = {field_name: 0}

    # Act / Assert
    with pytest.raises(ValidationError, match="greater than 0"):
        ContextPackSettings(**invalid_setting)
    with pytest.raises(ValidationError, match="greater than 0"):
        DoclingContextPack(
            pdf_path=Path("paper.pdf"),
            output_dir=Path("pack"),
            **invalid_setting,
        )
