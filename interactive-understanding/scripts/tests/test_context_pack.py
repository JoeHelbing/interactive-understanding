import ast
import json
from pathlib import Path

from PIL import Image

from interactive_understanding.docling_context_pack import (
    ContextPackSettings,
    CropSheetBuilder,
    CropSheetSettings,
    CropSheetSource,
    DoclingContextPack,
    PageSheetBuilder,
    PageSheetSettings,
    PageSheetSource,
    parse_args,
    settings_from_args,
)

PATCH_SIZE = 32
MAX_PATCHES = 2500
MAX_DIMENSION = 2048
SCRIPT_ROOT = Path(__file__).parents[1]
CROP_MODULE = (
    SCRIPT_ROOT / "src" / "interactive_understanding" / "crop_docling_bboxes.py"
)
CONTEXT_MODULE = (
    SCRIPT_ROOT / "src" / "interactive_understanding" / "docling_context_pack.py"
)


def write_image(path: Path, size: tuple[int, int], color: str = "white") -> None:
    Image.new("RGB", size, color).save(path)


def assert_codex_high_safe(width: int, height: int) -> None:
    patches = ((width + PATCH_SIZE - 1) // PATCH_SIZE) * (
        (height + PATCH_SIZE - 1) // PATCH_SIZE
    )
    assert width <= MAX_DIMENSION
    assert height <= MAX_DIMENSION
    assert patches <= MAX_PATCHES


def write_minimal_docling_export(docling_dir: Path) -> None:
    docling_dir.mkdir()
    write_image(docling_dir / "page-1.png", (1000, 1000))
    payload = {
        "body": {"children": [{"$ref": "#/pictures/0"}]},
        "texts": [],
        "pictures": [
            {
                "self_ref": "#/pictures/0",
                "label": "picture",
                "prov": [
                    {
                        "page_no": 1,
                        "bbox": {
                            "l": 72,
                            "t": 72,
                            "r": 216,
                            "b": 216,
                            "coord_origin": "TOPLEFT",
                        },
                    }
                ],
            }
        ],
        "tables": [],
        "groups": [],
        "pages": {
            "1": {
                "page_no": 1,
                "size": {"width": 720, "height": 720},
                "image": {"uri": "page-1.png"},
            }
        },
    }
    (docling_dir / "document.json").write_text(json.dumps(payload), encoding="utf-8")


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


def test_context_pack_orchestrator_does_not_use_computed_field_conveniences() -> None:
    tree = ast.parse(CONTEXT_MODULE.read_text(encoding="utf-8"))
    context_pack_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "DoclingContextPack"
    )
    computed_field_decorators = [
        decorator
        for node in context_pack_class.body
        if isinstance(node, ast.FunctionDef)
        for decorator in node.decorator_list
        if isinstance(decorator, ast.Name) and decorator.id == "computed_field"
    ]

    assert computed_field_decorators == []


def test_cropper_uses_plain_properties_instead_of_computed_fields() -> None:
    tree = ast.parse(CROP_MODULE.read_text(encoding="utf-8"))
    computed_field_decorators = [
        decorator
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        for decorator in node.decorator_list
        if isinstance(decorator, ast.Name) and decorator.id == "computed_field"
    ]

    assert computed_field_decorators == []
