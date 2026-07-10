from pathlib import Path

from docling.datamodel.accelerator_options import AcceleratorDevice
from docling.datamodel.base_models import InputFormat
from docling_core.types.doc import DoclingDocument, ImageRef, Size
from PIL import Image

from interactive_understanding.docling_context_pack import DoclingContextPack


def context_pack(tmp_path: Path, **options: str) -> DoclingContextPack:
    return DoclingContextPack(
        pdf_path=tmp_path / "source.pdf",
        output_dir=tmp_path,
        **options,
    )


def test_converter_maps_supported_cli_controls_to_docling_options(
    tmp_path: Path,
) -> None:
    converter = context_pack(
        tmp_path,
        docling_device="cpu",
        docling_ocr_engine="auto",
    ).document_converter()

    options = converter.format_to_options[InputFormat.PDF].pipeline_options
    assert options is not None
    assert options.accelerator_options.device == AcceleratorDevice.CPU
    assert options.do_ocr is True
    assert options.ocr_options.kind == "auto"
    assert options.images_scale == 2
    assert options.generate_page_images is True
    assert options.generate_picture_images is True
    assert options.enable_remote_services is False


def test_export_uses_docling_serializer_with_referenced_webp_images(
    tmp_path: Path,
) -> None:
    document = DoclingDocument(name="source")
    document.add_page(
        page_no=1,
        size=Size(width=72, height=144),
        image=ImageRef.from_pil(Image.new("RGB", (144, 288), "white"), dpi=144),
    )
    document.add_picture(
        image=ImageRef.from_pil(Image.new("RGB", (20, 10), "blue"), dpi=144),
        parent=document.body,
    )
    output_dir = tmp_path / "docling"
    json_path = output_dir / "source.json"

    context_pack(tmp_path).export_docling_document(
        document,
        output_dir=output_dir,
        json_path=json_path,
    )

    reloaded = DoclingDocument.load_from_json(json_path)
    image_path = reloaded.pages[1].image.uri
    assert isinstance(image_path, Path)
    assert image_path.suffix == ".webp"
    assert image_path.exists()
    picture_image = reloaded.pictures[0].image
    assert picture_image is not None
    assert isinstance(picture_image.uri, Path)
    assert picture_image.uri.suffix == ".webp"
    assert picture_image.uri.exists()
    assert list(output_dir.rglob("*.png")) == []
    assert len(list(output_dir.rglob("*.webp"))) == 2
    assert reloaded.pages[1].image.mimetype == "image/webp"
    assert picture_image.mimetype == "image/webp"
    assert reloaded.pages[1].image.pil_image.size == (144, 288)
