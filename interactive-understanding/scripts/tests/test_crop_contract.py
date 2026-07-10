import pytest
from docling_core.types.doc import BoundingBox, CoordOrigin, Size

from interactive_understanding.crop_docling_bboxes import (
    output_pixel_size,
    padded_page_box,
    pixel_box,
)


@pytest.mark.parametrize(
    ("bbox", "expected_page_box"),
    [
        (
            BoundingBox(
                l=30,
                t=40,
                r=10,
                b=20,
                coord_origin=CoordOrigin.TOPLEFT,
            ),
            (5.0, 15.0, 35.0, 45.0),
        ),
        (
            BoundingBox(
                l=10,
                t=20,
                r=30,
                b=40,
                coord_origin=CoordOrigin.BOTTOMLEFT,
            ),
            (5.0, 55.0, 35.0, 85.0),
        ),
        (
            BoundingBox(
                l=-10,
                t=-10,
                r=110,
                b=110,
                coord_origin=CoordOrigin.TOPLEFT,
            ),
            (0.0, 0.0, 100.0, 100.0),
        ),
    ],
)
def test_bbox_normalizes_origin_order_and_padding(
    bbox: BoundingBox,
    expected_page_box: tuple[float, float, float, float],
) -> None:
    page_box = padded_page_box(
        bbox,
        page_size=Size(width=100, height=100),
        padding_points=5,
    )

    assert page_box.as_tuple() == expected_page_box


def test_page_box_scales_outward_to_pixels_and_requested_dpi() -> None:
    page_size = Size(width=100, height=100)
    page_box = padded_page_box(
        BoundingBox(
            l=10.2,
            t=20.2,
            r=30.7,
            b=40.7,
            coord_origin=CoordOrigin.TOPLEFT,
        ),
        page_size=page_size,
    )

    assert pixel_box(page_box, page_size=page_size, image_size=(200, 300)) == (
        20,
        60,
        62,
        123,
    )
    assert output_pixel_size(page_box, dpi=144) == (41, 41)
