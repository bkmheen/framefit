"""Smoke tests for the permissive core (no DL/HEIC extras required)."""
import numpy as np

from framefit.geometry import (
    aspect_ratio,
    aspect_score,
    inset_quad,
    order_corners,
    warp_from_quad,
)


def _synthetic_slide(w=1200, h=800):
    """Dark frame with a bright, slightly keystoned 16:10 'slide' inside it."""
    img = np.full((h, w, 3), 12, np.uint8)  # near-black room
    quad = np.array([[180, 140], [1030, 190], [1010, 660], [200, 610]], np.float32)
    import cv2

    cv2.fillConvexPoly(img, quad.astype(np.int32), (235, 235, 235))
    return img, order_corners(quad)


def test_order_corners():
    pts = np.array([[10, 10], [0, 0], [0, 10], [10, 0]], np.float32)
    tl, tr, br, bl = order_corners(pts)
    assert tuple(tl) == (0, 0) and tuple(br) == (10, 10)
    assert tuple(tr) == (10, 0) and tuple(bl) == (0, 10)


def test_warp_shape_and_content():
    img, quad = _synthetic_slide()
    out = warp_from_quad(img, quad)
    assert out.ndim == 3 and out.shape[0] > 50 and out.shape[1] > 50
    # the rectified crop should be mostly bright
    assert out.mean() > 180


def test_aspect_helpers():
    quad = np.array([[0, 0], [160, 0], [160, 100], [0, 100]], np.float32)
    assert abs(aspect_ratio(quad) - 1.6) < 1e-6
    assert aspect_score(quad) > 0.99  # 16:10 is a standard ratio
    smaller = inset_quad(quad, 0.1)
    assert smaller[0][0] > 0 and smaller[2][0] < 160  # moved inward


def test_classic_backend_detects_synthetic_slide():
    from framefit.backends import get_backend

    img, truth = _synthetic_slide()
    quad = get_backend("classic").detect(img)
    assert quad is not None and quad.shape == (4, 2)
    # detected corners should be near the true corners
    assert np.linalg.norm(quad - truth, axis=1).mean() < 25


def test_process_image_classic():
    from framefit import process_image

    img, _ = _synthetic_slide()
    r = process_image(img, backend="classic")
    assert r.ok and r.image is not None and r.aspect_score > 0.8


def test_trim_dark_margins():
    from framefit.geometry import trim_dark_margins

    # bright content with a dark uniform band on top only
    img = np.full((300, 400, 3), 230, np.uint8)
    img[:60, :, :] = 8  # dark top band
    out, (t, b, l, r) = trim_dark_margins(img)
    assert t >= 55 and b == 0 and l == 0 and r == 0
    assert out.shape[0] < img.shape[0]
    assert out.mean() > 200  # dark band removed


def test_trim_preserves_textured_dark_edge():
    from framefit.geometry import trim_dark_margins

    # a dark BUT textured top row (content) must NOT be trimmed
    img = np.full((300, 400, 3), 230, np.uint8)
    img[:40, :, :] = 8
    img[:40, ::6, :] = 250  # vertical stripes -> high variance -> content
    out, (t, _, _, _) = trim_dark_margins(img)
    assert t == 0  # textured band kept


def test_trim_preserves_dim_header_band():
    # regression: a dim-but-not-black uniform band (like a navy header, ~0.17)
    # sitting under a thin near-black margin must NOT be trimmed away.
    img = np.full((300, 400, 3), 230, np.uint8)
    img[:20, :, :] = 8    # near-black empty margin (trim this)
    img[20:70, :, :] = 44  # dim navy-ish header band ~0.17 (keep this)
    from framefit.geometry import trim_dark_margins

    _, (t, _, _, _) = trim_dark_margins(img)
    assert 15 <= t <= 22  # only the near-black margin removed, header kept
