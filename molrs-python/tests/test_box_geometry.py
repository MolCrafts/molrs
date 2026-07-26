import numpy as np
import pytest

import molrs


def test_box_exposes_native_minimum_image_geometry():
    box = molrs.Box.cube(10.0)
    r1 = np.array([1.0, 1.0, 1.0])
    r2 = np.array([9.0, 1.0, 1.0])
    np.testing.assert_allclose(box.shortest_vector(r1, r2), [-2.0, 0.0, 0.0])
    assert box.distance_squared(r1, r2) == pytest.approx(4.0)
    assert box.distance(r1, r2) == pytest.approx(2.0)


def test_box_exposes_native_face_distances_and_corners():
    box = molrs.Box.ortho(np.array([10.0, 20.0, 30.0]))
    np.testing.assert_allclose(box.nearest_plane_distance, [10.0, 20.0, 30.0])
    corners = box.corners()
    assert corners.shape == (8, 3)
    np.testing.assert_allclose(corners.min(axis=0), [0.0, 0.0, 0.0])
    np.testing.assert_allclose(corners.max(axis=0), [10.0, 20.0, 30.0])


def test_shortest_vector_respects_partial_pbc():
    box = molrs.Box.ortho(
        np.array([10.0, 10.0, 10.0]), pbc=np.array([True, False, True])
    )
    delta = box.shortest_vector(np.zeros(3), np.array([9.0, 9.0, 9.0]))
    np.testing.assert_allclose(delta, [-1.0, 9.0, -1.0])


def test_images_and_unwrap_round_trip_natively():
    box = molrs.Box.cube(10.0)
    unwrapped = np.array([[21.0, -9.0, 5.0], [2.0, 3.0, 34.0]])
    images = box.images(unwrapped)
    wrapped = box.wrap(unwrapped)
    np.testing.assert_array_equal(images, [[2, -1, 0], [0, 0, 3]])
    np.testing.assert_allclose(box.unwrap(wrapped, images), unwrapped)


def test_triclinic_corners_use_lattice_vectors_not_axis_lengths():
    h = np.array([[10.0, 2.0, 1.0], [0.0, 8.0, 3.0], [0.0, 0.0, 6.0]])
    box = molrs.Box(h)
    corners = box.corners()
    assert any(np.allclose(corner, h.sum(axis=1)) for corner in corners)


def test_from_bounds_and_batched_geometry():
    points = np.array([[0.0, -1.0, 0.0], [2.0, 3.0, 4.0]])
    box = molrs.Box.from_bounds(
        points,
        np.array([1.0, 2.0, 3.0]),
        np.array([True, True, True]),
    )
    np.testing.assert_allclose(box.origin, [-1.0, -3.0, -3.0])
    np.testing.assert_allclose(box.lengths, [4.0, 8.0, 10.0])

    left = np.array([[0.0, 0.0, 0.0], [3.5, 0.0, 0.0]])
    right = np.array([[1.0, 0.0, 0.0]])
    np.testing.assert_allclose(
        box.pairwise_delta(left, right), [[[1, 0, 0]], [[1.5, 0, 0]]]
    )
    np.testing.assert_allclose(box.pairwise_distances(left, right), [[1.0], [1.5]])


def test_transformed_preserves_origin_and_pbc():
    box = molrs.Box.ortho(
        np.array([2.0, 3.0, 4.0]),
        origin=np.array([1.0, 2.0, 3.0]),
        pbc=np.array([True, False, True]),
    )
    transformed = box.transformed(np.diag([2.0, 1.0, 0.5]))
    np.testing.assert_allclose(transformed.h, np.diag([4.0, 3.0, 2.0]))
    np.testing.assert_allclose(transformed.origin, box.origin)
    np.testing.assert_array_equal(transformed.pbc, box.pbc)
