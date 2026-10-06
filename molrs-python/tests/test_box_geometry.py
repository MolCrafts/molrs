import molrs
import numpy as np
import pytest


def test_box_exposes_native_minimum_image_geometry():
    box = molrs.spatial.Box.cube(10.0)
    r1 = np.array([1.0, 1.0, 1.0])
    r2 = np.array([9.0, 1.0, 1.0])
    np.testing.assert_allclose(box.shortest_vector(r1, r2), [-2.0, 0.0, 0.0])
    assert box.distance_squared(r1, r2) == pytest.approx(4.0)
    assert box.distance(r1, r2) == pytest.approx(2.0)


def test_box_exposes_native_face_distances_and_corners():
    box = molrs.spatial.Box.ortho(np.array([10.0, 20.0, 30.0]))
    np.testing.assert_allclose(box.nearest_plane_distance, [10.0, 20.0, 30.0])
    corners = box.corners()
    assert corners.shape == (8, 3)
    np.testing.assert_allclose(corners.min(axis=0), [0.0, 0.0, 0.0])
    np.testing.assert_allclose(corners.max(axis=0), [10.0, 20.0, 30.0])


def test_images_and_unwrap_round_trip_natively():
    box = molrs.spatial.Box.cube(10.0)
    unwrapped = np.array([[21.0, -9.0, 5.0], [2.0, 3.0, 34.0]])
    images = box.images(unwrapped)
    wrapped = box.wrap(unwrapped)
    np.testing.assert_array_equal(images, [[2, -1, 0], [0, 0, 3]])
    np.testing.assert_allclose(box.unwrap(wrapped, images), unwrapped)


def test_images_are_int32_like_the_frame_image_columns():
    box = molrs.spatial.Box.cube(10.0)
    images = box.images(np.array([[21.0, -9.0, 5.0]]))
    assert images.dtype == np.int32
    np.testing.assert_array_equal(images, [[2, -1, 0]])


def test_unwrap_accepts_the_int32_image_columns_of_a_frame():
    # ix/iy/iz are the schema's integer type (int32), as a LAMMPS reader
    # stores them; unwrap must take them without a cast.
    atoms = molrs.store.Block(
        {
            "x": np.array([1.0, 9.0]),
            "y": np.array([2.0, 5.0]),
            "z": np.array([3.0, 0.5]),
            "ix": np.array([1, 0], dtype=np.int32),
            "iy": np.array([0, -1], dtype=np.int32),
            "iz": np.array([0, 2], dtype=np.int32),
        }
    )
    frame = molrs.store.Frame({"atoms": atoms}, box=molrs.spatial.Box.cube(10.0))
    atoms = frame["atoms"]

    unwrapped = frame.box.unwrap(atoms["x", "y", "z"], atoms["ix", "iy", "iz"])

    # xyz + L * image, L = 10 on every axis.
    np.testing.assert_allclose(
        unwrapped, [[11.0, 2.0, 3.0], [9.0, -5.0, 20.5]], atol=1e-12
    )


def test_from_bounds_and_batched_geometry():
    points = np.array([[0.0, -1.0, 0.0], [2.0, 3.0, 4.0]])
    box = molrs.spatial.Box.from_bounds(
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


def test_from_bounds_takes_a_frame_and_a_scalar_padding():
    frame = molrs.store.Frame()
    atoms = molrs.store.Block()
    atoms.insert("x", np.array([0.0, 2.0]))
    atoms.insert("y", np.array([-1.0, 3.0]))
    atoms.insert("z", np.array([0.0, 4.0]))
    frame["atoms"] = atoms
    points = np.array([[0.0, -1.0, 0.0], [2.0, 3.0, 4.0]])

    from_frame = molrs.spatial.Box.from_bounds(frame, 1.0)
    from_points = molrs.spatial.Box.from_bounds(points, np.array([1.0, 1.0, 1.0]))
    np.testing.assert_allclose(from_frame.origin, [-1.0, -2.0, -1.0])
    np.testing.assert_allclose(from_frame.lengths, [4.0, 6.0, 6.0])
    assert from_frame.approx_eq(from_points, 0.0)


def test_from_bounds_rejects_a_frame_without_atoms_and_a_bad_padding():
    with pytest.raises(ValueError, match="atoms"):
        molrs.spatial.Box.from_bounds(molrs.store.Frame(), 1.0)
    with pytest.raises(ValueError, match="length 3"):
        molrs.spatial.Box.from_bounds(np.zeros((1, 3)), np.array([1.0, 1.0]))


def test_approx_eq_uses_an_absolute_tolerance_and_exact_pbc():
    a = molrs.spatial.Box.cube(10.0)
    b = molrs.spatial.Box.cube(10.0 + 1e-6)
    assert a.approx_eq(b, 1e-5)
    assert not a.approx_eq(b, 1e-7)
    assert not a.approx_eq(molrs.spatial.Box.cube(10.0, pbc=np.array([True, True, False])), 1.0)
    with pytest.raises(ValueError):
        a.approx_eq(b, -1.0)


def test_transformed_preserves_origin_and_pbc():
    box = molrs.spatial.Box.ortho(
        np.array([2.0, 3.0, 4.0]),
        origin=np.array([1.0, 2.0, 3.0]),
        pbc=np.array([True, False, True]),
    )
    transformed = box.transformed(np.diag([2.0, 1.0, 0.5]))
    np.testing.assert_allclose(transformed.h, np.diag([4.0, 3.0, 2.0]))
    np.testing.assert_allclose(transformed.origin, box.origin)
    np.testing.assert_array_equal(transformed.pbc, box.pbc)
