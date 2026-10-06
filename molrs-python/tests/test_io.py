"""FFI smoke tests for top-level IO readers/writers.

Self-contained: every fixture is written by molrs itself. No external corpus,
no third-party scientific software.
"""

from __future__ import annotations

from io import StringIO

import molrs
import numpy as np
import pytest


class TestAmberAliasDeleted:
    def test_read_prmtop_and_read_inpcrd_are_gone(self):
        assert not hasattr(molrs.io, "read_prmtop")
        assert not hasattr(molrs.io, "read_inpcrd")
        assert "read_prmtop" not in molrs.io.__all__
        assert not hasattr(molrs.io.raw, "read_prmtop")
        assert not hasattr(molrs._lib, "read_prmtop")
        assert callable(molrs.io.read_amber_prmtop)
        assert callable(molrs.io.read_amber_inpcrd)


class TestErrorMessages:
    def test_pyo3_type_error_names_the_argument(self):
        with pytest.raises(TypeError, match="center"):
            molrs.Sphere("not-an-array", 1.0)


class TestReadPdb:
    def test_basic(self, water_pdb):
        frame = molrs.io.raw.read_pdb(str(water_pdb))
        assert "atoms" in frame
        assert frame["atoms"].nrows == 3

    def test_has_coordinates(self, water_pdb):
        frame = molrs.io.raw.read_pdb(str(water_pdb))
        atoms = frame["atoms"]
        assert atoms["x"] is not None
        assert atoms["y"] is not None
        assert atoms["z"] is not None

    def test_missing_file_raises_os_error(self):
        with pytest.raises(OSError):
            molrs.io.raw.read_pdb("/nonexistent/path.pdb")

    def test_missing_file_names_the_path(self):
        with pytest.raises(OSError, match="missing.pdb"):
            molrs.io.read_pdb("missing.pdb")


class TestReadGro:
    def test_native_basic(self, water_gro):
        frames = molrs.io.raw.read_gro_trajectory(str(water_gro))
        assert len(frames) == 1
        f0 = frames[0]
        assert "atoms" in f0
        assert f0["atoms"].nrows == 3
        assert f0.box is not None

    def test_native_columns(self, water_gro):
        frames = molrs.io.raw.read_gro_trajectory(str(water_gro))
        atoms = frames[0]["atoms"]
        # The reader emits canonical names directly; `resid`/`atom_id` were
        # format-native spellings that something downstream had to rename, and
        # that rename is now a write into a UInt key an Int column cannot pass.
        for col in ["res_id", "res_name", "name", "id", "x", "y", "z"]:
            assert col in atoms, f"missing column: {col}"

    def test_facade_canonical_columns(self, water_gro):
        atoms = molrs.io.read_gro(str(water_gro))["atoms"]
        for col in ["res_id", "res_name", "name", "id", "x", "y", "z"]:
            assert col in atoms, f"missing canonical column: {col}"

    def test_facade_no_format_native_columns(self, water_gro):
        atoms = molrs.io.read_gro(str(water_gro))["atoms"]
        for col in ["resid", "atom_name", "atom_id"]:
            assert col not in atoms, f"format-native column leaked: {col}"

    def test_round_trip(self, water_gro, tmp_path):
        f0 = molrs.io.read_gro(str(water_gro))
        out = tmp_path / "out.gro"
        molrs.io.write_gro(out, f0)
        # A write must not rename the caller's columns.
        assert "res_name" in f0["atoms"] and "resname" not in f0["atoms"]
        f1 = molrs.io.read_gro(out)
        assert f0["atoms"].nrows == f1["atoms"].nrows
        assert list(f1["atoms"]["res_id"]) == list(f0["atoms"]["res_id"])
        assert list(f1["atoms"]["id"]) == list(f0["atoms"]["id"])

    def test_trajectory_round_trip(self, water_gro, tmp_path):
        f0 = molrs.io.read_gro(str(water_gro))
        out = tmp_path / "traj.gro"
        molrs.io.write_gro_trajectory(out, [f0, f0])
        frames = molrs.io.read_gro_trajectory(out)
        assert len(frames) == 2
        assert frames[1]["atoms"].nrows == f0["atoms"].nrows

    def test_missing_file_raises_os_error(self):
        with pytest.raises(OSError):
            molrs.io.raw.read_gro_trajectory("/nonexistent/path.gro")


class TestReadXyz:
    def test_basic(self, water_xyz):
        frame = molrs.io.raw.read_xyz(str(water_xyz))
        assert "atoms" in frame
        assert frame["atoms"].nrows == 3

    def test_has_coordinates(self, water_xyz):
        frame = molrs.io.raw.read_xyz(str(water_xyz))
        atoms = frame["atoms"]
        assert atoms["x"] is not None

    def test_missing_file_raises_os_error(self):
        with pytest.raises(OSError):
            molrs.io.raw.read_xyz("/nonexistent/path.xyz")

    def test_trajectory_round_trip(self, water_xyz, tmp_path):
        frame = molrs.io.read_xyz(str(water_xyz))
        out = tmp_path / "traj.xyz"
        molrs.io.write_xyz_trajectory(out, [frame, frame, frame])
        with molrs.io.read_xyz_trajectory(out) as reader:
            assert reader.n_frames == 3
            assert reader[2]["atoms"].nrows == frame["atoms"].nrows


def test_every_trajectory_reader_has_its_writer() -> None:
    """Names pair: a ``read_X_trajectory`` door has a ``write_X_trajectory``."""
    public = set(molrs.io.__all__)
    for name in public:
        if name.startswith("read_") and name.endswith("_trajectory"):
            assert "write_" + name[len("read_") :] in public, name


def test_read_stl_gives_a_watertight_mesh(tmp_path) -> None:
    import numpy as np

    # A closed tetrahedron in ASCII STL, written in-process.
    verts = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    faces = [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]
    lines = ["solid tet"]
    for f in faces:
        lines.append("  facet normal 0 0 0")
        lines.append("    outer loop")
        for i in f:
            lines.append("      vertex {} {} {}".format(*verts[i]))
        lines.append("    endloop")
        lines.append("  endfacet")
    lines.append("endsolid tet")
    path = tmp_path / "tet.stl"
    path.write_text("\n".join(lines) + "\n")

    mesh = molrs.io.read_stl(str(path))
    assert isinstance(mesh, molrs.TriMesh)
    assert mesh.n_faces == 4 and mesh.n_vertices == 4
    assert mesh.is_watertight()
    tet = molrs.Polyhedron(mesh.scaled(2.0))
    assert tet.contains(np.array([[0.2, 0.2, 0.2]]))[0]
    assert not tet.contains(np.array([[3.0, 3.0, 3.0]]))[0]


class TestBlockCsv:
    """``molrs.io.read_block_csv`` / ``write_block_csv`` — CSV for one Block."""

    def test_headered_round_trip(self):
        src = molrs.Block(
            {
                "x": [1.0, 2.0],
                "id": np.array([10, 20], dtype=np.uint32),
                "name": ["p", "q"],
            }
        )
        rt = molrs.io.read_block_csv(StringIO(molrs.io.write_block_csv(src)))
        np.testing.assert_allclose(rt["x"], [1.0, 2.0])
        np.testing.assert_array_equal(rt["id"], [10, 20])
        assert list(rt["name"]) == ["p", "q"]

    def test_dtype_inference(self):
        rt = molrs.io.read_block_csv(StringIO("a,b,c\n1,1.5,x\n2,2.5,y\n"))
        assert str(rt["a"].dtype).startswith("int")
        assert str(rt["b"].dtype).startswith("float")
        assert list(rt["c"]) == ["x", "y"]

    def test_headerless_with_names(self):
        rt = molrs.io.read_block_csv(StringIO("1,2\n3,4\n"), header=["a", "b"])
        np.testing.assert_array_equal(rt["a"], [1, 3])
        np.testing.assert_array_equal(rt["b"], [2, 4])

    def test_empty_csv_raises_value_error(self):
        with pytest.raises(ValueError):
            molrs.io.read_block_csv(StringIO(""))

    def test_no_header(self):
        b = molrs.Block({"count": np.array([1, 2], dtype=np.int64)})
        text = molrs.io.write_block_csv(b, header=False)
        assert "count" not in text.splitlines()[0]

    def test_writes_a_file(self, tmp_path):
        path = tmp_path / "out.csv"
        assert molrs.io.write_block_csv(molrs.Block({"x": [1.0, 2.0]}), path) is None
        np.testing.assert_allclose(molrs.io.read_block_csv(path)["x"], [1.0, 2.0])


class TestCanonicalNativeColumns:
    """Readers whose native columns are the canonical ones are re-exported
    by identity: no Python renaming layer."""

    def test_mol2_reader_and_writer_are_the_compiled_functions(self):
        assert molrs.io.read_mol2 is molrs._lib.read_mol2
        assert molrs.io.write_mol2 is molrs._lib.write_mol2
        assert molrs.io.read_xyz is molrs._lib.read_xyz
        assert molrs.io.read_lammps_data is molrs._lib.read_lammps_data

    def test_mol2_columns_are_canonical_and_round_trip(self, tmp_path):
        path = tmp_path / "x.mol2"
        path.write_text(
            "@<TRIPOS>MOLECULE\nX\n2 1 2 0 0\nSMALL\nNO_CHARGES\n@<TRIPOS>ATOM\n"
            "1 C1 0.0 0.0 0.0 C.ar 7 BEN 0.0\n2 N1 1.4 0.0 0.0 N.am 9 AMD 0.0\n"
            "@<TRIPOS>BOND\n1 1 2 am\n"
        )
        frame = molrs.io.read_mol2(path)
        atoms = frame["atoms"]
        assert list(atoms["type"]) == ["C.ar", "N.am"]
        assert list(atoms["res_id"]) == [7, 9]
        assert list(frame["bonds"]["type"]) == ["am"]
        out = tmp_path / "y.mol2"
        molrs.io.write_mol2(out, frame)
        back = molrs.io.read_mol2(out)
        assert list(back["atoms"]["res_name"]) == ["BEN", "AMD"]

    def test_extxyz_species_is_element_and_wide_properties_stay_whole(self, tmp_path):
        path = tmp_path / "f.xyz"
        path.write_text(
            "2\nProperties=species:S:1:pos:R:3:forces:R:3\n"
            "O 0 0 0 0.1 0.2 0.3\nH 1 0 0 -0.1 -0.2 -0.3\n"
        )
        atoms = molrs.io.read_xyz(path)["atoms"]
        assert list(atoms["element"]) == ["O", "H"]
        assert atoms["forces"].shape == (2, 3)
        assert "forces_1" not in atoms


class TestLammpsDataLabels:
    LABELLED = (
        "test\n\n3 atoms\n1 bonds\n2 atom types\n1 bond types\n\n"
        "0 10 xlo xhi\n0 10 ylo yhi\n0 10 zlo zhi\n\n"
        "Atom Type Labels\n\n1 OW\n2 HW\n\nBond Type Labels\n\n1 OW-HW\n\n"
        "Atoms # full\n\n1 1 1 -0.8 0 0 0\n2 1 2 0.4 1 0 0\n3 1 2 0.4 0 1 0\n\n"
        "Bonds\n\n1 1 1 2\n"
    )

    def test_typed_blocks_carry_string_types(self, tmp_path):
        path = tmp_path / "w.data"
        path.write_text(self.LABELLED)
        frame = molrs.io.read_lammps_data(path)
        assert list(frame["atoms"]["type"]) == ["OW", "HW", "HW"]
        assert list(frame["atoms"]["type_id"]) == [1, 2, 2]
        assert list(frame["bonds"]["type"]) == ["OW-HW"]

    def test_atom_style_sets_the_atoms_layout(self, tmp_path):
        path = tmp_path / "a.data"
        path.write_text(
            "t\n\n1 atoms\n1 atom types\n\n0 1 xlo xhi\n0 1 ylo yhi\n0 1 zlo zhi\n\n"
            "Atoms # full\n\n1 1 0.1 0.2 0.3\n"
        )
        atoms = molrs.io.read_lammps_data(path, atom_style="atomic")["atoms"]
        assert "charge" not in atoms and "mol_id" not in atoms
        with pytest.raises(OSError, match="atom style"):
            molrs.io.read_lammps_data(path, atom_style="nonsense")

    def test_write_declares_extra_type_labels(self, tmp_path):
        src = tmp_path / "w.data"
        src.write_text(self.LABELLED)
        frame = molrs.io.read_lammps_data(src)
        out = tmp_path / "o.data"
        molrs.io.write_lammps_data(out, frame, type_labels={"atoms": ["OX"]})
        text = out.read_text()
        assert "3 atom types" in text
        assert "Atom Type Labels\n\n1 HW\n2 OW\n3 OX\n" in text
        with pytest.raises(ValueError, match="empty"):
            molrs.io.write_lammps_data(out, frame, type_labels={"atoms": [""]})


class TestInpcrdIntoFrame:
    TEXT = (
        "two atoms\n  2\n"
        "  1.0000000   2.0000000   3.0000000   4.0000000   5.0000000   6.0000000\n"
    )

    def test_coordinates_go_into_the_given_frame(self, tmp_path):
        path = tmp_path / "x.inpcrd"
        path.write_text(self.TEXT)
        frame = molrs.Frame()
        atoms = molrs.Block()
        atoms.insert("x", np.zeros(2))
        atoms.insert("charge", np.array([0.5, -0.5]))
        frame["atoms"] = atoms
        same = molrs.io.read_amber_inpcrd(path, frame)
        assert same is frame
        np.testing.assert_allclose(frame["atoms"]["z"], [3.0, 6.0])
        np.testing.assert_allclose(frame["atoms"]["charge"], [0.5, -0.5])
        assert frame.meta["title"] == "two atoms"

    def test_a_count_mismatch_leaves_the_frame_alone(self, tmp_path):
        path = tmp_path / "x.inpcrd"
        path.write_text(self.TEXT)
        frame = molrs.Frame()
        atoms = molrs.Block()
        atoms.insert("x", np.zeros(3))
        frame["atoms"] = atoms
        with pytest.raises(OSError, match="rows"):
            molrs.io.read_amber_inpcrd(path, frame)
        assert "y" not in frame["atoms"]
