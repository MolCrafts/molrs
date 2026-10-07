"""One public path per symbol, mirroring the Rust owner.

``molrs`` holds the subsystems and nothing else (as the Rust crate root
does); every class, function and constant is reachable at exactly one public
path, and a class's or function's ``__module__`` and ``__name__`` are that
path, so ``repr``, pickle and the docs all name it the way users import it.

The walk is over ``dir()``, not only ``__all__``: a public module exports
exactly its ``__all__`` and nothing else, so a ``typing`` or stdlib name
imported for an annotation cannot leak into it as a second public spelling.
"""

from __future__ import annotations

import importlib
import inspect
import re
from pathlib import Path
from types import ModuleType

import molrs
import pytest

SUBSYSTEMS = {
    "builder",
    "compute",
    "conformer",
    "core",
    "ff",
    "io",
    "md",
    "op",
    "optimize",
    "perceive",
    "signal",
    "stream",
}


def _public_names(module: ModuleType) -> set[str]:
    return {name for name in dir(module) if not name.startswith("_")}


def _modules() -> dict[str, ModuleType]:
    """Every public module, by its path."""
    out: dict[str, ModuleType] = {}

    def walk(module: ModuleType, path: str) -> None:
        out[path] = module
        for name in module.__all__:
            value = getattr(module, name)
            if inspect.ismodule(value):
                walk(value, f"{path}.{name}")

    walk(molrs, "molrs")
    return out


def _objects() -> dict[str, object]:
    """Every public class, function and constant, by its public path."""
    return {
        f"{path}.{name}": getattr(module, name)
        for path, module in _modules().items()
        for name in module.__all__
        if not inspect.ismodule(getattr(module, name))
    }


def _named(value: object) -> bool:
    """A class or a function: an object that states its own path."""
    return inspect.isclass(value) or inspect.isroutine(value)


def test_the_top_level_is_the_subsystems():
    assert set(molrs.__all__) == SUBSYSTEMS
    assert _public_names(molrs) == SUBSYSTEMS


@pytest.mark.parametrize("path", sorted(_modules()))
def test_a_module_exports_exactly_its_all(path):
    module = _modules()[path]
    assert module.__name__ == path
    assert len(module.__all__) == len(set(module.__all__)), path
    assert _public_names(module) == set(module.__all__)


def test_every_class_and_function_has_one_public_path():
    paths: dict[int, list[str]] = {}
    for path, value in _objects().items():
        if _named(value):
            paths.setdefault(id(value), []).append(path)
    assert not [p for p in paths.values() if len(p) > 1]


def test_a_class_or_function_names_its_public_path():
    wrong = [
        (path, f"{value.__module__}.{value.__name__}")
        for path, value in _objects().items()
        if _named(value) and f"{value.__module__}.{value.__name__}" != path
    ]
    assert not wrong


def test_a_constant_is_a_value_not_a_second_door():
    """A public non-class, non-function name is plain data: it re-exports no
    class or function under a second name."""
    callables = [
        path
        for path, value in _objects().items()
        if not _named(value) and callable(value)
    ]
    assert not callables


@pytest.mark.parametrize(
    "gone",
    [
        "molrs.fields",
        "molrs.io.raw",
        "molrs.keys",
        "molrs.schema",
        "molrs.md.driver",
        "molrs.compute.protocol",
        "molrs.compute.density",
        "molrs.ff.potential.protocol",
        "molrs.io._trajectory",
        # 0.16: one core; its vocabularies are core.keys / schema / constants.
        "molrs.store",
        "molrs.store.keys",
        "molrs.store.schema",
        "molrs.spatial",
        "molrs.system",
        "molrs.units",
        # 0.16 wave S2: io is one module per format.
        "molrs.io.trajectory",
        "molrs.io.log",
        "molrs.io.lammps_bond_react",
        "molrs.io._trajectory_doors",
        "molrs.io.mrec.schema",
    ],
)
def test_retired_modules_do_not_import(gone):
    with pytest.raises(ModuleNotFoundError):
        __import__(gone)


@pytest.mark.parametrize(
    "gone",
    [
        # 0.16: the top level is the subsystems only.
        "molrs.Block",
        "molrs.Box",
        "molrs.Frame",
        "molrs.Atomistic",
        "molrs.Element",
        "molrs.Trajectory",
        "molrs.fields",
        # MD integrates potentials; it defines none.
        "molrs.md.LJCut",
        "molrs.md.Potential",
        "molrs.md.Potentials",
        "molrs.md.kernel",
        # molrs.ff holds only its submodules.
        "molrs.ff.ForceField",
        "molrs.ff.Potential",
        # io: no raw layer, no aliases.
        "molrs.io.raw",
        "molrs.io.write_smiles",
        "molrs.io.parse_lammps_log_text",
        # Every file reader / writer is a function at the top of molrs.io:
        # the mrec doors are flat there ...
        "molrs.io.mrec.read",
        "molrs.io.mrec.write",
        "molrs.io.mrec.read_system",
        "molrs.io.mrec.write_system",
        "molrs.io.mrec.read_trajectory",
        "molrs.io.mrec.write_trajectory",
        "molrs.io.mrec.read_forcefield",
        "molrs.io.mrec.write_forcefield",
        "molrs.io.mrec.read_meta",
        "molrs.io.mrec_sections",
        # ... and its store reader / writer are named as in Rust.
        "molrs.io.mrec.FrameSequence",
        "molrs.io.mrec.FrameSequenceWriter",
        "molrs.io.mrec.TrajectoryReader",
        "molrs.io.mrec.TrajectoryWriter",
        # The frame-bytes codec is io's, not stream's.
        "molrs.stream.read_frame_bytes",
        "molrs.stream.write_frame_bytes",
        # Force-field files are io's; ff.forcefield is the data model only.
        "molrs.ff.forcefield.read_lammps_forcefield",
        "molrs.ff.forcefield.read_lammps_data_coeffs",
        "molrs.ff.forcefield.read_lammps_cmap",
        "molrs.ff.forcefield.read_gromacs_top_forcefield",
        "molrs.ff.forcefield.read_gromacs_system",
        "molrs.ff.forcefield.read_amber_prmtop_forcefield",
        "molrs.ff.forcefield.read_amber_prmtop_system",
        "molrs.ff.forcefield.read_forcefield_xml",
        "molrs.ff.forcefield.read_openmm_xml_forcefield",
        "molrs.ff.forcefield.write_lammps_forcefield",
        "molrs.ff.forcefield.write_lammps_forcefield_str",
        "molrs.ff.forcefield.write_lammps_data_coeffs",
        "molrs.ff.forcefield.write_lammps_cmap",
        "molrs.ff.forcefield.write_gromacs_top_forcefield",
        "molrs.ff.forcefield.write_gromacs_system",
        "molrs.ff.forcefield.write_amber_frcmod",
        "molrs.ff.forcefield.write_openmm_xml_forcefield",
        # A class of one format is that format's submodule's.
        "molrs.io.TrajectoryReader",
        "molrs.io.SmilesIr",
        "molrs.io.SmilesError",
        "molrs.io.CgSmilesIr",
        "molrs.io.CgGraph",
        "molrs.io.CgNode",
        "molrs.io.CgEdge",
        "molrs.io.CgFragmentDef",
        "molrs.io.ResolvedPair",
        "molrs.io.PairEnd",
        "molrs.io.BondingDescriptor",
        "molrs.io.BondReactTemplate",
        "molrs.io.LammpsLog",
        "molrs.io.LammpsLogHeader",
        "molrs.io.LammpsRun",
        "molrs.io.LammpsThermo",
        "molrs.io.LammpsWarning",
        "molrs.io.LammpsPerformance",
        "molrs.io.LammpsTimingBreakdown",
        "molrs.io.LammpsTimingRow",
        "molrs.io.LammpsCpuUse",
        "molrs.io.LammpsLoadBalance",
        "molrs.io.LammpsLoopTime",
        "molrs.io.LammpsMemoryUsage",
        "molrs.io.LammpsNeighborStatistics",
        # One core: no top-level store / spatial / system / units.
        "molrs.store",
        "molrs.spatial",
        "molrs.system",
        "molrs.units",
        # The graph is MolGraph, as in Rust.
        "molrs.core.Graph",
        # Constants are core.constants'; ff.params holds tables only.
        "molrs.core.AMBER_COULOMB",
        "molrs.ff.params.AMBER_SCEE",
        "molrs.ff.params.AMBER_SCNB",
        # The record's version is io.mrec's, not its schema checker's.
        "molrs.io.mrec.validation.MOLREC_VERSION",
        "molrs.io.mrec.validation.RESERVED_META_KEYS",
        # The force-field <-> section mapping is the section's.
        "molrs.ff.forcefield.ForceField.to_section",
        "molrs.ff.forcefield.ForceField.from_section",
        # Wave S2: no format dispatch; every door names its format.
        "molrs.io.read_frame",
        "molrs.io.write_frame",
        "molrs.io.read_frame_bytes",
        "molrs.io.write_frame_bytes",
        # ... in-memory doors are read_<fmt>_str / _bytes ...
        "molrs.io.read_smiles",
        "molrs.io.write_smarts",
        "molrs.io.smiles.SmilesIr.write_smiles",
        "molrs.io.smiles.SmilesIr.write_smarts",
        "molrs.io.read_block_csv",
        "molrs.io.write_block_csv",
        # ... family formats carry the family's name ...
        "molrs.io.read_chgcar",
        "molrs.io.read_ac",
        "molrs.io.read_prep",
        "molrs.io.write_prep",
        "molrs.io.write_bond_react_map",
        "molrs.io.read_mrec",
        "molrs.io.write_mrec",
        "molrs.io.mrec.pack",
        # ... force-field doors name the format, never `_ff` ...
        "molrs.io.read_amber_prmtop_ff",
        "molrs.io.read_gromacs_top_ff",
        "molrs.io.write_gromacs_top_ff",
        "molrs.io.read_forcefield_xml",
        "molrs.io.write_forcefield_xml",
        "molrs.io.read_opls_xml",
        # ... and a door that returns a ForceField says so.
        "molrs.io.read_lammps_cmap",
        "molrs.io.write_lammps_cmap",
        # ... and a class lives in its own format's module.
        "molrs.io.smiles.CGSmilesIR",
        "molrs.io.smiles.CGGraph",
        # Acronyms are cased as words: the line-notation IRs and records.
        "molrs.io.smiles.SmilesIR",
        "molrs.io.cgsmiles.CGSmilesIR",
        "molrs.io.cgsmiles.CGGraph",
        "molrs.io.cgsmiles.CGNode",
        "molrs.io.cgsmiles.CGEdge",
        "molrs.io.cgsmiles.CGFragmentDef",
        "molrs.core.CGBond",
        # Acronyms are cased as words in every subpackage; counts are n_*.
        "molrs.md.MD",
        "molrs.md.MDState",
        "molrs.compute.KMeans",
        "molrs.compute.KMeansResult",
        "molrs.perceive.SmartsPattern.num_query_atoms",
        # Second doors on a class.
        "molrs.core.Trajectory.from_frames",
        "molrs.core.Trajectory.count_frames",
        "molrs.perceive.SmartsMatch.as_list",
        "molrs.perceive.SmartsMatch.as_dict",
    ],
)
def test_retired_names_are_absent(gone):
    owner_path, _, name = gone.rpartition(".")
    owner: object = molrs
    for part in owner_path.split(".")[1:]:
        owner = getattr(owner, part)
    assert not hasattr(owner, name), gone


@pytest.mark.parametrize(
    "gone",
    [
        # Kernels are <Category><Style>; the explicit-term door names its job.
        "molrs.ff.potential.LJCut",
        "molrs.ff.potential.kernel",
        "molrs.ff.potential.TypedPotentials",
        "molrs.ff.potential.PairLjCut.eval",
        "molrs.ff.potential.PairLjCut.eval_table",
        "molrs.ff.potential.PairLjCut.eval_pairs",
        "molrs.ff.potential.PairLjCut.pair_eval",
        # The IR's Python names are the Rust ones; refusals end in Error.
        "molrs.ff.ir.Param",
        "molrs.ff.ir.StyleInfo",
        "molrs.ff.ir.CategoryInfo",
        "molrs.ff.ir.unregister",
        "molrs.ff.ir.Arity",
        "molrs.ff.ir.Dim",
        "molrs.ff.ir.Sealed",
        # One accessor per question on the force-field model.
        "molrs.ff.forcefield.Type",
        "molrs.ff.forcefield.Style.types",
        # Typifiers: acronyms cased as words, assign -> TypeAssignment.
        "molrs.ff.typifier.Match",
        "molrs.ff.typifier.OPLSAATypifier",
        "molrs.ff.typifier.MMFF94Typifier",
        "molrs.ff.typifier.MMFF94STypifier",
        "molrs.ff.typifier.Typifier.match",
        "molrs.ff.typifier.Typifier.library",
        # CL&Pol scaling is its own module; its table is a params table.
        "molrs.ff.scale_lj",
        "molrs.ff.clpol_scaling.fragment_scaling_data",
    ],
)
def test_force_field_names_retired_by_wave_s3_are_absent(gone):
    owner_path, _, name = gone.rpartition(".")
    owner: object = molrs
    for part in owner_path.split(".")[1:]:
        owner = getattr(owner, part)
    assert not hasattr(owner, name), gone


# --- One shape per file-format factory --------------------------------------
#
# A factory that reads or writes a file format is either a function at the
# top of molrs.io (``read_<fmt>[_<what>]`` / ``write_<fmt>[_<what>]``) or a
# class of the format's own submodule (``molrs.io.<fmt>.<Fmt>Reader`` /
# ``<Fmt>Writer``). Nothing else may carry those names.


def _factory_name(name: str) -> bool:
    return name.startswith(("read_", "write_"))


def test_read_and_write_functions_live_at_the_top_of_io():
    stray = sorted(
        path
        for path, value in _objects().items()
        if callable(value)
        and _factory_name(path.rpartition(".")[2])
        and path.rpartition(".")[0] != "molrs.io"
    )
    assert not stray


def test_no_class_hides_a_read_or_write_factory():
    """A ``read_*`` / ``write_*`` static or class method is a second door
    onto a format: the door is a function of molrs.io."""
    doors = []
    for path, value in _objects().items():
        if not inspect.isclass(value):
            continue
        for name, attr in vars(value).items():
            if _factory_name(name) and isinstance(attr, (staticmethod, classmethod)):
                doors.append(f"{path}.{name}")
    assert not doors


def test_readers_and_writers_are_classes_of_a_format_submodule():
    stray = []
    for path, value in _objects().items():
        if not inspect.isclass(value) or not value.__name__.endswith(("Reader", "Writer")):
            continue
        owner = path.rpartition(".")[0]
        if not (owner.startswith("molrs.io.") and owner.count(".") == 2):
            stray.append(path)
    assert not stray


def test_the_top_of_io_is_functions_and_format_submodules():
    """A class belongs to one format, so it lives in that format's
    submodule; the top of molrs.io holds the read / write functions."""
    classes = [
        name for name in molrs.io.__all__ if inspect.isclass(getattr(molrs.io, name))
    ]
    assert not classes
    functions = [
        name for name in molrs.io.__all__ if inspect.isroutine(getattr(molrs.io, name))
    ]
    assert all(_factory_name(name) for name in functions), functions


def test_forcefield_is_the_data_model_and_stream_the_transport():
    assert not [n for n in molrs.ff.forcefield.__all__ if _factory_name(n)]
    assert set(molrs.stream.__all__) <= {"ControlCommand", "Publisher"}
    assert set(molrs.io.mrec.__all__) == {
        "MOLREC_VERSION",
        "RESERVED_META_KEYS",
        "ForceFieldSection",
        "MrecReader",
        "MrecWriter",
        "SequenceSchema",
        "pack_mrec_zip",
        "section_names",
        "validation",
    }


@pytest.mark.parametrize(
    "path",
    [
        "molrs.io.read_smiles_str",
        "molrs.io.write_smiles_str",
        "molrs.io.read_cgsmiles_str",
        "molrs.io.read_mrec_frame",
        "molrs.io.write_mrec_trajectory",
        "molrs.io.read_msgpack_frame_bytes",
        "molrs.io.write_msgpack_frame_bytes",
        "molrs.io.read_json_frame_str",
        "molrs.io.write_json_frame_str",
        "molrs.io.read_csv_block",
        "molrs.io.read_csv_block_str",
        "molrs.io.write_csv_block",
        "molrs.io.write_csv_block_str",
        "molrs.io.read_lammps_forcefield",
        "molrs.io.read_openmm_xml_forcefield",
        "molrs.io.write_openmm_xml_forcefield",
        "molrs.io.read_molrs_xml_forcefield",
        "molrs.io.write_molrs_xml_forcefield",
        "molrs.io.read_lammps_log_str",
        "molrs.io.write_lammps_bond_react_map",
        "molrs.io.pdb.PdbReader",
        "molrs.io.xyz.XyzReader",
        "molrs.io.gro.GroReader",
        "molrs.io.dcd.DcdReader",
        "molrs.io.trr.TrrReader",
        "molrs.io.xtc.XtcReader",
        "molrs.io.lammps.LammpsDumpReader",
        "molrs.io.mrec.pack_mrec_zip",
        "molrs.io.mrec.MrecReader",
        "molrs.io.mrec.MrecWriter",
        "molrs.io.smiles.SmilesIr",
        "molrs.io.cgsmiles.CgSmilesIr",
        "molrs.io.lammps.LammpsLog",
        "molrs.io.lammps.BondReactTemplate",
        "molrs.io.mrec.MOLREC_VERSION",
        "molrs.io.mrec.ForceFieldSection",
        "molrs.core.Frame",
        "molrs.core.Block",
        "molrs.core.Box",
        "molrs.core.MolGraph",
        "molrs.core.Atomistic",
        "molrs.core.Element",
        "molrs.core.NeighborList",
        "molrs.core.Quantity",
        "molrs.core.keys.PORTS",
        "molrs.core.keys.FRAG_ID",
        "molrs.core.keys.LAMMPS_UNITS",
        "molrs.core.schema.ColumnSpec",
        "molrs.core.constants.AMBER_COULOMB",
        "molrs.core.constants.AMBER_SCEE",
        "molrs.core.constants.BOHR_RADIUS",
    ],
)
def test_the_one_path_exists(path):
    assert path in _objects()


# --- Wave S2: every door names its format -----------------------------------

_DOOR = re.compile(r"^(read|write)_[a-z0-9]+(_[a-z0-9]+)*$")


def _io_doors() -> dict[str, object]:
    return {
        name: getattr(molrs.io, name)
        for name in molrs.io.__all__
        if inspect.isroutine(getattr(molrs.io, name))
    }


def _parameters(function: object) -> list[str]:
    try:
        return list(inspect.signature(function).parameters)
    except (TypeError, ValueError):  # a builtin without a text signature
        return []


def test_no_door_dispatches_on_a_format_argument():
    """A door that picks a format for the caller is format dispatch: every
    door names its format, so none takes a ``format`` / ``fmt`` / ``encoding
    name`` argument."""
    dispatching = sorted(
        name
        for name, door in _io_doors().items()
        if {"format", "fmt"} & set(_parameters(door))
    )
    assert not dispatching


def test_every_door_is_read_or_write_of_a_named_format():
    doors = _io_doors()
    assert doors
    assert all(_DOOR.match(name) for name in doors), sorted(doors)
    generic = {"read_frame", "write_frame", "read", "write", "read_file", "write_file"}
    assert not generic & set(doors)


def test_memory_doors_end_in_str_or_bytes():
    """Text in memory is ``_str``, bytes ``_bytes``: always the last word, and
    never spelled another way (``_text``, ``_string``, ``from_bytes``…)."""
    names = list(_io_doors())
    for word in ("str", "bytes"):
        assert all(n.endswith(f"_{word}") for n in names if f"_{word}_" in n), word
    for spelling in ("_text", "_string", "parse_", "format_", "_from_", "_to_"):
        assert not [n for n in names if spelling in n], spelling
    assert "read_smiles_str" in names and "read_msgpack_frame_bytes" in names


@pytest.mark.parametrize(
    ("door", "reader"),
    [
        ("read_pdb_trajectory", "molrs.io.pdb.PdbReader"),
        ("read_xyz_trajectory", "molrs.io.xyz.XyzReader"),
        ("read_gro_trajectory", "molrs.io.gro.GroReader"),
        ("read_lammps_dump_trajectory", "molrs.io.lammps.LammpsDumpReader"),
        ("read_dcd_trajectory", "molrs.io.dcd.DcdReader"),
        ("read_trr_trajectory", "molrs.io.trr.TrrReader"),
        ("read_xtc_trajectory", "molrs.io.xtc.XtcReader"),
    ],
)
def test_a_trajectory_door_is_its_formats_reader(door, reader):
    """``read_<fmt>_trajectory`` returns ``molrs.io.<fmt>.<Fmt>Reader``, and the
    generic concatenating reader is gone."""
    assert reader in _objects()
    assert getattr(molrs.io, door).__doc__
    module, _, name = reader.rpartition(".")
    assert name in getattr(molrs.io, module.rpartition(".")[2]).__all__


def test_a_reader_or_writer_is_named_after_its_module_with_word_cased_acronyms():
    """``io.<fmt>.<Fmt>…Reader``: the class name starts with its module's name
    cased as a word (``PdbReader``, ``LammpsDumpReader``), never an all-caps
    acronym (``PDBReader``, ``DCDTrajReader``)."""
    wrong = []
    for path, value in _objects().items():
        if not inspect.isclass(value) or not value.__name__.endswith(("Reader", "Writer")):
            continue
        module = path.rpartition(".")[0].rpartition(".")[2]
        word = "".join(part.capitalize() for part in module.split("_"))
        if not value.__name__.startswith(word) or re.search(r"[A-Z]{2}", value.__name__):
            wrong.append(path)
    assert not wrong



# Every subpackage cases acronyms as words. Kept as their owners write them:
# numpy's ``DType``, and ``HBond`` -- H is the element, not an acronym.
_CASING_KEPT_PREFIXES = ("DType", "HBond")
_ALL_CAPS_ACRONYM = re.compile(r"[A-Z]{2,}[a-z]|[A-Z]{3,}|[a-z0-9][A-Z]{2,}$|^[A-Z]{2,}$")


def _without_kept_prefix(name: str) -> str:
    for kept in _CASING_KEPT_PREFIXES:
        if name.startswith(kept):
            return name[len(kept) :]
    return name


def test_class_names_case_acronyms_as_words():
    """``PdbReader``, ``SmilesIr``, ``Mmff94Typifier``, ``Rdf``, ``MdState``,
    ``Lbfgs``: an acronym is cased as a word, never ``PDB``, ``IR``, ``MSD`` or
    ``MD``, in every subpackage."""
    wrong = sorted(
        path
        for path, value in _objects().items()
        if inspect.isclass(value)
        and _ALL_CAPS_ACRONYM.search(_without_kept_prefix(value.__name__))
    )
    assert not wrong


def test_function_names_are_snake_case():
    """A public function is ``snake_case``: no capital letter, so no acronym
    can be spelled in capitals (``rdf``, never ``RDF``)."""
    wrong = sorted(
        path
        for path, value in _objects().items()
        if inspect.isroutine(value) and value.__name__ != value.__name__.lower()
    )
    assert not wrong


def test_smarts_is_perceptions_wholly():
    """SMARTS is generated by ``SmartsPattern.from_environment`` and written by
    ``str(pattern)``; ``molrs.io`` holds no SMARTS door."""
    assert callable(molrs.perceive.SmartsPattern.from_environment)
    assert not [n for n in molrs.io.__all__ if "smarts" in n]


def test_find_matches_has_no_mapped_shortcut():
    """``SmartsMatch.mapping`` is the one door onto the atom-map projection."""
    assert "mapped" not in molrs.perceive.SmartsPattern.find_matches.__text_signature__


def test_core_constants_mirror_rust_in_full():
    """``molrs.core.constants`` is ``molrs::core::constants``, name for name."""
    rust = Path(__file__).parents[2] / "molrs" / "src" / "core" / "constants.rs"
    names = set(re.findall(r"^pub const ([A-Z0-9_]+):", rust.read_text(), re.MULTILINE)) - {"ALL"}
    assert names
    assert set(molrs.core.constants.__all__) == names


def test_the_version_is_the_package_version():
    from importlib.metadata import version

    assert molrs.__version__ == version("molcrafts-molrs")


@pytest.mark.parametrize(
    "gone",
    [
        # Wave S6: the native module is molrs._native.
        "molrs._lib",
        # Every door names its format: gromacs_top, lammps_dump.
        "molrs.io.read_gromacs_system",
        "molrs.io.write_gromacs_system",
        "molrs.io.read_lammps_trajectory",
        "molrs.io.write_lammps_trajectory",
        # Counts are n_*.
        "molrs.core.Block.nrows",
        # Unit conversions are the unit registry's, not constants.
        "molrs.core.constants.KJ_PER_KCAL",
        "molrs.core.constants.ANGSTROM_PER_NM",
        "molrs.core.constants.ANGSTROM_PER_BOHR",
        "molrs.core.constants.ANGSTROM3_PER_CM3",
        "molrs.core.constants.ANGSTROM_M",
        "molrs.core.constants.FEMTOSECOND_S",
        "molrs.core.constants.CENTIMETER_PER_METER",
        "molrs.core.constants.OPENMM_COULOMB",
        "molrs.core.constants.GROMACS_COULOMB",
    ],
)
def test_names_retired_by_wave_s6_are_absent(gone):
    if gone == "molrs._lib":
        with pytest.raises(ImportError):
            importlib.import_module(gone)
        return
    owner_path, _, name = gone.rpartition(".")
    owner: object = molrs
    for part in owner_path.split(".")[1:]:
        owner = getattr(owner, part)
    assert not hasattr(owner, name), gone


def test_wave_s6_names_exist():
    assert molrs.io.read_gromacs_top_system
    assert molrs.io.write_gromacs_top_system
    assert molrs.io.read_lammps_dump_trajectory
    assert molrs.io.write_lammps_dump_trajectory
    assert molrs.core.Block().n_rows == 0
    assert molrs.__dict__["_native"].__name__ == "molrs._native"


def test_unit_conversions_come_from_the_unit_registry():
    units = molrs.core.UnitRegistry()
    assert units.factor("kcal", "kJ") == 4.184
    assert units.factor("nm", "angstrom") == 10.0
    assert units.factor("angstrom", "nm") == 0.1
    assert units.factor("bohr", "angstrom") == pytest.approx(0.529177210903, rel=1e-15)
    names = set(molrs.core.constants.__all__)
    assert not {n for n in names if n.endswith(("_PER_NM", "_PER_BOHR", "_PER_KCAL", "_PER_CM3"))}


def test_scripts_convert_units_through_the_registry():
    """No engine-check script spells a conversion factor or a retired constant."""
    root = Path(__file__).parents[2] / "scripts"
    retired = re.compile(
        r"\b(KJ_PER_KCAL|ANGSTROM_PER_NM|ANGSTROM_PER_BOHR|ANGSTROM3_PER_CM3|OPENMM_COULOMB|GROMACS_COULOMB)\b"
        r"|(?<![\w.])4\.184(?![\w])"
    )
    offenders = []
    for path in sorted(root.iterdir()):
        if path.suffix not in {".py", ".sh"}:
            continue
        for n, line in enumerate(path.read_text().splitlines(), 1):
            code = line.split("#", 1)[0]
            if retired.search(code):
                offenders.append(f"{path.name}:{n}: {line.strip()}")
    assert not offenders, "\n".join(offenders)


# Wave S7: ``ff``'s submodules are layers. The registry left ``ir`` for
# ``style_registry``, the compiler left ``potential`` for ``compile``, and the
# CL&Pol fragment table is ``clpol_scaling``'s.
@pytest.mark.parametrize(
    "gone",
    [
        "molrs.ff.ir.register_category",
        "molrs.ff.ir.register_style",
        "molrs.ff.ir.register_engine_form",
        "molrs.ff.ir.unregister_style",
        "molrs.ff.ir.styles",
        "molrs.ff.ir.categories",
        "molrs.ff.ir.evaluate",
        "molrs.ff.ir.StyleDeclaration",
        "molrs.ff.potential.PotentialCompiler",
        "molrs.ff.potential.compile_explicit_terms",
        "molrs.ff.params.clpol_fragment_scaling",
    ],
)
def test_names_moved_by_wave_s7_are_absent(gone):
    owner_path, _, name = gone.rpartition(".")
    owner: object = molrs
    for part in owner_path.split(".")[1:]:
        owner = getattr(owner, part)
    assert not hasattr(owner, name), gone


def test_wave_s7_ff_layers_exist():
    for name in (
        "register_category",
        "register_style",
        "register_engine_form",
        "unregister_style",
        "styles",
        "categories",
        "evaluate",
        "StyleDeclaration",
    ):
        assert hasattr(molrs.ff.style_registry, name), name
    assert molrs.ff.compile.PotentialCompiler
    assert molrs.ff.compile.compile_explicit_terms
    assert "c2c1im" in molrs.ff.clpol_scaling.fragment_table()
