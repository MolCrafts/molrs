"""molrs.compute.Compute is a runtime_checkable Protocol."""

from __future__ import annotations

import molrs
from molrs.compute import Compute


class OnlyCompute:
    def compute(self, *args, **kwargs):
        return None


class TestComputeProtocol:
    def test_structural_class_satisfies_the_protocol(self) -> None:
        assert isinstance(OnlyCompute(), Compute)

    def test_kernels_satisfy_the_protocol_without_modification(self) -> None:
        # Kernels live in the subpackages, one per molrs::compute domain.
        satisfied = 0
        for sub in (
            getattr(molrs.compute, n)
            for n in dir(molrs.compute)
            if not n.startswith("_")
        ):
            if not hasattr(sub, "__dict__"):
                continue
            for name in dir(sub):
                obj = getattr(sub, name, None)
                if isinstance(obj, type) and callable(getattr(obj, "compute", None)):
                    # Instantiate is not required: the protocol is structural on instances.
                    # Check the class itself declares compute.
                    satisfied += 1
        assert satisfied >= 1

    def test_contract_is_compute_only(self) -> None:
        members = {n for n in Compute.__dict__ if not n.startswith("_")}
        assert "compute" in members
        assert "dump" not in members
        assert "__call__" not in members

    def test_layout_doc_records_the_protocol_exception(self) -> None:
        assert molrs.compute.__doc__ is not None
        assert "protocol" in molrs.compute.__doc__.lower()

    def test_presence_only_isinstance_ignores_signature(self) -> None:
        class WrongSig:
            def compute(self) -> int:
                return 1

        assert isinstance(WrongSig(), Compute)
