from __future__ import annotations

import unittest

from facehide.infer.device import DxgiAdapter, pick_hardware_adapter, plan_device


def _adapter(name: str, index: int, vram: int = 1) -> DxgiAdapter:
    return DxgiAdapter(name=name, dxgi_index=index, dedicated_bytes=vram)


class DevicePlanTests(unittest.TestCase):
    def test_gameviewer_then_rtx_uses_unfiltered_index_one(self) -> None:
        adapters = [
            _adapter("GameViewer Virtual Display Adapter", 0, 0),
            _adapter("NVIDIA GeForce RTX 3060 Ti", 1, 8_000_000_000),
        ]
        chosen = pick_hardware_adapter(adapters)
        self.assertIsNotNone(chosen)
        assert chosen is not None
        self.assertEqual(chosen.dxgi_index, 1)
        self.assertNotEqual(chosen.dxgi_index, 0)
        plan = plan_device("auto", adapters=adapters, dml_available=True, ort_available=True)
        self.assertTrue(plan.use_dml)
        self.assertEqual(plan.dxgi_index, 1)
        self.assertEqual(plan.probe_ids, ())

    def test_basic_render_only_goes_cpu_without_probe(self) -> None:
        adapters = [_adapter("Microsoft Basic Render Driver", 0, 0)]
        plan = plan_device("auto", adapters=adapters, dml_available=True, ort_available=True)
        self.assertFalse(plan.use_dml)
        self.assertIsNone(plan.dxgi_index)
        self.assertEqual(plan.probe_ids, ())
        self.assertEqual(plan.reason, "no_hardware")

    def test_enum_failure_probes_zero_then_one(self) -> None:
        plan = plan_device(
            "gpu",
            adapters=None,
            enum_failed=True,
            dml_available=True,
            ort_available=True,
        )
        self.assertTrue(plan.use_dml)
        self.assertEqual(plan.probe_ids, (0, 1))

    def test_cpu_never_uses_dml(self) -> None:
        adapters = [_adapter("NVIDIA GeForce RTX 3060 Ti", 0, 8)]
        plan = plan_device("cpu", adapters=adapters, dml_available=True, ort_available=True)
        self.assertFalse(plan.use_dml)
        self.assertEqual(plan.reason, "user_cpu")
