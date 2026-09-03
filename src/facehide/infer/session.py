from __future__ import annotations

from pathlib import Path

import numpy as np

from facehide.infer.types import InferenceFn
from facehide.models import SFACE_NAME
from facehide.threads import intra_op


class OrtSessionFactory:
    def __init__(
        self,
        runner: InferenceFn | None = None,
        *,
        input_name: str = "input",
        output_names: list[str] | None = None,
    ) -> None:
        self._runner = runner
        self.input_name = input_name
        self.output_names = list(output_names or [])

    def is_stub(self) -> bool:
        return self._runner is not None

    def make(
        self,
        model_path: Path | str,
        providers: list,
        *,
        input_name: str | None = None,
        output_names: list[str] | None = None,
    ) -> InferenceFn:
        if self._runner is not None:
            name = Path(model_path).name.lower()
            if "sface" in name or "recognition" in name:
                return _BoundRunner(self._runner, "data")
            return _BoundRunner(self._runner, input_name or self.input_name or "input")
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = intra_op()
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.enable_mem_pattern = False
        session = ort.InferenceSession(str(model_path), sess_options=so, providers=providers)
        return _OrtRunner(session)

    def providers_for(self, dxgi_index: int) -> list:
        return [
            ("DmlExecutionProvider", {"device_id": int(dxgi_index)}),
            "CPUExecutionProvider",
        ]

    def cpu_providers(self) -> list:
        return ["CPUExecutionProvider"]


class _BoundRunner:
    def __init__(self, inner: InferenceFn, input_name: str) -> None:
        self._inner = inner
        self.input_name = input_name
        self.output_names = list(getattr(inner, "output_names", []) or [])

    def run(
        self,
        output_names: list[str] | None,
        input_feed: dict[str, np.ndarray],
    ) -> list[np.ndarray]:
        return self._inner.run(output_names, input_feed)


class _OrtRunner:
    def __init__(self, session) -> None:
        self._session = session
        self.input_name = session.get_inputs()[0].name
        self.output_names = [item.name for item in session.get_outputs()]

    def run(
        self,
        output_names: list[str] | None,
        input_feed: dict[str, np.ndarray],
    ) -> list[np.ndarray]:
        return self._session.run(output_names, input_feed)


def sanitize_sface(src: Path, dest: Path) -> Path:
    if dest.is_file() and dest.stat().st_size >= 1_000_000:
        return dest
    try:
        import onnx
    except ImportError:
        return src
    model = onnx.load(str(src))
    names = {init.name for init in model.graph.initializer}
    keep = [inp for inp in model.graph.input if inp.name not in names]
    if len(keep) == len(model.graph.input):
        return src
    while len(model.graph.input):
        model.graph.input.pop()
    model.graph.input.extend(keep)
    dest.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(dest))
    return dest


def sface_model_path(canonical: Path) -> Path:
    dest = canonical.with_name(canonical.stem + ".ortin.onnx")
    if canonical.name != SFACE_NAME:
        dest = canonical.with_suffix(".ortin.onnx")
    return sanitize_sface(canonical, dest)
