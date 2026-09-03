from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from facehide.gallery import MatchResult, Person, best_match
from facehide.infer.opencv_backend import OpenCvSfaceRecognizer, OpenCvYunetDetector
from facehide.infer.ort_backend import OrtSfaceRecognizer, OrtYunetDetector
from facehide.infer.preprocess import CANVAS, fit_canvas, unpad_rows
from facehide.infer.session import OrtSessionFactory, sface_model_path
from facehide.infer.types import BackendInfo, Detection, DmlDeadFlag, normalize_device
from facehide.models import ensure_models, sface_path, yunet_path
from facehide.threads import apply as apply_threads

# Live OpenCV YuNet longest side. 640 keeps default 640×480 camera native
# (matches enroll geometry). Larger frames still downscale.
DETECT_MAX_SIDE = 640


class NoFaceError(RuntimeError):
    pass


def detect_working_size(
    width: int,
    height: int,
    max_side: int = DETECT_MAX_SIDE,
) -> tuple[int, int]:
    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0:
        return max(1, width), max(1, height)
    if max_side <= 0:
        return width, height
    longest = max(width, height)
    if longest <= max_side:
        return width, height
    scale = max_side / float(longest)
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def working_view(bgr: np.ndarray, max_side: int) -> tuple[np.ndarray, float, float]:
    height, width = bgr.shape[:2]
    if max_side <= 0:
        return bgr, 1.0, 1.0
    work_w, work_h = detect_working_size(width, height, max_side)
    if work_w == width and work_h == height:
        return bgr, 1.0, 1.0
    work = cv2.resize(bgr, (work_w, work_h), interpolation=cv2.INTER_AREA)
    return work, width / float(work_w), height / float(work_h)


def map_box_to_source(
    x: int,
    y: int,
    w: int,
    h: int,
    src_size: tuple[int, int],
    work_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    src_w, src_h = src_size
    work_w, work_h = work_size
    if work_w <= 0 or work_h <= 0:
        return x, y, max(1, w), max(1, h)
    scale_x = src_w / float(work_w)
    scale_y = src_h / float(work_h)
    return (
        int(round(x * scale_x)),
        int(round(y * scale_y)),
        max(1, int(round(w * scale_x))),
        max(1, int(round(h * scale_y))),
    )


def scale_face_row(face: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    out = np.array(face, dtype=np.float32, copy=True)
    if scale_x == 1.0 and scale_y == 1.0:
        return out
    out[0] *= scale_x
    out[1] *= scale_y
    out[2] *= scale_x
    out[3] *= scale_y
    if out.size >= 14:
        out[4:14:2] *= scale_x
        out[5:14:2] *= scale_y
    return out


def should_extract_features(face_count: int) -> bool:
    return face_count > 0


@dataclass
class FaceHit:
    x: int
    y: int
    w: int
    h: int
    det_score: float
    raw: np.ndarray
    feature: np.ndarray | None
    match: MatchResult | None = None


def hits_from_detections(
    faces: np.ndarray | None,
    *,
    scale_x: float,
    scale_y: float,
    extract_features: bool,
    feature_fn: Callable[[np.ndarray], np.ndarray] | None,
) -> list[FaceHit]:
    if faces is None:
        return []
    rows = np.asarray(faces)
    if rows.size == 0:
        return []
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    count = len(rows)
    want_features = bool(extract_features) and should_extract_features(count) and feature_fn is not None
    hits: list[FaceHit] = []
    for face in rows:
        scaled = scale_face_row(face, scale_x, scale_y)
        x, y, fw, fh = [int(round(v)) for v in scaled[:4]]
        feature = feature_fn(scaled) if want_features else None
        hits.append(
            FaceHit(
                x=x,
                y=y,
                w=max(1, fw),
                h=max(1, fh),
                det_score=float(scaled[-1]) if scaled.size else 0.0,
                raw=scaled,
                feature=feature,
            )
        )
    return hits


def crop_face(bgr: np.ndarray, hit: FaceHit, pad: float = 0.25) -> np.ndarray:
    h, w = bgr.shape[:2]
    px = int(hit.w * pad)
    py = int(hit.h * pad)
    x1 = max(0, hit.x - px)
    y1 = max(0, hit.y - py)
    x2 = min(w, hit.x + hit.w + px)
    y2 = min(h, hit.y + hit.h + py)
    crop = bgr[y1:y2, x1:x2]
    if crop.size == 0:
        raise NoFaceError("人脸裁剪失败")
    return crop


class FaceEngine:
    def __init__(
        self,
        det_model: Path | None = None,
        rec_model: Path | None = None,
        detect_score: float = 0.7,
        device: str = "auto",
        session_factory: OrtSessionFactory | None = None,
        adapter_enum=None,
    ) -> None:
        self._det_model = Path(det_model) if det_model else yunet_path()
        self._rec_model = Path(rec_model) if rec_model else sface_path()
        self.detect_score = detect_score
        self._lock = threading.Lock()
        self._device = normalize_device(device)
        self._factory = session_factory or OrtSessionFactory()
        self._enum = adapter_enum
        self._flag = DmlDeadFlag()
        self._enroll_det: OpenCvYunetDetector | None = None
        self._enroll_rec: OpenCvSfaceRecognizer | None = None
        self._live_det = None
        self._live_rec = None
        self._live_is_dml = False
        self._live_info: BackendInfo | None = None
        self._fallback_logged = False
        self.extract_count = 0

    def ensure_ready(self) -> None:
        with self._lock:
            self._ensure_unlocked(for_enroll=True)

    def backend_info(self) -> BackendInfo:
        with self._lock:
            if self._live_info is not None:
                return self._live_info
            return BackendInfo("opencv", "opencv", "CPU", "CPU", None, None, True)

    def consume_fallback(self) -> str | None:
        with self._lock:
            if self._flag.dead and not self._fallback_logged:
                self._fallback_logged = True
                return self._flag.error or "DirectML"
            return None

    def reconfigure(self, device: str) -> BackendInfo:
        with self._lock:
            self._device = normalize_device(device)
            if self._device in ("gpu", "auto"):
                self._flag.clear()
            self._live_det = None
            self._live_rec = None
            self._live_is_dml = False
            self._live_info = None
            self._fallback_logged = False
            if self._live_info is None:
                return BackendInfo("opencv", "opencv", "CPU", "CPU", None, None, True)
            return self._live_info

    def _ensure_unlocked(self, *, for_enroll: bool = False) -> None:
        if not self._det_model.exists() or not self._rec_model.exists():
            det, rec = ensure_models()
            self._det_model, self._rec_model = det, rec
        if self._enroll_det is None or self._enroll_rec is None:
            self._enroll_det = OpenCvYunetDetector(self._det_model, self.detect_score)
            self._enroll_rec = OpenCvSfaceRecognizer(self._rec_model)
        if for_enroll:
            return
        if self._flag.dead:
            self._cpu_live()
            return
        if self._live_det is not None and self._live_rec is not None:
            return
        self._build_live()

    def _cpu_live(self) -> None:
        assert self._enroll_det is not None and self._enroll_rec is not None
        self._live_det = self._enroll_det
        self._live_rec = self._enroll_rec
        self._live_is_dml = False
        self._live_info = BackendInfo("opencv", "opencv", "CPU", "CPU", None, None, True)
        apply_threads(dml_active=False)

    def _build_live(self) -> None:
        if self._device == "cpu" and not self._factory.is_stub():
            self._cpu_live()
            return
        try:
            self._try_dml()
        except Exception as exc:
            self._flag.trip(str(exc))
            self._cpu_live()

    def _attach_dml(self, yunet_fn, sface_fn, info: BackendInfo) -> None:
        assert self._enroll_rec is not None
        y_in = getattr(yunet_fn, "input_name", None) or "input"
        s_in = getattr(sface_fn, "input_name", None) or "data"
        yunet_fn.run(None, {y_in: np.zeros((1, 3, 640, 640), dtype=np.float32)})
        sface_fn.run(None, {s_in: np.zeros((1, 3, 112, 112), dtype=np.float32)})
        self._live_det = OrtYunetDetector(yunet_fn, flag=self._flag, info=info)
        self._live_rec = OrtSfaceRecognizer(
            sface_fn, aligner=self._enroll_rec, flag=self._flag, info=info
        )
        self._live_info = info
        self._live_is_dml = True
        apply_threads(dml_active=True)

    def _try_dml(self) -> None:
        from facehide.infer.device import plan_live_device

        assert self._enroll_rec is not None
        if self._factory.is_stub():
            yunet_fn = self._factory.make(self._det_model, [])
            sface_fn = self._factory.make(self._rec_model, [])
            info = BackendInfo("yunet", "sface", "DmlExecutionProvider", "stub", 1, None, False)
            self._attach_dml(yunet_fn, sface_fn, info)
            return
        plan = plan_live_device(self._device, enum_fn=self._enum)
        if not plan.use_dml:
            self._cpu_live()
            return
        ids = plan.probe_ids or ((plan.dxgi_index,) if plan.dxgi_index is not None else ())
        last_error: Exception | None = None
        for device_id in ids:
            try:
                providers = self._factory.providers_for(int(device_id))
                yunet_fn = self._factory.make(self._det_model, providers)
                sface_fn = self._factory.make(sface_model_path(self._rec_model), providers)
                info = BackendInfo(
                    "yunet",
                    "sface",
                    "DmlExecutionProvider",
                    plan.adapter_name or f"dxgi:{device_id}",
                    int(device_id),
                    plan.dedicated_bytes,
                    False,
                )
                self._attach_dml(yunet_fn, sface_fn, info)
                return
            except Exception as exc:
                last_error = exc
        raise RuntimeError(str(last_error or "DirectML unavailable"))

    def detect(
        self,
        bgr: np.ndarray,
        extract_features: bool = True,
        *,
        max_side: int = 0,
        for_enroll: bool = False,
    ) -> list[FaceHit]:
        if bgr is None or bgr.size == 0:
            return []
        with self._lock:
            self._ensure_unlocked(for_enroll=for_enroll)
            det = self._enroll_det if for_enroll else self._live_det
            rec = self._enroll_rec if for_enroll else self._live_rec
            assert det is not None and rec is not None
            side = 0 if (for_enroll or det.uses_fixed_input) else max_side
            work, scale_x, scale_y = working_view(bgr, side)
            pad_enroll = (
                for_enroll
                and not det.uses_fixed_input
                and max(work.shape[0], work.shape[1]) <= CANVAS
            )
            if pad_enroll:
                canvas, pad_scale = fit_canvas(work)
                detections = det.detect(canvas, float(self.detect_score))
                if detections:
                    rows = unpad_rows(
                        np.stack([item.raw for item in detections]).astype(np.float32),
                        pad_scale,
                    )
                    detections = [
                        Detection(
                            box=(float(row[0]), float(row[1]), float(row[2]), float(row[3])),
                            score=float(row[14]),
                            raw=row,
                        )
                        for row in rows
                    ]
            else:
                detections = det.detect(work, float(self.detect_score))
            if not detections:
                faces = None
            else:
                faces = np.stack([item.raw for item in detections]).astype(np.float32)
            src = bgr
            engine = self

            def feature_fn(raw: np.ndarray) -> np.ndarray:
                engine.extract_count += 1
                return rec.embed(src, raw)

            return hits_from_detections(
                faces,
                scale_x=scale_x,
                scale_y=scale_y,
                extract_features=extract_features,
                feature_fn=feature_fn,
            )

    def annotate(
        self,
        bgr: np.ndarray,
        people: list[Person],
        threshold: float,
        *,
        max_side: int = 0,
        extract_features: bool = True,
    ) -> list[FaceHit]:
        hits = self.detect(bgr, extract_features=extract_features, max_side=max_side)
        if not people:
            return hits
        for hit in hits:
            if hit.feature is None:
                continue
            hit.match = best_match(hit.feature, people, threshold)
        return hits

    def enroll(self, bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        faces = self.enroll_all(bgr)
        return faces[0]

    def enroll_all(self, bgr: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        hits = self.detect(bgr, extract_features=True, for_enroll=True)
        if not hits:
            raise NoFaceError("未检测到人脸，请换一张正脸清晰、光线充足的照片")
        out: list[tuple[np.ndarray, np.ndarray]] = []
        for hit in sorted(hits, key=lambda item: item.w * item.h, reverse=True):
            if hit.feature is None:
                continue
            out.append((hit.feature, crop_face(bgr, hit)))
        if not out:
            raise NoFaceError("特征提取失败")
        return out
