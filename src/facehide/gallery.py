from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from facehide.paths import gallery_dir, gallery_index_path

ASK_FLOOR = 0.28


def new_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass
class Sample:
    id: str
    feature: np.ndarray
    thumb_path: Path
    source: str = "enroll"


@dataclass
class Person:
    id: str
    name: str
    samples: list[Sample] = field(default_factory=list)
    enabled: bool = True
    blacklisted: bool = False
    nickname: str = ""

    @property
    def auto_linked(self) -> bool:
        return any(sample.source == "auto" for sample in self.samples)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    left = np.asarray(a, dtype=np.float32).reshape(-1)
    right = np.asarray(b, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 1e-8:
        return -1.0
    return float(np.dot(left, right) / denom)


@dataclass(frozen=True)
class MatchResult:
    person: Person
    score: float
    sample_id: str


def best_match(
    feature: np.ndarray,
    people: list[Person],
    threshold: float,
) -> MatchResult | None:
    ranked = rank_people(feature, people)
    if not ranked:
        return None
    top = ranked[0]
    if top.score < threshold:
        return None
    return top


def rank_people(feature: np.ndarray, people: list[Person]) -> list[MatchResult]:
    ranked: list[MatchResult] = []
    for person in people:
        best: MatchResult | None = None
        for sample in person.samples:
            score = cosine_similarity(feature, sample.feature)
            if best is None or score > best.score:
                best = MatchResult(person=person, score=score, sample_id=sample.id)
        if best is not None:
            ranked.append(best)
    ranked.sort(key=lambda item: item.score, reverse=True)
    return ranked


def cluster_indices(features: list[np.ndarray], threshold: float) -> list[list[int]]:
    count = len(features)
    if count == 0:
        return []
    parent = list(range(count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_l, root_r = find(left), find(right)
        if root_l != root_r:
            parent[root_r] = root_l

    for i, left in enumerate(features):
        for j in range(i + 1, count):
            if cosine_similarity(left, features[j]) >= threshold:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for index in range(count):
        groups.setdefault(find(index), []).append(index)
    return list(groups.values())


def can_trigger(match: MatchResult | None, threshold: float) -> bool:
    return bool(match is not None and match.score >= threshold and match.person.enabled)


def decide_link(score: float, threshold: float, auto_link: bool) -> str:
    if score >= threshold and auto_link:
        return "auto"
    if score >= threshold:
        return "ask"
    if score >= ASK_FLOOR:
        return "ask"
    return "new"


def write_thumb(bgr: np.ndarray, dest: Path, size: int = 160) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    h, w = bgr.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError("empty crop")
    scale = size / max(h, w)
    resized = cv2.resize(bgr, (max(1, int(w * scale)), max(1, int(h * scale))))
    cv2.imwrite(str(dest), resized)


class Gallery:
    def __init__(self, index_path: Path | None = None, root: Path | None = None) -> None:
        self._index_path = index_path or gallery_index_path()
        self._root = root or gallery_dir()
        self._root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._people: list[Person] = []
        self.reload()

    def people(self) -> list[Person]:
        with self._lock:
            return list(self._people)

    def person(self, person_id: str) -> Person | None:
        with self._lock:
            return self._find(person_id)

    def reload(self) -> None:
        with self._lock:
            self._people = self._load()

    def add_person(
        self,
        name: str,
        feature: np.ndarray,
        thumb_bgr: np.ndarray,
        source: str = "enroll",
        enabled: bool = True,
    ) -> Person:
        name = name.strip() or "未命名"
        person = Person(id=new_id(), name=name, enabled=enabled)
        with self._lock:
            self._people.append(person)
            self._add_sample_unlocked(person, feature, thumb_bgr, source)
            self._save()
            return person

    def add_sample(
        self,
        person_id: str,
        feature: np.ndarray,
        thumb_bgr: np.ndarray,
        source: str = "manual",
    ) -> Sample:
        with self._lock:
            person = self._find(person_id)
            if person is None:
                raise KeyError(person_id)
            sample = self._add_sample_unlocked(person, feature, thumb_bgr, source)
            self._save()
            return sample

    def rename(self, person_id: str, name: str) -> None:
        with self._lock:
            person = self._find(person_id)
            if person is None:
                raise KeyError(person_id)
            person.name = name.strip() or person.name
            self._save()

    def set_enabled(self, person_id: str, enabled: bool) -> None:
        with self._lock:
            person = self._find(person_id)
            if person is None:
                raise KeyError(person_id)
            person.enabled = bool(enabled)
            self._save()

    def set_blacklisted(self, person_id: str, blacklisted: bool) -> None:
        with self._lock:
            person = self._find(person_id)
            if person is None:
                raise KeyError(person_id)
            person.blacklisted = bool(blacklisted)
            self._save()

    def set_nickname(self, person_id: str, nickname: str) -> None:
        with self._lock:
            person = self._find(person_id)
            if person is None:
                raise KeyError(person_id)
            person.nickname = nickname.strip()
            self._save()

    def merge_people(self, keep_id: str, absorb_id: str) -> Person:
        if keep_id == absorb_id:
            person = self.person(keep_id)
            if person is None:
                raise KeyError(keep_id)
            return person
        with self._lock:
            keep = self._find(keep_id)
            absorb = self._find(absorb_id)
            if keep is None:
                raise KeyError(keep_id)
            if absorb is None:
                raise KeyError(absorb_id)
            keep.samples.extend(absorb.samples)
            keep.blacklisted = keep.blacklisted or absorb.blacklisted
            if not keep.nickname.strip() and absorb.nickname.strip():
                keep.nickname = absorb.nickname
            self._people = [item for item in self._people if item.id != absorb_id]
            self._save()
            return keep

    def split_sample(self, person_id: str, sample_id: str, name: str) -> Person:
        with self._lock:
            person = self._find(person_id)
            if person is None:
                raise KeyError(person_id)
            sample = next((item for item in person.samples if item.id == sample_id), None)
            if sample is None:
                raise KeyError(sample_id)
            if len(person.samples) <= 1:
                raise ValueError("只剩一张，不能再拆")
            person.samples = [item for item in person.samples if item.id != sample_id]
            created = Person(
                id=new_id(),
                name=name.strip() or "未命名",
                samples=[sample],
                enabled=person.enabled,
                blacklisted=person.blacklisted,
            )
            self._people.append(created)
            self._save()
            return created

    def remove_person(self, person_id: str) -> None:
        with self._lock:
            person = self._find(person_id)
            if person is None:
                return
            for sample in person.samples:
                self._delete_files(sample)
            self._people = [item for item in self._people if item.id != person_id]
            self._save()

    def remove_sample(self, person_id: str, sample_id: str) -> None:
        with self._lock:
            person = self._find(person_id)
            if person is None:
                return
            kept: list[Sample] = []
            for sample in person.samples:
                if sample.id == sample_id:
                    self._delete_files(sample)
                else:
                    kept.append(sample)
            person.samples = kept
            if not person.samples:
                self._people = [item for item in self._people if item.id != person_id]
            self._save()

    def _find(self, person_id: str) -> Person | None:
        for person in self._people:
            if person.id == person_id:
                return person
        return None

    def _add_sample_unlocked(
        self,
        person: Person,
        feature: np.ndarray,
        thumb_bgr: np.ndarray,
        source: str = "enroll",
    ) -> Sample:
        sample_id = new_id()
        feat_path = self._root / f"{sample_id}.npy"
        thumb_path = self._root / f"{sample_id}.jpg"
        np.save(feat_path, np.asarray(feature, dtype=np.float32).reshape(-1))
        write_thumb(thumb_bgr, thumb_path)
        sample = Sample(
            id=sample_id,
            feature=np.asarray(feature, dtype=np.float32).reshape(-1),
            thumb_path=thumb_path,
            source=source,
        )
        person.samples.append(sample)
        return sample

    def _delete_files(self, sample: Sample) -> None:
        for path in (self._root / f"{sample.id}.npy", sample.thumb_path):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass

    def _load(self) -> list[Person]:
        if not self._index_path.exists():
            return []
        try:
            raw = json.loads(self._index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        people: list[Person] = []
        for item in raw.get("people") or []:
            person = self._person_from(item)
            if person and person.samples:
                people.append(person)
        return people

    def _person_from(self, item: Any) -> Person | None:
        if not isinstance(item, dict):
            return None
        person = Person(
            id=str(item.get("id") or new_id()),
            name=str(item.get("name") or "未命名"),
            enabled=bool(item.get("enabled", True)),
            blacklisted=bool(item.get("blacklisted", False)),
            nickname=str(item.get("nickname") or "").strip(),
        )
        for sample_raw in item.get("samples") or []:
            if not isinstance(sample_raw, dict):
                continue
            sample_id = str(sample_raw.get("id") or "")
            feat_name = str(sample_raw.get("feature") or f"{sample_id}.npy")
            thumb_name = str(sample_raw.get("thumb") or f"{sample_id}.jpg")
            feat_path = self._root / feat_name
            thumb_path = self._root / thumb_name
            if not sample_id or not feat_path.exists():
                continue
            try:
                feature = np.load(feat_path).astype(np.float32).reshape(-1)
            except (OSError, ValueError):
                continue
            source = str(sample_raw.get("source") or "enroll")
            if source not in {"enroll", "auto", "manual"}:
                source = "enroll"
            person.samples.append(
                Sample(id=sample_id, feature=feature, thumb_path=thumb_path, source=source)
            )
        return person

    def _save(self) -> None:
        payload = {
            "people": [
                {
                    "id": person.id,
                    "name": person.name,
                    "enabled": person.enabled,
                    "blacklisted": person.blacklisted,
                    "nickname": person.nickname,
                    "samples": [
                        {
                            "id": sample.id,
                            "feature": f"{sample.id}.npy",
                            "thumb": sample.thumb_path.name,
                            "source": sample.source,
                        }
                        for sample in person.samples
                    ],
                }
                for person in self._people
            ]
        }
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        self._index_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
