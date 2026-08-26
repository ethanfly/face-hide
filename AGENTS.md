# AGENTS.md

FaceHide（当面隐藏）is a **Windows-only** desktop app: a webcam watcher that, when a registered face appears, hides entertainment/game windows and opens configured work apps (or shows the desktop). Stack: Python 3.10+ (validated up to 3.14), PySide6, OpenCV (YuNet detection + SFace recognition), pywin32, psutil. All user-facing text is Chinese-first with an English translation table.

## Commands

```bat
python -m pip install -e .          :: install deps + package (editable)
python -m facehide                  :: run (or run.bat)
python -m facehide --check          :: self-check: models, blank-image detection, camera enum (needs network)
python -m facehide --dev            :: dev mode: overlay scores, triggers are dry-run (see gotchas)
python -m facehide --minimized      :: start hidden in tray
python -m unittest discover -s tests -q   :: run tests (exact CI command)
python pack\build.py                :: local build: PyInstaller + Inno Setup (or pack.bat)
python pack\version.py show         :: print current version
python pack\version.py bump --kind patch|minor|major|none
```

There is no pytest, no linter/formatter config. Tests are stdlib `unittest` only.

## Architecture and data flow

src-layout package `facehide` under `src/`. Control flow:

1. `ui/app.py:main()` — parses `--check/--dev/--minimized`, enforces single instance (`instance.py`, QLocalServer `"FaceHide.SingleInstance"`), downloads models with a progress dialog, builds `SettingsStore` / `Gallery` / `FaceEngine`, then `MainWindow` + tray.
2. `monitor.py:MonitorThread` (QThread) — loop: reopen camera when `settings.camera_index` changes → `engine.annotate(frame, gallery.people(), threshold)` → streak of consecutive matched frames → when armed and `streak >= confirm_frames` and cooldown elapsed → `actions.perform_switch(...)` and emit `triggered`. Emits `PreviewFrame` every frame for the UI. `protected_pids={os.getpid()}` so FaceHide never minimizes itself.
3. `engine.py:FaceEngine` — lazy, lock-guarded wrapper around `cv2.FaceDetectorYN` (YuNet) + `cv2.FaceRecognizerSF` (SFace). Matching is **cosine similarity** on SFace features (`gallery.cosine_similarity`), default threshold 0.40.
4. `actions.py` — **pure planning + side-effectful execution split**: `plan_switch()` / `windows_to_minimize()` compute a `SwitchPlan` from `(Settings, list[WindowInfo], fg_hwnd, protected_hwnds, protected_pids)` with zero win32 calls; `perform_switch()` executes it. All win32 imports are deferred inside `_win32()` so pure planning is testable anywhere.
5. `notify.py` — channel senders (DingTalk group/app, Feishu, generic webhook) keyed in `_SENDERS`. All HTTP goes through an injectable `JsonFn` callable `http(method, url, body, headers) -> (status, payload)`; tests pass a stub, never real network.

Runtime data lives in `%LOCALAPPDATA%\FaceHide\` (`paths.py`): `config.json`, `gallery.json` + `gallery/` (`.npy` features + thumbnails), `models/` (ONNX). `models.py:ensure_models()` seeds from bundled models (frozen app) then downloads from HuggingFace with opencv_zoo fallback; partial downloads use `.part` suffixes.

### Module map

| Module | Role |
| --- | --- |
| `config.py` | `Settings` dataclass + tolerant `settings_from_dict` loader; `SettingsStore` returns **copies** (`get()` copies, `replace()` deep-copies and writes JSON). Entertainment process names are lowercased on load. |
| `gallery.py` | Thread-safe person/sample store; every mutation re-serializes `gallery.json`. `decide_link()` → `auto`/`ask`/`new` enrollment decision; `ASK_FLOOR = 0.28`. |
| `camera.py` | DShow-first camera open; `is_placeholder_frame()` detects virtual-camera logo feeds; `pick_camera()` skips placeholder cameras. |
| `instance.py` | Single-instance guard; a second launch signals the existing one to reveal. |
| `startup.py` | Start-on-boot via `HKCU\Software\Microsoft\Windows\CurrentVersion\Run` (`FaceHide` value). |
| `logbook.py` | Recognition log records + Excel export (`openpyxl`). |
| `mark.py` | App icon/tray mark rendered at runtime with PIL (no image assets). |
| `runtime.py` | `is_frozen()` / `bundle_dir()` / `exe_dir()` distinguish dev vs PyInstaller; `asset_path()` probes several locations. |
| `ui/main_window.py` | One ~2200-line file: main window + all dialogs (enroll, capture, pickers, merge, same-person). |
| `ui/styles.py` | Global QSS (`APP_QSS`); Fusion style. |
| `ui/icons.py` | Glyph icons drawn programmatically (PIL → QPixmap); only real image asset is `ui/check.png`. |
| `pack/` | Build tooling: `build.py` (models → icon → PyInstaller spec → zip → Inno Setup), `version.py`, `shots.py` (screenshots), `entry.py` (frozen entry with MessageBox crash reporting). |

## CI / release behavior (important)

- Pushing to `main` triggers `.github/workflows/build.yml` **unless the commit message contains `[skip ci]`**. The workflow runs tests, then **bumps the patch version itself**, commits `chore: release vX.Y.Z [skip ci]`, tags, and publishes to Releases. Do not hand-edit version files before pushing; do not add `[skip ci]` to feature commits unless a release is unwanted.
- The version lives in **three files kept in sync only by `pack/version.py`**: `VERSION` (source of truth), `pyproject.toml`, `src/facehide/__init__.py`. Always bump via the script.
- `pack/build.py` copies ONNX models into the PyInstaller bundle: it looks in `./models/` then `%LOCALAPPDATA%\FaceHide\models`, downloading into `./models/` if missing. Locally packaging without network requires models already present in one of those folders.
- Generated/gitignored: `dist/`, `build/`, `models/*.onnx`, `pack/FaceHide.ico`, `pack/facehide.spec`, `pack/_inno/` (auto-installed Inno Setup). Don't commit these.

## Conventions

- `from __future__ import annotations` everywhere; dataclasses for value types; double quotes; 4-space indent.
- Every user-facing string goes through `i18n.t("key", **kwargs)`. Strings live in the single `_STRINGS` dict in `i18n.py` with `zh` and `en` tables; `t()` falls back en→zh→key. **A new string must be added to both tables** — `test_i18n.py` asserts the key sets are identical.
- Error/log text that never reaches the UI may be Chinese literals; code comments are sparse and Chinese.
- Commit messages: short imperative English subject ("Add …", "Fix …").
- Settings changes go through `store.get()` → mutate copy → `store.replace()`; never hold a `Settings` reference across threads (monitor thread re-reads it each frame, which is how settings changes apply live).
- Window/exe matching is always case-insensitive: compare lowercased exe names (`work_exes`, `SKIP_OPEN_APP_EXES`).

## Testing approach

- Plain `unittest`, one file per module under `tests/`, run with discovery from the repo root. No pytest dependency — don't add one.
- Tests avoid OS/hardware: `test_actions.py` feeds fake `WindowInfo` lists into pure planners; `test_notify.py` injects a stub `http` function; `test_instance.py` uses unique QLocalServer names with cleanup.
- `test_engine.py` self-skips when ONNX models aren't in `%LOCALAPPDATA%\FaceHide\models` — the suite passes on a fresh machine without them.
- Icon/Qt tests create `QApplication.instance() or QApplication([])` in `setUpClass`.
- Anything touching `i18n` global state calls `set_language("zh")` in `setUp`/`tearDown` (language is a module global; leaks between tests otherwise).
- `test_version.py` inserts the repo root into `sys.path` to import `pack.version` (pack scripts are not installed packages).

## Gotchas

- `--dev` **persists**: it writes `dev_mode=true` into the real `config.json` (and `--minimized` is ignored while dev mode is on; closing the window quits instead of tray). A dev flag on one run affects later runs until unset in settings.
- Model files are matched by exact filenames (`face_detection_yunet_2023mar.onnx`, `face_recognition_sface_2021dec.onnx`) in three places: `models.py`, `pack/build.py` `MODEL_NAMES`. Renaming/upgrading a model means updating both.
- `Gallery` features are stored as `.npy` files next to `gallery.json`; deleting `gallery.json` without deleting `gallery/` (or vice versa) leaves orphans. The loader tolerates missing feature files by skipping the sample.
- Camera open prefers DShow (`CAP_DSHOW`) and falls back to the default backend; buffer size is forced to 1 to avoid stale frames.
- `plan_switch` with empty `work_apps` sets `show_desktop=True` (Win+M) instead of launching nothing — UI and tests both rely on this.
- `MainWindow` minimizes windows but protects: its own hwnds (`protected_hwnds`), FaceHide's pid, and any exe that is a configured work app.
- `pack/shots.py` regenerates `docs/screenshots/` with a real (fake-data) UI run; don't hand-edit those PNGs.
