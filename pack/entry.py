from __future__ import annotations

import traceback


def _main() -> int:
    from facehide.threads import apply_env

    apply_env()
    from facehide.ui.app import main

    return main()


if __name__ == "__main__":
    try:
        raise SystemExit(_main())
    except SystemExit:
        raise
    except Exception:
        text = traceback.format_exc()[-1500:]
        try:
            import ctypes

            ctypes.windll.user32.MessageBoxW(0, text, "当面隐藏", 0x10)
        except Exception:
            pass
        raise
