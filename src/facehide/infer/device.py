from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

SKIP_SUBSTR = (
    "gameviewer",
    "basic render",
    "remote",
    "virtual",
    "microsoft basic display",
)

S_OK = 0
S_FALSE = 1
RPC_E_CHANGED_MODE = 0x80010106
COINIT_MULTITHREADED = 0x0


@dataclass(frozen=True)
class DxgiAdapter:
    name: str
    dxgi_index: int
    dedicated_bytes: int


@dataclass(frozen=True)
class DevicePlan:
    use_dml: bool
    dxgi_index: int | None
    adapter_name: str
    dedicated_bytes: int | None
    probe_ids: tuple[int, ...]
    reason: str


def is_virtual_adapter(name: str) -> bool:
    low = (name or "").lower()
    return any(token in low for token in SKIP_SUBSTR)


def pick_hardware_adapter(adapters: list[DxgiAdapter]) -> DxgiAdapter | None:
    hardware = [item for item in adapters if not is_virtual_adapter(item.name)]
    if not hardware:
        return None
    return max(hardware, key=lambda item: item.dedicated_bytes)


def plan_device(
    device: str,
    *,
    adapters: list[DxgiAdapter] | None = None,
    enum_failed: bool = False,
    dml_available: bool = True,
    ort_available: bool = True,
) -> DevicePlan:
    choice = (device or "auto").strip().lower()
    if choice == "cpu":
        return DevicePlan(False, None, "CPU", None, (), "user_cpu")
    if not ort_available:
        return DevicePlan(False, None, "CPU", None, (), "ort_missing")
    if not dml_available:
        return DevicePlan(False, None, "CPU", None, (), "dml_missing")
    if enum_failed:
        return DevicePlan(True, None, "", None, (0, 1), "enum_failed_probe")
    if adapters is None:
        return DevicePlan(False, None, "CPU", None, (), "no_adapters")
    chosen = pick_hardware_adapter(adapters)
    if chosen is None:
        return DevicePlan(False, None, "CPU", None, (), "no_hardware")
    return DevicePlan(
        True,
        chosen.dxgi_index,
        chosen.name,
        chosen.dedicated_bytes,
        (),
        "hardware",
    )


def coinitialize_mta() -> bool:
    try:
        import ctypes

        hr = int(ctypes.windll.ole32.CoInitializeEx(None, COINIT_MULTITHREADED))
        hr_u = hr & 0xFFFFFFFF
        if hr in (S_OK, S_FALSE) or hr_u in (S_OK, S_FALSE):
            return True
        return False
    except Exception:
        return False


def couninitialize() -> None:
    try:
        import ctypes

        ctypes.windll.ole32.CoUninitialize()
    except Exception:
        return


def enum_dxgi_adapters() -> list[DxgiAdapter]:
    import ctypes
    from ctypes import POINTER, Structure, byref, c_ubyte, c_uint, c_ulong, c_ushort, c_void_p, c_wchar

    class GUID(Structure):
        _fields_ = [
            ("Data1", c_ulong),
            ("Data2", c_ushort),
            ("Data3", c_ushort),
            ("Data4", c_ubyte * 8),
        ]

    class DXGI_ADAPTER_DESC(Structure):
        _fields_ = [
            ("Description", c_wchar * 128),
            ("VendorId", c_uint),
            ("DeviceId", c_uint),
            ("SubSysId", c_uint),
            ("Revision", c_uint),
            ("DedicatedVideoMemory", ctypes.c_size_t),
            ("DedicatedSystemMemory", ctypes.c_size_t),
            ("SharedSystemMemory", ctypes.c_size_t),
            ("AdapterLuid", c_ubyte * 8),
        ]

    iid_factory = GUID(0x7B7166EC, 0x21C7, 0x44AE, (c_ubyte * 8)(0xB2, 0x1A, 0xC9, 0xAE, 0x32, 0x1A, 0xE3, 0x69))
    dxgi = ctypes.WinDLL("dxgi.dll")
    create = dxgi.CreateDXGIFactory
    create.argtypes = [POINTER(GUID), POINTER(c_void_p)]
    create.restype = ctypes.c_long
    factory = c_void_p()
    hr = int(create(byref(iid_factory), byref(factory)))
    if hr != 0 or not factory.value:
        raise OSError(f"CreateDXGIFactory failed: {hr:#x}")

    vtbl = ctypes.cast(factory, POINTER(c_void_p)).contents
    methods = ctypes.cast(vtbl, POINTER(c_void_p * 16)).contents
    enum_adapters = ctypes.WINFUNCTYPE(ctypes.c_long, c_void_p, c_uint, POINTER(c_void_p))(methods[7])
    release = ctypes.WINFUNCTYPE(c_ulong, c_void_p)(methods[2])

    found: list[DxgiAdapter] = []
    index = 0
    try:
        while True:
            adapter = c_void_p()
            ahr = int(enum_adapters(factory, index, byref(adapter)))
            if ahr != 0 or not adapter.value:
                break
            avtbl = ctypes.cast(adapter, POINTER(c_void_p)).contents
            amethods = ctypes.cast(avtbl, POINTER(c_void_p * 16)).contents
            get_desc = ctypes.WINFUNCTYPE(ctypes.c_long, c_void_p, POINTER(DXGI_ADAPTER_DESC))(amethods[8])
            arelease = ctypes.WINFUNCTYPE(c_ulong, c_void_p)(amethods[2])
            desc = DXGI_ADAPTER_DESC()
            try:
                if int(get_desc(adapter, byref(desc))) == 0:
                    found.append(
                        DxgiAdapter(
                            name=str(desc.Description),
                            dxgi_index=index,
                            dedicated_bytes=int(desc.DedicatedVideoMemory),
                        )
                    )
            finally:
                arelease(adapter)
            index += 1
    finally:
        release(factory)
    return found


def ort_available() -> bool:
    try:
        import onnxruntime  # noqa: F401

        return True
    except Exception:
        return False


def dml_available() -> bool:
    try:
        import onnxruntime as ort

        return "DmlExecutionProvider" in ort.get_available_providers()
    except Exception:
        return False


AdapterEnum = Callable[[], list[DxgiAdapter]]


def plan_live_device(
    device: str,
    *,
    enum_fn: AdapterEnum | None = None,
    ort_ok: bool | None = None,
    dml_ok: bool | None = None,
) -> DevicePlan:
    if ort_ok is None:
        ort_ok = ort_available()
    if dml_ok is None:
        dml_ok = dml_available() if ort_ok else False
    adapters: list[DxgiAdapter] | None = None
    enum_failed = False
    if (device or "auto").strip().lower() != "cpu" and ort_ok and dml_ok:
        try:
            adapters = (enum_fn or enum_dxgi_adapters)()
        except Exception:
            enum_failed = True
            adapters = None
    return plan_device(
        device,
        adapters=adapters,
        enum_failed=enum_failed,
        dml_available=bool(dml_ok),
        ort_available=bool(ort_ok),
    )
