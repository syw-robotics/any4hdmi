"""Motion datasets and viewers; load training dependencies only when needed."""

from importlib import import_module


__all__ = [
    "BaseDataset",
    "FullMotionDataset",
    "MotionDatasetRouter",
    "MotionData",
    "MotionSample",
    "WindowedMotionDataset",
    "load_any4hdmi_dataset",
]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module("any4hdmi.dataset"), name)
    globals()[name] = value
    return value
