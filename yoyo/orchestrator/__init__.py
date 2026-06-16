"""Pipeline orchestrator — multi-threaded coordination."""


def __getattr__(name):
    if name == "AimBotPipeline":
        from .pipeline import AimBotPipeline
        return AimBotPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["AimBotPipeline"]
