# Agents are implemented in explorer, engineer, builder, coordinator.
# Import after implementation to avoid NotImplementedError in run().
__all__ = ["ExplorerAgent", "EngineerAgent", "BuilderAgent", "CoordinatorAgent"]

def __getattr__(name):
    if name == "ExplorerAgent":
        from .explorer import ExplorerAgent
        return ExplorerAgent
    if name == "EngineerAgent":
        from .engineer import EngineerAgent
        return EngineerAgent
    if name == "BuilderAgent":
        from .builder import BuilderAgent
        return BuilderAgent
    if name == "CoordinatorAgent":
        from .coordinator import CoordinatorAgent
        return CoordinatorAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
