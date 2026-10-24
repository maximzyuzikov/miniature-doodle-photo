"""ComfyUI custom-node entry point (not used when running stand-alone tests)."""

# Pytest may import the repository root as a plain __init__ module. ComfyUI
# loads it as an actual package, which is the only case requiring torch.
if __package__:
    from .comfyui_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
else:
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
