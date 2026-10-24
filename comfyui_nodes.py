"""Optional ComfyUI node. Requires ComfyUI's torch runtime and pip-installed film_agent."""
import numpy as np
import torch

from film_agent.core import STYLES, process_array


class AdaptiveFilmLook:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "style": (list(STYLES), {"default": "hp5-120"}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "grain": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
                "auto": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2147483647}),
            },
            "optional": {"iso": ("INT", {"default": 0, "min": 0, "max": 204800})},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    CATEGORY = "Film Agent"
    FUNCTION = "render"

    def render(self, image, style, strength, grain, auto, seed, iso=0):
        outputs = []
        for frame in image:
            rgb = np.uint8(np.round(np.clip(frame.detach().cpu().numpy(), 0, 1) * 255))
            result, _ = process_array(rgb, style, strength=strength, grain=grain,
                                      seed=seed, iso=iso or None, auto=auto)
            outputs.append(torch.from_numpy(result.astype(np.float32) / 255.0))
        return (torch.stack(outputs, dim=0).to(image.device),)


NODE_CLASS_MAPPINGS = {"AdaptiveFilmLook": AdaptiveFilmLook}
NODE_DISPLAY_NAME_MAPPINGS = {"AdaptiveFilmLook": "Film Agent · Adaptive Film Look"}
