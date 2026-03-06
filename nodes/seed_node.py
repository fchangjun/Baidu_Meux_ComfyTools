MAX_SAFE_SEED = 0xFFFFFFFFFFFFFFFF


class MeuxSeed:
    """
    Upstream logic aligned to WAS Node Suite's "Seed".
    """

    RETURN_TYPES = ("SEED", "NUMBER", "FLOAT", "INT")
    RETURN_NAMES = ("seed", "number", "float", "int")
    FUNCTION = "seed"
    CATEGORY = "utils/Meux"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SAFE_SEED, "step": 1}),
            }
        }

    def seed(self, seed):
        safe_seed = max(0, min(int(seed), MAX_SAFE_SEED))
        return (
            {"seed": safe_seed},
            safe_seed,
            float(safe_seed),
            int(safe_seed),
        )


__all__ = ["MeuxSeed"]
