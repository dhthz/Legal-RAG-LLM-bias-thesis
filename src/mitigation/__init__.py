from importlib import import_module
from typing import List, Sequence

from src.mitigation.base import MitigationArm, retrieve_with_arms

# Arm name -> "module:Class". Imported on demand, so loading the pipeline does not pull in spaCy, torch or GPT-2.
ARM_REGISTRY = {
    "blind_query": "src.mitigation.neutral_rewrite:BlindQuery",
    "pcf": "src.mitigation.pcf:PCF",
}


def build_arms(names: Sequence[str]) -> List[MitigationArm]:
    arms = []
    for name in names:
        if name not in ARM_REGISTRY:
            raise ValueError(f"Unknown arm '{name}'. Available: {', '.join(ARM_REGISTRY)}")
        module, cls = ARM_REGISTRY[name].split(":")
        arms.append(getattr(import_module(module), cls)())
    return arms


__all__ = ["ARM_REGISTRY", "MitigationArm", "build_arms", "retrieve_with_arms"]
