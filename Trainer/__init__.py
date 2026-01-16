"""
Trainer module - Class-based training infrastructure for adversarial TTS.

Classes:
    - EnvironmentLoader: Handles environment initialization and model loading
    - AdversarialTrainer: Main optimization loop (returns OptimizationResult)
    - OptimizationResult: Dataclass containing optimization results
    - RunLogger: Handles logging and result persistence (called separately)
    - GraphPlotter: Generates visualization graphs
    - VectorManipulator: Handles interpolation of style vectors for batched TTS inference
"""

from Trainer.EnvironmentLoader import EnvironmentLoader
from Trainer.AdversarialTrainer import AdversarialTrainer
from Trainer.RunLogger import RunLogger
from Trainer.GraphPlotter import GraphPlotter
from Trainer.VectorManipulator import VectorManipulator

__all__ = [
    "EnvironmentLoader",
    "AdversarialTrainer",
    "RunLogger",
    "GraphPlotter",
    "VectorManipulator",
]
