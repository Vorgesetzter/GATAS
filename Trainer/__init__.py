"""
Trainer module - Class-based training infrastructure for adversarial TTS.

Classes:
    - EnvironmentLoader: Handles environment initialization and model loading
    - AdversarialTrainer: Main optimization loop (returns OptimizationResult)
    - OptimizationResult: Dataclass containing optimization results
    - RunLogger: Handles logging and result persistence (called separately)
    - GraphPlotter: Generates visualization graphs
"""

from Trainer.ModelLoader import EnvironmentLoader, initialize_environment, load_optimizer
from Trainer.AdversarialTrainer import AdversarialTrainer, OptimizationResult, run_optimization_generation
from Trainer.RunLogger import RunLogger
from Trainer.GraphPlotter import GraphPlotter

__all__ = [
    # Classes
    "EnvironmentLoader",
    "AdversarialTrainer",
    "OptimizationResult",
    "RunLogger",
    "GraphPlotter",
    # Backward-compatible functions
    "initialize_environment",
    "load_optimizer",
    "run_optimization_generation",
]
