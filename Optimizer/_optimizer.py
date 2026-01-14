from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Iterable, Optional, Type

import numpy as np
from numpy.typing import NDArray
from torch import Tensor

from Optimizer._optimizer_candidate import OptimizerCandidate

class Optimizer(ABC):
    """An abstract optimizer class."""

    # Standard elements (if applicable).
    _best_candidates: list[OptimizerCandidate]  # The current best candidate.
    _previous_best: list[OptimizerCandidate]  # The previous best solution candidate.
    _x_current: NDArray  # The current solution.
    _fitness: tuple[NDArray, ...]  # The current fitness.
    _bounds: tuple[int, int]  # Solution bounds.
    _n_var: int  # Solution size.

    # Always exist.
    _optimizer_type: Type[Any]
    _num_objectives: int

    def __init__(self, num_objectives: int) -> None:
        """
        Init default values.

        :param num_objectives: The number of objectives.
        """
        self._optimizer_type = type(self)
        self._num_objectives = num_objectives

    @abstractmethod
    def update(self) -> None:
        """
        Generate a new population.
        """
        ...

    @abstractmethod
    def get_x_current(self) -> NDArray:
        """
        Return the current population in specific format.

        :return: The population as array.
        """
        ...

    def assign_fitness(self, fitness: Iterable[NDArray], *data: Optional[Iterable[Any]]) -> None:
        """
        Assign fitness to the current population and extract the best individual using pareto frontier.

        :param fitness: The fitness to assign.
        :param data: Additional data for the candidates.
        :raises ValueError: If metrics array is not a 2D vector for some reason.
        """
        logging.info(f"Assigning fitness to {self.__class__.__name__}")
        # Format fitness into tuple if it is list or singular item.
        fitness = [f.numpy() if isinstance(f, Tensor) else f for f in fitness]
        fitness = tuple(fitness)
        assert (
            len(fitness) == self._num_objectives
        ), f"Error: {len(fitness)} Fitness values found, {self._num_objectives} expected."

        self._fitness = fitness

        """Since we can have an arbitrary amount of metrics we extract best candidates using a pareto frontier."""
        new_metrics = np.ascontiguousarray(
            fitness
        ).T  # Metrics are rows, instances are columns -> we transpose.
        old_metrics = np.ascontiguousarray([cand.fitness for cand in self._best_candidates])
        metrics = np.vstack((new_metrics, old_metrics))  # (n_new + n_old, n_obj)

        new_data: list[Any] = [None] * new_metrics.shape[0] if not data else list(zip(*data))
        data = tuple(new_data + [cand.data for cand in self._best_candidates])

        solutions = np.vstack(
            (self._x_current, np.array([cand.solution for cand in self._best_candidates]))
        )

        M = np.asarray(metrics)
        if M.ndim != 2:
            raise ValueError("metrics must be 2D: (n_points, n_objectives)")

        le = M[:, None, :] <= M[None, :, :]  # (n, n, k)
        lt = M[:, None, :] < M[None, :, :]  # (n, n, k)
        dominates = le.all(axis=2) & lt.any(axis=2)  # (n, n)
        dominated = dominates.any(axis=0)  # (n,)

        kept_indices = np.flatnonzero(dominated)

        candidates = [
            OptimizerCandidate(
                solution=solutions[i],
                fitness=metrics[i],
                data=data[i],
            )
            for i in kept_indices
        ]

        self._previous_best = self._best_candidates
        self._best_candidates = candidates

    def reset(self) -> None:
        """Reset the learner to the default."""
        self._x_current = np.random.uniform(
            low=self._bounds[0], high=self._bounds[1], size=self._x_current.shape
        )
        self._best_candidates = [
            OptimizerCandidate(self._x_current[0], [np.inf] * self._num_objectives)
        ]
        self._previous_best = self._best_candidates.copy()

    @property
    def best_candidates(self) -> list[OptimizerCandidate]:
        """
        Get the best candidates so far (if more than one it is a pareto frontier).

        :return: The candidate.
        """
        return self._best_candidates

    @property
    def previous_best(self) -> list[OptimizerCandidate]:
        """
        Get the previous best candidates.

        :return: The candidate.
        """
        return self._previous_best

    @property
    def optimizer_type(self) -> Type[Any]:
        """
        Get the type of the optimizer.

        :returns: The type.
        """
        return self._optimizer_type

    @property
    def n_var(self) -> int:
        """
        Get size of genome for optimizer.

        :returns: The size of the genome.
        """
        return self._n_var

    def _clip_to_bounds(self, element: NDArray) -> NDArray:
        """
        Clip array values to the range of bounds.

        :param element: The array to be clipped.
        :returns: The clipped array.
        """
        element = np.maximum(element, self._bounds[0])
        element = np.minimum(element, self._bounds[1])
        return element