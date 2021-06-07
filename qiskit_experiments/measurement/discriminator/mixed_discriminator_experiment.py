"""
Discriminator Experiment class for mixed states.
"""


from .mixed_discriminator_analysis import MixedDiscriminatorAnalysis
from .discriminator_experiment import DiscriminatorExperiment
from typing import Optional, Union, Iterable
import copy
from qiskit import transpile, assemble
from qiskit.providers.backend import Backend
from qiskit.providers.basebackend import BaseBackend as LegacyBackend
from qiskit_experiments.experiment_data import ExperimentData



class MixedDiscriminatorExperiment(DiscriminatorExperiment):
    """Discriminator Experiment class"""

    # Analysis class for experiment
    __analysis_class__ = MixedDiscriminatorAnalysis

    def __init__(
        self,
        qubits: Union[int, Iterable[int]]
    ):
        """Standard discriminator experiment

        Args:
            qubits: the number of qubits or list of
                    physical qubits for the experiment.
        """
        super().__init__(qubits)
