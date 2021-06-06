# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test version string generation."""

import numpy as np
from qiskit.providers import BaseBackend
from qiskit.providers.models import QasmBackendConfiguration, PulseBackendConfiguration
from qiskit.result import Result
from qiskit_experiments import ExperimentData
from qiskit.test import QiskitTestCase


from qiskit_experiments.measurement.discriminator import (
    ESPDiscriminatorExperiment,
    ESPDiscriminatorAnalysis,
)


class ESPDiscriminatorBackend(BaseBackend):
    """
    A simple backend that generates gaussian data for discriminator tests
    """

    def __init__(self):
        """
        Initialize the discriminator backend
        """
        configuration = PulseBackendConfiguration(
            backend_name="esp_discriminator_simulator",
            backend_version="0",
            n_qubits=int(1),
            basis_gates=["x", "id", "x_01_gate", "x_12_gate", "measure"],
            gates=[],
            local=True,
            simulator=True,
            conditional=False,
            open_pulse=True,
            memory=True,
            max_shots=8192,
            n_uchannels=0,
            u_channel_lo=[],
            meas_levels=[1, 2],
            qubit_lo_range=[[4.471852852405576, 5.471852852405577]],
            meas_lo_range=[[6.493370669000002, 7.493370669000002]],
            meas_kernels=['hw_boxcar'],
            discriminators=['quadratic_discriminator', 'linear_discriminator', 'esp_discriminator'],
            coupling_map=None,
            rep_times=[1000.0],
            dt=0.2222222222222222,
            dtm=0.2222222222222222,     
        )

        super().__init__(configuration)

    def sample_gaussian(
        self, centroid=np.array([0, 0]), cov=np.array([[0.1, 0], [0, 0.1]]), size=1
    ):
        """
        Draws random samples from a gaussian distribution.
        """
        return np.random.multivariate_normal(centroid, cov, size)

    # pylint: disable = arguments-differ
    def run(self, qobj):
        """
        Run the discriminator backend
        """

        shots = qobj.config.shots

        result = {
            "backend_name": "ESP discriminator backend",
            "backend_version": "0",
            "qobj_id": 0,
            "job_id": 0,
            "success": True,
            "results": [],
        }

        for circ in qobj.experiments:
            nqubits = circ.config.n_qubits
            centroids = np.zeros([nqubits, 2])
            counts = dict()
            memory = np.zeros([shots, circ.config.memory_slots, 2])

            for i in range(shots):
                clbits = np.zeros(circ.config.memory_slots, dtype=int)
                meas_res = 0
                #print(circ.instructions)
                for op in circ.instructions:
                    qubit = op.qubits[0]
                    if op.name == "x_01_gate":
                        meas_res = 1
                    elif op.name == 'x_12_gate':
                        meas_res = 2
                    elif op.name == "measure":
                        # centroid is either (0,0) for |0>, (1,1) for |1>, or (2,2) for |2>
                        memory[i, op.memory[0]] = self.sample_gaussian(
                            centroid=np.array([meas_res, meas_res])
                        )
                        clbits[op.memory[0]] = meas_res
                        
                clstr = ""
                for clbit in clbits[::-1]:
                    clstr = clstr + str(clbit)

                if clstr in counts:
                    counts[clstr] += 1
                else:
                    counts[clstr] = 1

            result["results"].append(
                {
                    "shots": shots,
                    "success": True,
                    "header": {"metadata": circ.header.metadata},
                    "data": {"counts": counts, "memory": memory},
                }
            )

        return Result.from_dict(result)

class TestDiscriminator(QiskitTestCase):
    def test_single_qubit(self):
        backend = ESPDiscriminatorBackend()
        exp = ESPDiscriminatorExperiment(1)
        res = exp.run(backend, shots=10, meas_level=1, meas_return="single").analysis_result(0)
