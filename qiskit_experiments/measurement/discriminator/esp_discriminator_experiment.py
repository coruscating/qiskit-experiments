"""
ESP Standard Discriminator Experiment class.
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt
from qiskit import circuit
from qiskit.providers import backend
from qiskit.pulse.builder import num_qubits
from qiskit_experiments.base_experiment import BaseExperiment
from qiskit.exceptions import QiskitError
from qiskit.circuit import QuantumCircuit, Gate
from .esp_discriminator_analysis import ESPDiscriminatorAnalysis
from typing import List, Optional, Union, Iterable
from qiskit.pulse.instructions import Instruction
import qiskit.pulse as pulse
from qiskit.pulse.library import Gaussian, GaussianSquare
from qiskit.compiler import assemble, transpile
from qiskit.pulse import Play, ShiftPhase, Schedule, Waveform, \
    ControlChannel, DriveChannel, AcquireChannel, MeasureChannel, MemorySlot
from qiskit.ignis.characterization.calibrations import rabi_schedules, RabiFitter
from scipy.optimize import curve_fit


ScheduleComponent = Union['Schedule', Instruction]
"""An element that composes a pulse schedule."""

class ESPDiscriminatorExperiment(BaseExperiment):
    """ESP Discriminator Experiment class"""

    # Analysis class for experiment
    __analysis_class__ = ESPDiscriminatorAnalysis

    def __init__(
        self,
        qubits: Union[int, Iterable[int]],
    ):
        """ESP discriminator experiment

        Args:
            qubits: the number of qubits or list of
                    physical qubits for the experiment.
        """

        super().__init__(qubits)
        self.backend = None
        self.backend_model = None
        self.dt = 0
        self.frequency_estimates = [0]*qubits
        self.anharmonicity_estimates = [-340*1.0e6]*qubits
        self.inst_map = None

    def apply_sideband(self, x_01_pulse: Waveform, mod_freq: float, carrier_freq: float, duration: Union[int, float]) -> Waveform:
        """Returns sideband pulse for generating |1> -> |2> pulse
        Args: 
            x_01_pulse (Waveform): x_01_pulse's waveform required for generating sideband pulse
            mod_freq (float): Frequency of modfied pulse from experiment
            carrier_freq (float): Frequency of carrier pulse
            duration Union[int, float]: Duration of sideband pulse
        Return:
            sideband_pulse (Waveform): Sideband pulse for |1> ->  |2> pulse
        """
        t_samples = np.linspace(0, self.dt*duration, duration)
        sine_pulse = np.sin(2*np.pi*(mod_freq-carrier_freq)*t_samples)
        sideband_pulse = Waveform(np.multiply(np.real(x_01_pulse.samples), sine_pulse), 
                                    name="sideband")
        return sideband_pulse
        
    def get_sched_12(self, x_sched_01: List[ScheduleComponent], measure_sched: List[ScheduleComponent]) -> List[ScheduleComponent]:
        """Returns a x_sched_12 pulse schedule that sends qubit from |1> -> |2>.
        Args:
            x_sched_01 (ScheduleComponent): Schedule consisting of pulse for sending qubit from |0> -> |1>
            measure_sched (ScheduleComponent): Schedule to perform measuurement of the given qubit 
            qubit (int): qubit index for which pulse is being generated
        Returns:
            x_sched_12 (ScheduleComponent): Schedule consisting of pulse for sending qubit from |1> -> |2>
        """

        if self.backend is None:
            raise QiskitError("""Backend is not initialized. Initialize it by 
                                    calling circuits() method first.""")

        MHz = 1.0e6 # Megahertz
        GHz = 1.0e9 # Gigahertz 
        qubit_lo_freq = self.frequency_estimates # Qubits frequencies
        anharm_freq = self.anharmonicity_estimates #340 MHz Anharmonicity of the qubit

        # Frequency Sweep
        excited_sweep_freqs = np.array([qubit_lo_freq[qubit] + anharm_freq[qubit] + np.linspace(-40*MHz, 40*MHz, 75) 
                                                for qubit in range(self.num_qubits)])
        x_pulse_01 = [x_sched_01[qubit].instructions[0][1].pulse for qubit in range(self.num_qubits)]

        sched_12 = []
        schedule_lo_freq = []
        
        for freqs in excited_sweep_freqs.T:

            schedule_lo_freq_dict = {}
            sched = pulse.Schedule(name=f"0->2 freq spec, f={freqs/GHz} GHz")

            for qubit, freq in enumerate(freqs):
                schedule_lo_freq_dict.update({pulse.DriveChannel(qubit): qubit_lo_freq[qubit]})
                freq_sweep_12_pulse = self.apply_sideband(x_pulse_01[qubit].get_waveform(), freq, 
                                                    qubit_lo_freq[qubit], x_pulse_01[qubit].duration)
                temp_sched = pulse.Play(x_pulse_01[qubit], pulse.DriveChannel(qubit))
                temp_sched |= pulse.Play(freq_sweep_12_pulse, pulse.DriveChannel(qubit)) << temp_sched.duration

                sched |= temp_sched
            sched |= measure_sched << sched.duration
            sched_12.append(sched)
            schedule_lo_freq.append(schedule_lo_freq_dict)

        excited_freq_sweep_program = assemble(sched_12,
                                      backend=self.backend, 
                                      meas_level=1,
                                      meas_return='avg',
                                      qubit_lo_freq=qubit_lo_freq,
                                      shots=1024,
                                      schedule_los=schedule_lo_freq)

        excited_freq_sweep_result = self.backend.run(excited_freq_sweep_program, 
                                                    system_model=self.backend_model).result()

        plt.figure(figsize=(15, 4*(self.num_qubits+1)//2))
        x_pulse_12_frequency = np.zeros(self.num_qubits)
        lorentzian = lambda x, A, q_freq, B, C: (A / np.pi) * (B / ((x - q_freq)**2 + B**2)) + C

        for qubit in range(self.num_qubits):

            ax = plt.subplot((self.num_qubits+1)//2, 2, qubit + 1)

            excited_freq_sweep_data = np.array([excited_freq_sweep_result.get_memory(i)[qubit] for i \
                                    in range(len(excited_freq_sweep_result.results))])

            # Check if the signal data is upwards or downwards
            datums = [0.5*(np.max(excited_freq_sweep_data) + np.min(excited_freq_sweep_data))]*2
            up_down = [0]*2
            while True:
                up_down_check = [np.real(excited_freq_sweep_data - datum) for datum in datums]
                if np.sign(up_down_check[0][0]) == np.sign(up_down_check[0][-1]):
                    up_down = [up_down_check[0][0], up_down_check[0][-1]]
                    break
                elif np.sign(up_down_check[1][0]) == np.sign(up_down_check[1][-1]):
                    up_down = [up_down_check[1][0], up_down_check[1][-1]]
                    break
                else:
                    datums[0] = 0.5*(np.max(excited_freq_sweep_data) + datums[0])
                    datums[1] = 0.5*(np.min(excited_freq_sweep_data) + datums[1])
            
            if up_down[0] > 0 and up_down[-1] > 0:
                # half-width at half-maximum
                p3 = excited_sweep_freqs[qubit][np.argmin(excited_freq_sweep_data)] - \
                     excited_sweep_freqs[qubit][np.where(excited_freq_sweep_data < \
                            (np.real(np.max(excited_freq_sweep_data) + np.min(excited_freq_sweep_data)))/2)[0][0]]
                p4 = np.max(excited_freq_sweep_data) #Constant
                p2 = qubit_lo_freq[qubit] + anharm_freq[qubit] #centre of peak
                p1 = p3*np.pi*(excited_freq_sweep_data[np.where(excited_sweep_freqs[qubit] == p2)[0][0]] - p4) #Amplitude Intenisty
            else:
                # half-width at half-maximum
                p3 = excited_sweep_freqs[qubit][np.argmax(excited_freq_sweep_data)] - \
                    excited_sweep_freqs[qubit][np.where(excited_freq_sweep_data > \
                            (np.real(np.max(excited_freq_sweep_data) - np.min(excited_freq_sweep_data)))/2)[0][0]]   
                p4 = np.min(excited_freq_sweep_data) #Constant
                p2 = qubit_lo_freq[qubit] + anharm_freq[qubit] #centre of peak
                p1 = p3*np.pi*(excited_freq_sweep_data[np.where(excited_sweep_freqs[qubit] == p2)[0][0]] - p4) #Amplitude Intenisty

            fit_sweep_params, _ = curve_fit(lorentzian ,
                                np.real(excited_sweep_freqs[qubit]),
                                np.real(excited_freq_sweep_data),
                                np.real([p1, p2, p3, p4]))

            x_pulse_12_frequency[qubit] = fit_sweep_params[1]

            ax.scatter(excited_sweep_freqs[qubit], np.real(excited_freq_sweep_data))
            ax.plot(excited_sweep_freqs[qubit], np.real(lorentzian(excited_sweep_freqs[qubit], *fit_sweep_params)))
            ax.set_xlabel("Frequency [Hz]", fontsize=14)
            ax.set_ylabel("Measured signal [a.u.]", fontsize=14)
            ax.set_title(f'Qubit {qubit}: Frequency Sweep (1->2)', fontsize=16)
        
        plt.show()

        # Rabi
        excited_rabi_amps = np.array([np.linspace(-1, 1, 75) for qubit in range(self.num_qubits)])

        rabi_12 = []
        
        for amps in excited_rabi_amps.T:
            
            sched = pulse.Schedule(name=f"0->2 rabi amp, A={amps}")
            for qubit, amp in enumerate(amps):
                base_12_pulse = Gaussian(duration=x_pulse_01[qubit].duration, sigma=x_pulse_01[qubit].sigma,
                                        amp=amp, name='base_12_pulse')
                rabi_12_pulse = self.apply_sideband(base_12_pulse.get_waveform(), x_pulse_12_frequency[qubit],
                                                    qubit_lo_freq[qubit], x_pulse_01[qubit].duration)
                temp_sched = pulse.Play(x_pulse_01[qubit], pulse.DriveChannel(qubit))
                temp_sched |= pulse.Play(rabi_12_pulse, pulse.DriveChannel(qubit)) << temp_sched.duration
                
                sched |= temp_sched
            sched |= measure_sched << sched.duration
            rabi_12.append(sched)

        excited_rabi_program = assemble(rabi_12,
                                      backend=self.backend, 
                                      meas_level=1,
                                      meas_return='avg',
                                      qubit_lo_freq=qubit_lo_freq,
                                      shots=1024,
                                      schedule_los=schedule_lo_freq) 

        excited_rabi_result = self.backend.run(excited_rabi_program, 
                                    system_model=self.backend_model).result()

        plt.figure(figsize=(15, 4*(self.num_qubits+1)//2))
        x_pulse_12_amp = np.zeros(self.num_qubits)
        cosine_wave = lambda x, A, B, drive_period, phi: (A*np.cos(2*np.pi*x/drive_period - phi) + B)

        for qubit in range(self.num_qubits):

            ax = plt.subplot((self.num_qubits+1)//2, 2, qubit + 1)
            
            excited_rabi_data = np.array([excited_rabi_result.get_memory(i)[qubit] for i \
                                        in range(len(excited_rabi_result.results))])

            # Guess parameter calculation for cosine_wave function
            p1 = (np.max(excited_rabi_data) - np.min(excited_rabi_data))/2 # Wave Amplitude
            p2 = (np.max(excited_rabi_data) + np.min(excited_rabi_data))/2 # Vertical Shift
            p4 = -np.arccos((excited_rabi_data[np.where(excited_rabi_amps[qubit] == 0)[0][0]] - p2)/p1) # Phase Shift
            zero_crossings = np.where(np.diff(np.signbit(np.real(excited_rabi_data-p2))))[0]
            if len(zero_crossings) >= 2:
                p3 = 2 * (excited_rabi_amps[qubit][zero_crossings[1]] - excited_rabi_amps[qubit][zero_crossings[0]]) #Drive period
            else:
                p3 = 2*np.pi/(np.arccos((excited_rabi_data[-1] - p2)/p1) + p4) #Drive period

            fit_rabi_params, _ = curve_fit(cosine_wave,
                                np.real(excited_rabi_amps[qubit]),
                                np.real(excited_rabi_data),
                                np.real([p1, p2, p3, p4]))

            x_pulse_12_amp[qubit] = fit_rabi_params[2]/2 * (1 + fit_rabi_params[3]/np.pi) # Compensating phi in amplitude

            ax.scatter(excited_rabi_amps[qubit], np.real(excited_rabi_data))
            ax.plot(excited_rabi_amps[qubit], np.real(cosine_wave(excited_rabi_amps[qubit], *fit_rabi_params)))
            ax.set_xlabel("Drive amp [a.u.]", fontsize=14)
            ax.set_ylabel("Measured signal [a.u.]", fontsize=14)
            ax.set_title(f'Qubit {qubit}: Rabi Experiment (1->2)', fontsize=16)

        plt.show()

        x_sched_12 = []
        for qubit in range(self.num_qubits):
            sched = pulse.Schedule(name=f"0->2 pulse schedule")
            x_12_pulse = Gaussian(duration = x_pulse_01[qubit].duration, amp = np.min([x_pulse_12_amp[qubit], 1.0]),
                                    sigma = x_pulse_01[qubit].sigma, name = 'x_12_pulse')
            x_12_pulse = self.apply_sideband(x_12_pulse.get_waveform(), qubit_lo_freq[qubit], 
                                         x_pulse_12_frequency[qubit], x_pulse_01[qubit].duration)
            sched |= pulse.Play(x_12_pulse, pulse.DriveChannel(qubit))
            x_sched_12.append(sched)
        
        return x_sched_12

    def circuit_generator(self, x_sched_01: ScheduleComponent, x_sched_12: ScheduleComponent):
        
        circuits = []
        for label in (0, 1, 2):
            circ = QuantumCircuit(self.num_qubits)
            if label == 1:
                x_01_gate = Gate('x_01_gate', 1, [])
                for qubit in range(self.num_qubits):
                    circ.append(x_01_gate, [qubit])
                    circ.add_calibration(x_01_gate, [qubit], x_sched_01[qubit])
            elif label == 2:
                x_01_gate = Gate('x_01_gate', 1, [])
                x_12_gate = Gate('x_12_gate', 1, [])
                for qubit in range(self.num_qubits):
                    circ.append(x_01_gate, [qubit])
                    circ.append(x_12_gate, [qubit])
                    circ.add_calibration(x_01_gate, [qubit], x_sched_01[qubit])
                    circ.add_calibration(x_12_gate, [qubit], x_sched_12[qubit])
            
            circ.measure_all()

            circ.metadata = {
                "experiment_type": self._type,
                "ylabel": self.num_qubits * str(label),
                "qubits": self.physical_qubits,
            }
            circuits.append(circ)
        
        return circuits

    def circuits(self, backend: Optional["Backend"] = None) -> List[QuantumCircuit]:
        """Return a list of esp discriminator circuits.
        Args:
            backend (Backend): Optional, a backend object.
        Returns:
            List[QuantumCircuit]: A list of :class:`QuantumCircuit`s.
        """

        if self.backend is None and self.backend_model is None and self.inst_map is None:
            if backend is None or getattr(backend, "defaults", None) is None:
                from qiskit.providers.aer import PulseSimulator
                from qiskit.providers.aer.pulse import PulseSystemModel
                from qiskit.test.mock.backends.armonk.fake_armonk import FakeArmonk
                
                warnings.warn("Default PulseSimulator backend being built from FakeArmonk's system model.")
                armonk_backend = FakeArmonk()
                self.backend = PulseSimulator()
                self.backend_model = PulseSystemModel.from_backend(armonk_backend)
                self.dt = armonk_backend.configuration().dt
                self.backend_defaults = armonk_backend.defaults()
                self.backend_configurations = armonk_backend.configuration()
                self.anharmonicity_estimates = [armonk_backend.properties().qubit_property(qb)['anharmonicity'][0] 
                                                            for qb in range(self.num_qubits)]

            else:
                self.backend = backend
                self.backend_model = None
                self.dt = backend.configuration().dt
                self.backend_defaults = backend.defaults(refresh=True)
                self.backend_configurations = backend.configuration()
                self.anharmonicity_estimates = [backend.properties().qubit_property(qb)['anharmonicity'][0] 
                                                            for qb in range(self.num_qubits)]
            
            self.frequency_estimates = self.backend_defaults.qubit_freq_est
            self.inst_map = self.backend_defaults.instruction_schedule_map

        x_sched_01 = []
        x_sched_12 = []
        
        for qubit in range(self.num_qubits):
            x_sched_01.append(self.inst_map.get('x', [qubit]))
            
        measure_sched = self.inst_map.get('measure', range(self.num_qubits))
        x_sched_12 = self.get_sched_12(x_sched_01, measure_sched)

        return self.circuit_generator(x_sched_01, x_sched_12)
