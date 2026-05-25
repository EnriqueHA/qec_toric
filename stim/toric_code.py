# Toric code implementation for quantum error correction
import math
import numpy as np
import stim

class ToricCode:
    def __init__(self, d: int, rounds: int, noise_model: str = "depolarizing-equal_op", 
                 noise_prob = 0.001,
                 noise_single_prob: float = 0.001,
                 noise_two_prob: float = 0.001,
                 noise_spam_prob: float = 0.01):
        """
        Circuit-level noise is considered in this implementation.

        Depolarization (px = p/3, py = p/3, pz = p/3) and equal probabilities for the different operations are the default.

        Possible modifications to noise probabilities:
        - depolarizing-equal_op: single-qubit, two-qubit, and spam (preparation and measurement) errors 
                                all occur with the same probability p 
        - depolarizing-biased_op: Define different probabililities for single-qubit, two-qubit, and spam errors
        - custom-biased_op: Different error probabilities for X, Y, and Z errors and
                                 also different for single-qubit, two-qubit, and spam errors.
                                 two-qubit errors order: IX, IY, IZ, XI, XX, XY, XZ, YI, YX, YY, YZ, ZI, ZX, ZY, ZZ
        """
        if noise_model == "depolarizing-equal_op":
            self.p = noise_prob
            p_singleq = tuple([noise_prob / 3] * 3) # px, py, pz for single-qubit errors
            p_twoq = tuple([noise_prob / 15] * 15) # 15 possible two-qubit Pauli errors
            p_spam = tuple([noise_prob / 3] * 3) # px, py, pz for spam errors (preparation and measurement)
        
        elif noise_model == "depolarizing-biased_op":
            self.p = noise_prob
            p_singleq = tuple([noise_single_prob / 3] * 3) # px, py, pz for single-qubit errors
            p_twoq = tuple([noise_two_prob / 15] * 15) # 15 possible two-qubit Pauli errors
            p_spam = tuple([noise_spam_prob / 3] * 3) # px, py, pz for spam errors (preparation and measurement)

        elif noise_model == "custom-biased_op":
            # For custom-biased_op, the user can directly input the probabilities as parameters
            if len(noise_single_prob) != 3:
                raise ValueError("For custom-biased_op, noise_single_prob should be a tuple of 3 values for px, py, pz.")
            elif len(noise_two_prob) != 15:
                raise ValueError("For custom-biased_op, noise_two_prob should be a tuple of 15 values for the two-qubit Pauli errors." \
                                " The order of the 15 two-qubit Pauli errors should be: IX, IY, IZ, XI, XX, XY, XZ, YI, YX, YY, YZ, ZI, ZX, ZY, ZZ.")
            elif len(noise_spam_prob) != 3:
                raise ValueError("For custom-biased_op, noise_spam_prob should be a tuple of 3 values for px, py, pz.")
            else: 
                p_singleq = noise_single_prob
                p_twoq = noise_two_prob
                p_spam = noise_spam_prob
        else:
            raise ValueError("Invalid noise model. Choose from 'depolarizing-equal_op', 'depolarizing-biased_op', or 'custom-biased_op'.")

        self.d = d
        self.rounds = rounds # Number of rounds of error correction to perform
        self.noise_model = noise_model
        self.p_singleq = p_singleq # Tuple of (px, py, pz) for single-qubit errors
        self.p_twoq = p_twoq # IX, IY, IZ, XI, XX, XY, XZ, YI, YX, YY, YZ, ZI, ZX, ZY, ZZ for two-qubit errors
        self.p_spam = p_spam # Tuple of (px, py, pz) for state preparation and measurement errors
        self.circuit = stim.Circuit()
        self.data_coords, self.x_ancillas_coords, self.z_ancillas_coords = self._generate_coordinate_grid()
        self.coords_to_data = {coord: qubit for qubit, coord in self.data_coords.items()} # Map from coordinates to data qubit ids

        # List of qubits ids
        self.data_qubits = list(self.data_coords.keys()) 
        self.x_ancillas = list(self.x_ancillas_coords.keys()) 
        self.z_ancillas = list(self.z_ancillas_coords.keys()) 

        # Append qubits to the circuit
        for qubit_id, (x, y) in self.data_coords.items():
            self.circuit.append("QUBIT_COORDS", [qubit_id], [x, y])

        for qubit_id, (x, y) in self.x_ancillas_coords.items():
            self.circuit.append("QUBIT_COORDS", [qubit_id], [x, y])

        for qubit_id, (x, y) in self.z_ancillas_coords.items():
            self.circuit.append("QUBIT_COORDS", [qubit_id], [x, y])
        
        # First round of syndrome extraction and detector definition 
        self._initialization_data_qubits(self.circuit)
        self._extract_syndrome(self.circuit)

        if self.rounds == 1:
            # This is usually the case of spatial experiments, such that we don't have spam noise.
            # And we just care about the spatial syndrome.
            self._extract_syndrome(self.circuit) # One extra round to have X and Z syndrome in the same round
            self._include_loop_detectors(self.circuit) # Add detectors comparing the ancillas measurements
            self.circuit.append("M", self.data_qubits)
            self._logical_error_observables(self.circuit) # Add logical error observables to the circuit 
        
        if self.rounds > 1:
            # Memory experiment 
            # Rounds 2 to rounds-1
            self._include_initial_detectors(self.circuit)
            loop_circuit = stim.Circuit()
            self._extract_syndrome(loop_circuit)
            self._include_loop_detectors(loop_circuit)
            self.circuit += loop_circuit * (self.rounds - 1)

            self._noise_spam(self.circuit, self.data_qubits) # Apply measurement noise to data qubits
            self.circuit.append("M", self.data_qubits) # Measure data qubits at the end of the final round
            self._include_final_detectors(self.circuit) # Add detectors comparing final ancilla with data qubits measurements
            self._logical_error_observables(self.circuit) # Add logical error observables to the circuit
        
    def _generate_coordinate_grid(self) -> tuple[dict, dict, dict]:
        """
        Generates the coordinate grid for the data qubits and ancillas in the toric code
        """
        d =  self.d
        grid_total_L = 2 * d 
        num_data_qubits = 2 * d * d
        num_ancillas = d * d # Number of X or Z ancillas => total ancillas = 2 * d * d
        data_qubits_count = 0 
        x_ancillas_count = 0
        z_ancillas_count = 0    

        data_coords = {}
        x_ancillas_coords = {}
        z_ancillas_coords = {}

        for y in range(grid_total_L):
            for x in range(grid_total_L):
                if x % 2 != 0 and y % 2 == 0: # First lattice for data qubits
                    qubit_id = data_qubits_count
                    data_qubits_count += 1
                    data_coords[qubit_id] = (x, y)
                elif x % 2 == 0 and y % 2 != 0: # Second lattice for data qubits
                    qubit_id = data_qubits_count
                    data_qubits_count += 1
                    data_coords[qubit_id] = (x, y)
                elif x % 2 == 0 and y % 2 == 0: # X-ancillas
                    qubit_id = x_ancillas_count + num_data_qubits
                    x_ancillas_count += 1
                    x_ancillas_coords[qubit_id] = (x, y)
                elif x % 2 != 0 and y % 2 != 0: # Z-ancillas
                    qubit_id = z_ancillas_count + num_data_qubits + num_ancillas
                    z_ancillas_count += 1
                    z_ancillas_coords[qubit_id] = (x, y)
        return data_coords, x_ancillas_coords, z_ancillas_coords

    def _initialization_data_qubits(self, circuit):
        circuit.append("R", self.data_qubits)  # Initialize data qubits in |0>_L state
        self._noise_spam(circuit, self.data_qubits) # Apply state preparation noise to data qubits
        circuit.append("TICK")

    def _reset_ancillas(self, circuit):
        circuit.append("R", self.x_ancillas + self.z_ancillas)   # Initialize X and Z-type ancilla qubits in |0>_L
        self._noise_spam(circuit, self.x_ancillas + self.z_ancillas) 
        circuit.append("TICK")

    def _prepare_x_ancillas(self, circuit):
        circuit.append("H", self.x_ancillas) # Prepare X-ancillas in |+>
        self._noise_single_qubit(circuit, self.x_ancillas)
        circuit.append("TICK")

    def _CNOT_schedule(self, circuit):
        """
        Schedule the CNOTs between ancillas and data qubits to avoid hook errros. 
        """    
        # We are following the N -> E -> S -> W order for CNOTs
        cycle = [(0, -1), (-1, 0), (1, 0), (0, 1)] # Relative positions of ancillas to data qubits for CNOTs
        for dx, dy in cycle: # dx, dy are the relative positions
            parallel_cnot_pairs = []
            
            # X-ancillas interact with data qubits
            for ancilla in self.x_ancillas:
                ax, ay = self.x_ancillas_coords[ancilla]
                target_coord = ((ax + dx) % (2 * self.d), (ay + dy) % (2 * self.d)) # Wrap around for toric code
                if target_coord in self.coords_to_data:
                    data_qubit = self.coords_to_data[target_coord]
                    parallel_cnot_pairs.extend((ancilla, data_qubit)) # (ctr, tgt)
            
            # Z-ancillas interact with data qubits
            for ancilla in self.z_ancillas:
                ax, ay = self.z_ancillas_coords[ancilla]
                target_coord = ((ax + dx) % (2 * self.d), (ay + dy) % (2 * self.d)) # Wrap around for toric code
                if target_coord in self.coords_to_data:
                    data_qubit = self.coords_to_data[target_coord]
                    parallel_cnot_pairs.extend((data_qubit, ancilla)) # (ctr, tgt)
            
            # Apply the parallel CNOTs for this cycle
            if parallel_cnot_pairs:
                circuit.append("CX", parallel_cnot_pairs)
                self._noise_two_qubit(circuit, parallel_cnot_pairs)
                circuit.append("TICK")
    
    def _return_x_ancillas_to_z_basis(self, circuit):
        circuit.append("H", self.x_ancillas)
        self._noise_single_qubit(circuit, self.x_ancillas)
        circuit.append("TICK")

    def _measure_ancillas(self, circuit):
        self._noise_spam(circuit, self.x_ancillas + self.z_ancillas) 
        circuit.append("M", self.x_ancillas + self.z_ancillas)
        circuit.append("TICK")

    def _extract_syndrome(self, circuit):
        self._reset_ancillas(circuit)
        self._prepare_x_ancillas(circuit)
        self._CNOT_schedule(circuit)
        self._return_x_ancillas_to_z_basis(circuit)
        self._measure_ancillas(circuit)

    def _include_initial_detectors(self, circuit):
        """
        Add detectors that compare the current round's ancilla measurements with the ideal (no error) case
        """
        # Asuming a Z-memory experiment
        num_z_ancillas = len(self.z_ancillas)

        for i in range(num_z_ancillas):
            circuit.append("DETECTOR", [stim.target_rec(-i-1)])

    def _include_loop_detectors(self, circuit):
        """
        Add detectors that compare the current round's ancilla measurements with the previous round's measurements
        """
        num_x_ancillas = len(self.x_ancillas)
        num_z_ancillas = len(self.z_ancillas)
        num_tot_ancillas = num_x_ancillas + num_z_ancillas

        for i in range(num_x_ancillas):
            circuit.append("DETECTOR", [stim.target_rec(- num_z_ancillas - i - 1), 
                                         stim.target_rec(- num_z_ancillas - num_tot_ancillas - i - 1)])

        for i in range(num_z_ancillas):
            circuit.append("DETECTOR", [stim.target_rec(- i - 1), 
                                         stim.target_rec(- num_tot_ancillas - i - 1)])
            
    def _include_final_detectors(self, circuit):
        """
        Add detectors that compare the final round's ancilla measurements with the final data qubits measurements.
        """
        neighbours = [(-1, 0), (1, 0), (0, -1), (0, 1)] # Relative positions of ancillas to data qubits
        num_data_qubits = len(self.data_qubits)

        for count_ancilla, ancilla in enumerate(self.z_ancillas):
            ax, ay = self.z_ancillas_coords[ancilla]
            data_neighbours = []
            for dx, dy in neighbours:
                target_coord = ((ax + dx) % (2 * self.d), (ay + dy) % (2 * self.d)) # Wrap around for toric code
                data_qubit = self.coords_to_data[target_coord]
                data_neighbours.append(data_qubit)
            
            data_neighbours_offset = [stim.target_rec(-num_data_qubits + self.data_qubits.index(neighbour)) for neighbour in data_neighbours]
            ancilla_target = stim.target_rec(-num_data_qubits - len(self.z_ancillas) + count_ancilla)
            rec_targets = data_neighbours_offset + [ancilla_target]

            circuit.append("DETECTOR", rec_targets)

    def _logical_error_observables(self, circuit):
        """
        Adds the logical error observables to the circuit. 
        For the toric code 4 logical observables: X1, Z1, X2, Z2.
        """
        
        d = self.d
        data_qubits = self.data_qubits

        # Z1: Vertical loop of Z operators
        z1_qubits = [data_qubits[d*(2*k+1)] for k in range(d)] # Data qubits in the vertical line, coordinates (0, 2k+1)

        # For the moment I'm just taking the Z1 and Z2 logical operators, since that's the basis that is currently being measured
        # X1: Horizontal loop of X operators
        # x1_qubits = [data_qubits[k + d] for k in range(d)] # Data qubits in the horizontal line, coordinates (2k,1)

        # Z2: Vertical loop of Z operators
        z2_qubits = [data_qubits[k] for k in range(d)] # Data qubits in the vertical line, coordinates (2k+1,0)

        # X2: Horizontal loop of X operators
        # x2_qubits = [data_qubits[2*d*k] for k in range(d)] # Data qubits in the horizontal line, coordinates (1,2k)
        
        # Add the logical observables to the circuit
        for obs_idx, logical_qubits in enumerate([z1_qubits, z2_qubits]):
            num_data_qubits = len(self.data_qubits)
            targets = [stim.target_rec(-num_data_qubits + self.data_qubits.index(qubit)) for qubit in logical_qubits]
            circuit.append("OBSERVABLE_INCLUDE", targets, obs_idx)

    def _noise_single_qubit(self, circuit, qubits):
        """
        Applies single-qubit noise. 
        Optimized to use Stim built-in noise functions.
        """

        # Bit-flip errors
        px, py, pz = self.p_singleq

        if px == 0 and py == 0 and pz == 0:
            return # No noise to apply

        elif px > 0 and py == 0 and pz == 0:
            circuit.append("X_ERROR", qubits, px)

        # Phase-flip errors
        elif px == 0 and py == 0 and pz > 0:
            circuit.append("Z_ERROR", qubits, pz)

        # Depolarizing errors
        elif math.isclose(px, py) and math.isclose(py, pz) and math.isclose(px, pz):
            circuit.append("DEPOLARIZE1", qubits, sum(self.p_singleq))
        
        else:
            # Uses a general Pauli channel
            circuit.append("PAULI_CHANNEL_1", qubits, [px, py, pz])
    
    def _noise_two_qubit(self, circuit, qubits):
        """
        Applies two-qubit noise. 
        Optimized to use Stim built-in noise functions.
        """

        if all(p == 0 for p in self.p_twoq):
            return # No noise to apply
        
        # If all two-qubit error probabilities are equal
        elif all(math.isclose(p, self.p_twoq[0]) for p in self.p_twoq):
            circuit.append("DEPOLARIZE2", qubits, sum(self.p_twoq))

        else:
            # Uses a general two-qubit Pauli channel
            circuit.append("PAULI_CHANNEL_2", qubits, self.p_twoq)

    def _noise_spam(self, circuit, qubits):
        """
        Applies state preparation and measurement (SPAM) noise. 
        Optimized to use Stim built-in noise functions.
        """
        px, py, pz = self.p_spam

        if px == 0 and py == 0 and pz == 0:
            return # No noise to apply
        
        elif px > 0 and py == 0 and pz == 0:
            circuit.append("X_ERROR", qubits, px)

        elif px == 0 and py == 0 and pz > 0:
            circuit.append("Z_ERROR", qubits, pz)

        elif math.isclose(px, py) and math.isclose(py, pz) and math.isclose(px, pz):
            circuit.append("DEPOLARIZE1", qubits, sum(self.p_spam))

        else:
            # Uses a general Pauli channel for SPAM errors
            circuit.append("PAULI_CHANNEL_1", qubits, [px, py, pz])

    def print(self):
        print(self.circuit)

    def draw(self):
        return self.circuit.diagram('timeline-svg')

    def sample_detectors(self, shots: int, bit_packed: bool = False) -> np.ndarray:
        """
        Samples the detector events for the given number of shots. 
        bit_packed = True: Returns the detector events in a bit-packed format (for large d)
                   = False: Returns a 2D array of shape (shots, num_detectors) with boolean entries
        """
        sampler = self.circuit.compile_detector_sampler()
        detector_events, observable_flips = sampler.sample(shots, bit_packed = bit_packed) 
        return detector_events, observable_flips
    
    def save_circuit(self, filename: str):
        """
        Saves the circuit to a file in Stim's text format.
        """
        self.circuit.to_file(filename)
    
