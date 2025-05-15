import numpy as np
import qutip as qt
from typing import Dict, Any, List, Tuple, Optional
from tqdm import tqdm
from ..backend_interface import QuantumBackend, process_results

class QuTiPBackend(QuantumBackend):
    """QuTiP backend implementation."""
    
    def create_rydberg_hamiltonian(
        self,
        n_atoms: int, 
        lattice_spacing: float,
        rabi_frequency: float,
        detunings: np.ndarray,
        encoding_scale: float,
        use_gpu: bool = False
    ) -> qt.Qobj:
        """
        Create a Rydberg atom Hamiltonian using QuTiP.
        
        Parameters
        ----------
        n_atoms : int
            Number of atoms in the chain
        lattice_spacing : float
            Spacing between atoms in μm
        rabi_frequency : float
            Rabi frequency (constant)
        detunings : np.ndarray
            Array of detunings for each atom
        encoding_scale : float
            Scale for encoding features as detunings
        use_gpu : bool, optional
            Whether to use GPU for Hamiltonian construction, by default False
            
        Returns
        -------
        qt.Qobj
            QuTiP Hamiltonian operator
        """
        # Try to enable CuPy data layer if requested
        data_type = "dense"
        cupyd_available = False
        
        if use_gpu:
            try:
                # Import qutip_cupy to register the CuPyDense data type
                import qutip_cupy
                try:
                    # Test it by creating a simple object
                    test_obj = qt.qeye(2).to("cupyd")
                    cupyd_available = True
                    data_type = "cupyd"
                    print("QuTiP GPU acceleration enabled via CuPy data layer")
                except (RuntimeError, ImportError, OSError) as gpu_err:
                    print(f"CuPy data layer initialization failed: {str(gpu_err)}")
                    print("Falling back to CPU mode")
                    data_type = "dense"
            except ImportError:
                print("qutip_cupy package not available. Using CPU instead.")
                data_type = "dense"
        
        # Define operators for each atom
        operators = []
        for i in range(n_atoms):
            # Create sigma_z operators for each atom (ProjectionOperator in Rydberg system)
            sz_i = qt.tensor([qt.sigmaz() if j == i else qt.qeye(2) for j in range(n_atoms)])
            if data_type == "cupyd" and cupyd_available:
                try:
                    sz_i = sz_i.to("cupyd")  # Convert to GPU-backed Qobj
                except Exception as e:
                    print(f"Warning: Failed to convert operator to GPU: {str(e)}")
                    print("Falling back to CPU mode for all operators")
                    data_type = "dense"
                    cupyd_available = False
            operators.append(sz_i)
        
        # Create sigma_x operators for each atom (drive terms)
        sx_operators = []
        for i in range(n_atoms):
            sx_i = qt.tensor([qt.sigmax() if j == i else qt.qeye(2) for j in range(n_atoms)])
            if data_type == "cupyd" and cupyd_available:
                try:
                    sx_i = sx_i.to("cupyd")  # Convert to GPU-backed Qobj
                except Exception:
                    # We already reported error above, just skip conversion
                    pass
            sx_operators.append(sx_i)
        
        # Create the Hamiltonian
        H = 0
        
        # Add drive terms (Rabi frequency * sigma_x)
        for i in range(n_atoms):
            try:
                H += 0.5 * rabi_frequency * sx_operators[i]
            except Exception as e:
                print(f"Error adding drive term: {str(e)}")
                print("Rebuilding Hamiltonian in CPU mode")
                # If we get an error, rebuild everything in CPU mode
                return self.create_rydberg_hamiltonian(
                    n_atoms=n_atoms,
                    lattice_spacing=lattice_spacing,
                    rabi_frequency=rabi_frequency,
                    detunings=detunings,
                    encoding_scale=encoding_scale,
                    use_gpu=False  # Force CPU mode
                )
        
        # Add detuning terms
        for i in range(n_atoms):
            # Apply both base detuning and feature-specific detuning
            detuning_value = 0.5 * encoding_scale * detunings[i] - encoding_scale
            try:
                H += 0.5 * detuning_value * operators[i]
            except Exception as e:
                print(f"Error adding detuning term: {str(e)}")
                print("Rebuilding Hamiltonian in CPU mode")
                # If we get an error, rebuild everything in CPU mode
                return self.create_rydberg_hamiltonian(
                    n_atoms=n_atoms,
                    lattice_spacing=lattice_spacing,
                    rabi_frequency=rabi_frequency,
                    detunings=detunings,
                    encoding_scale=encoding_scale,
                    use_gpu=False  # Force CPU mode
                )
        
        # Add Rydberg interactions (van der Waals)
        c6 = 862690  # C6 coefficient for Rb atoms (in MHz * μm^6)
        
        # Create full identity operator with the correct dimensions
        full_identity = qt.tensor([qt.qeye(2) for _ in range(n_atoms)])
        if data_type == "cupyd" and cupyd_available:
            try:
                full_identity = full_identity.to("cupyd")
            except Exception:
                # We already reported error above, just skip conversion
                pass
        
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                # Calculate distance
                distance = lattice_spacing * abs(i - j)
                # Van der Waals interaction term V(r) = C6/r^6
                interaction = c6 / (distance ** 6)
                
                try:
                    # Projector for |rr⟩ interaction - corrected to use proper dimensions
                    # n_i = (I - σ_z)/2 for each atom
                    n_i = (full_identity - operators[i]) / 2
                    n_j = (full_identity - operators[j]) / 2
                    H += interaction * n_i * n_j
                except Exception as e:
                    print(f"Error adding interaction term: {str(e)}")
                    print("Rebuilding Hamiltonian in CPU mode")
                    # If we get an error, rebuild everything in CPU mode
                    return self.create_rydberg_hamiltonian(
                        n_atoms=n_atoms,
                        lattice_spacing=lattice_spacing,
                        rabi_frequency=rabi_frequency,
                        detunings=detunings,
                        encoding_scale=encoding_scale,
                        use_gpu=False  # Force CPU mode
                    )
        
        return H

    def simulate_dynamics(
        self,
        H: qt.Qobj,
        t_list: List[float],
        n_atoms: int,
        n_shots: int = 1000,
        use_gpu: bool = False
    ) -> List[np.ndarray]:
        """
        Simulate the quantum dynamics and sample measurement outcomes.
        
        Parameters
        ----------
        H : qt.Qobj
            Hamiltonian operator
        t_list : List[float]
            List of time points
        n_atoms : int
            Number of atoms
        n_shots : int, optional
            Number of shots for simulation, by default 1000
        use_gpu : bool, optional
            Whether to use GPU acceleration, by default False
            
        Returns
        -------
        List[np.ndarray]
            Measurement samples at each time point
        """
        # Check if we're using CuPy data layer
        using_gpu_data = False
        if use_gpu and hasattr(H, 'data') and hasattr(H.data, 'type'):
            using_gpu_data = (H.data.type == "cupyd")
        
        # If requested GPU but not using it, try to convert
        if use_gpu and not using_gpu_data:
            try:
                # Try to convert to CuPy data layer
                import qutip_cupy
                try:
                    H = H.to("cupyd")
                    using_gpu_data = True
                    print("Successfully converted Hamiltonian to GPU data type")
                except Exception as e:
                    print(f"Cannot convert Hamiltonian to GPU: {str(e)}")
                    print("Continuing with CPU data type")
            except ImportError:
                print("qutip_cupy not available. Using CPU data type.")
        
        # Initial state: all atoms in ground state
        psi0 = qt.tensor([qt.basis(2, 0) for _ in range(n_atoms)])
        
        # Convert psi0 to same data type as H if needed
        if using_gpu_data:
            try:
                psi0 = psi0.to("cupyd")
            except Exception as e:
                print(f"Cannot convert initial state to GPU: {str(e)}")
                print("Note: Mixed GPU/CPU operations may be slower")
        
        # Define collapse operators (optional - for dissipative dynamics)
        c_ops = []
        
        # Time evolution - note this runs on CPU regardless of data type
        print("Running QuTiP mesolve (CPU operation with potential GPU data acceleration)")
        result = qt.mesolve(H, psi0, t_list, c_ops=c_ops)
        
        # Generate measurement samples
        samples = []
        
        # Define measurement operators (sigmaz for each atom)
        measurements = [qt.tensor([qt.sigmaz() if j == i else qt.qeye(2) for j in range(n_atoms)]) 
                        for i in range(n_atoms)]
        
        # Convert measurement operators to same data type as H if needed
        if using_gpu_data:
            try:
                measurements = [m.to("cupyd") for m in measurements]
            except Exception as e:
                print(f"Cannot convert measurement operators to GPU: {str(e)}")
        
        # For each time step
        for t_idx, t in enumerate(t_list):
            state = result.states[t_idx]
            
            # Generate shots
            t_samples = np.zeros((n_shots, n_atoms))
            
            for shot in range(n_shots):
                # Perform projective measurement
                measurement_results = []
                
                # Measure each atom
                for m_op in measurements:
                    # Calculate expectation value
                    expct = qt.expect(m_op, state)
                    
                    # Probabilistic outcome based on expectation value
                    # sigmaz eigenvalues are +1 (ground) and -1 (excited)
                    # Convert to 0 (ground) and 1 (excited) for consistency with Bloqade
                    if np.random.random() < (1 - expct) / 2:
                        # Excited state (convert -1 to 1)
                        measurement_results.append(1)
                    else:
                        # Ground state (convert +1 to 0)
                        measurement_results.append(0)
                
                t_samples[shot] = np.array(measurement_results)
            
            samples.append(t_samples)
        
        return samples
    
    def build_task(self, QRC_parameters: Dict[str, Any], detunings: np.ndarray):
        """
        Build a quantum task using QuTiP.
        
        Parameters
        ----------
        QRC_parameters : Dict[str, Any]
            Dictionary with QRC parameters
        detunings : np.ndarray
            Array of detunings for each atom
        
        Returns
        -------
            A function that executes the QuTiP simulation when called
        """
        # Extract parameters
        n_atoms = int(QRC_parameters["atom_number"])
        lattice_spacing = float(QRC_parameters["lattice_spacing"])
        rabi_frequency = float(QRC_parameters["rabi_frequency"])
        total_time = float(QRC_parameters["total_time"])
        time_steps = int(QRC_parameters["time_steps"])
        encoding_scale = float(QRC_parameters["encoding_scale"])
        use_gpu = QRC_parameters.get("use_gpu", False)
        
        # Create time list for simulation
        delta_t = total_time / time_steps
        t_list = np.arange(1, time_steps + 1, 1) * delta_t
        
        # Create Hamiltonian with better error handling
        try:
            print(f"Creating Hamiltonian for {n_atoms} atoms (GPU mode: {use_gpu})")
            H = self.create_rydberg_hamiltonian(
                n_atoms=n_atoms,
                lattice_spacing=lattice_spacing, 
                rabi_frequency=rabi_frequency,
                detunings=detunings,
                encoding_scale=encoding_scale,
                use_gpu=use_gpu
            )
        except Exception as e:
            print(f"Error creating Hamiltonian: {str(e)}")
            print("Falling back to CPU mode")
            H = self.create_rydberg_hamiltonian(
                n_atoms=n_atoms,
                lattice_spacing=lattice_spacing, 
                rabi_frequency=rabi_frequency,
                detunings=detunings,
                encoding_scale=encoding_scale,
                use_gpu=False  # Force CPU mode
            )
        
        # Return a callable that will run the simulation
        def run_simulation(n_shots=1000):
            # Create a simple object to mimic Bloqade's report interface
            class QutipReport:
                def __init__(self, samples):
                    self.samples = samples
                    
                def bitstrings(self):
                    return self.samples
            
            # Run the simulation with proper error handling
            try:
                samples = self.simulate_dynamics(
                    H=H,
                    t_list=t_list,
                    n_atoms=n_atoms,
                    n_shots=n_shots,
                    use_gpu=use_gpu
                )
            except Exception as e:
                print(f"Error in simulation: {str(e)}")
                print("Retrying simulation in CPU mode")
                samples = self.simulate_dynamics(
                    H=H,
                    t_list=t_list,
                    n_atoms=n_atoms,
                    n_shots=n_shots,
                    use_gpu=False  # Force CPU mode
                )
            
            return QutipReport(samples)
        
        return run_simulation
    
    def get_embeddings(self, xs: np.ndarray, qrc_params: Dict[str, Any], 
                      num_examples: int, n_shots: int = 1000) -> np.ndarray:
        """
        Function to get the embeddings using QuTiP simulation.
        
        Parameters
        ----------
        xs : np.ndarray
            Training set
        qrc_params : Dict[str, Any]
            Dictionary with QRC parameters
        num_examples : int
            Number of examples to process
        n_shots : int, optional
            Number of shots for the quantum task, by default 1000
        
        Returns
        -------
        np.ndarray
            Array with the embeddings of the training set
        """
        embeddings = []
        
        iterator = tqdm(range(num_examples), desc="QuTiP simulation", unit="sample", position=2, leave=False) 
        
        # Process each example one at a time
        for i in iterator:   
            # Extract features for current example 
            features = xs[:, i] if len(xs.shape) > 1 else xs
            
            # Build and run quantum task with QuTiP
            task = self.build_task(
                        QRC_parameters=qrc_params, 
                        detunings=features
                    )
            result = task(n_shots=n_shots)
            
            # Process results and add to embeddings
            embedding = process_results(qrc_params, result)
            embeddings.append(embedding)
        
        return np.column_stack(embeddings)
