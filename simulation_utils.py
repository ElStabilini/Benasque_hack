from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
from mitiq.zne import execute_with_zne
from mitiq.zne.inference import LinearFactory
from mitiq.zne.scaling import fold_gates_at_random
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.primitives import BackendEstimatorV2
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_algorithms.optimizers import NFT, SLSQP
from qiskit_ibm_runtime.fake_provider import FakeManilaV2
from qiskit_nature.second_q.circuit.library import UCC, UCCSD, HartreeFock
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from scipy.optimize import minimize

hardwareType = Literal["ion_trap", "superconducting"]
optimizerType = Literal["COBYLA", "SLSQP", "NFT"]
ansatzType = Literal["UCCSD", "UCC", "HEA_TI", "HEA_RING"]
mapperType = Literal["jw", "parity"]


def exact_ground_energy(hamiltonian) -> float:
    return float(np.min(np.linalg.eigvalsh(hamiltonian.to_matrix())))


def build_molecule_hamiltonian_jw(
    atom: str = "H 0 0 0; H 0 0 0.735",
    basis: str = "sto3g",
    charge: int = 0,
    spin: int = 0,
    active_electrons: int | None = None,
    active_orbitals: int | None = None,
) -> tuple:
    from qiskit_nature.second_q.drivers import PySCFDriver
    from qiskit_nature.second_q.mappers import JordanWignerMapper
    from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
    from qiskit_nature.units import DistanceUnit

    driver = PySCFDriver(
        atom=atom,
        basis=basis,
        charge=charge,
        spin=spin,
        unit=DistanceUnit.ANGSTROM,
    )
    problem = driver.run()

    if active_electrons is not None and active_orbitals is not None:
        transformer = ActiveSpaceTransformer(active_electrons, active_orbitals)
        problem = transformer.transform(problem)

    hamiltonian_jw = JordanWignerMapper().map(problem.hamiltonian.second_q_op())
    nuclear_repulsion = problem.nuclear_repulsion_energy

    num_spatial_orbitals = problem.num_spatial_orbitals
    num_particles = problem.num_particles  # (n_alpha, n_beta)
    classical_shift = sum(problem.hamiltonian.constants.values())

    e_ground_elec = exact_ground_energy(hamiltonian_jw)
    print(f"Molecule:                            {atom}")
    print(f"Basis:                               {basis}")
    print(f"Qubits (JW mapping):                 {hamiltonian_jw.num_qubits}")
    print(f"Spatial orbitals:                    {num_spatial_orbitals}")
    print(f"Electrons (alpha, beta):             {num_particles}")
    print(f"Exact Electronic Ground Energy:      {e_ground_elec:.6f} Ha")
    print(
        f"Exact Total Ground State Energy:     {e_ground_elec + nuclear_repulsion:.6f} Ha"
    )
    print(f"classical shift (Hamiltonian constant terms): {classical_shift:.6f} Ha")

    return (
        hamiltonian_jw,
        nuclear_repulsion,
        e_ground_elec,
        num_spatial_orbitals,
        num_particles,
        classical_shift,
    )


def build_molecule(
    atom="H 0 0 0; H 0 0 0.735",
    basis="sto3g",
    mapper_name: mapperType = "jw",
    active_electrons: int | None = None,
    active_orbitals: int | None = None,
):
    """
    Build qubit Hamiltonian and molecular problem using PySCF + JW mapping.
    """

    driver = PySCFDriver(atom=atom, basis=basis)
    problem = driver.run()

    if active_electrons is not None and active_orbitals is not None:
        transformer = ActiveSpaceTransformer(
            num_electrons=active_electrons,
            num_spatial_orbitals=active_orbitals,
        )
        problem = transformer.transform(problem)

    if mapper_name == "jw":
        mapper = JordanWignerMapper()
    elif mapper_name == "parity":
        mapper = ParityMapper(num_particles=problem.num_particles)
    else:
        raise ValueError(mapper_name)

    qubit_op = mapper.map(problem.hamiltonian.second_q_op())

    nuclear_shift = float(problem.nuclear_repulsion_energy)

    return problem, qubit_op, mapper, nuclear_shift


def build_ansatz(
    problem,
    mapper,
    ansatz_type: ansatzType,
    hea_layers: int = 2,
    ucc_excitations: str = "d",
):
    initial_state = HartreeFock(
        problem.num_spatial_orbitals, problem.num_particles, mapper
    )

    if ansatz_type == "UCCSD":
        ansatz = UCCSD(
            problem.num_spatial_orbitals,
            problem.num_particles,
            mapper,
            initial_state=initial_state,
        )

    elif ansatz_type == "UCC":
        ansatz = UCC(
            num_spatial_orbitals=problem.num_spatial_orbitals,
            num_particles=problem.num_particles,
            qubit_mapper=mapper,
            excitations=ucc_excitations,
            initial_state=initial_state,
        )

    elif ansatz_type == "HEA_TI":
        n_qubits = problem.num_spatial_orbitals * 2
        qc = QuantumCircuit(n_qubits)

        params = ParameterVector("θ", length=n_qubits * 3)

        idx = 0
        for q in range(n_qubits):
            qc.rx(params[idx], q)
            qc.ry(params[idx + 1], q)
            qc.rz(params[idx + 2], q)
            idx += 3

        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                qc.rxx(np.pi / 2, i, j)

        ansatz = qc

    elif ansatz_type == "HEA_RING":
        n_qubits = problem.num_spatial_orbitals * 2
        qc = QuantumCircuit(n_qubits)
        params = ParameterVector("θ", length=n_qubits * hea_layers)

        idx = 0
        for _ in range(hea_layers):
            for q in range(n_qubits):
                qc.ry(params[idx], q)
                idx += 1

            for q in range(n_qubits - 1):
                qc.cx(q, q + 1)

            if n_qubits > 1:
                qc.cx(n_qubits - 1, 0)

        ansatz = qc

    else:
        raise ValueError(ansatz_type)

    return ansatz


def build_hardware_backend(
    hardware: hardwareType,
    num_qubits: int,
    p1q_error=0.001,
    p2q_error=0.03,
    shots=8192,
):
    noise_model = NoiseModel()

    if hardware == "ion_trap":
        coupling = CouplingMap.from_full(num_qubits)
        basis = ["rx", "ry", "rz", "rxx"]

        noise_model.add_all_qubit_quantum_error(
            depolarizing_error(p1q_error, 1), ["rx", "ry", "rz"]
        )

        noise_model.add_all_qubit_quantum_error(
            depolarizing_error(p2q_error, 2), ["rxx"]
        )

    elif hardware == "superconducting":
        coupling = FakeManilaV2().coupling_map
        basis = ["sx", "rz", "cx"]

        noise_model.add_all_qubit_quantum_error(
            depolarizing_error(p1q_error, 1), ["sx", "rz"]
        )

        noise_model.add_all_qubit_quantum_error(
            depolarizing_error(p2q_error, 2), ["cx"]
        )

    else:
        raise ValueError(hardware)

    backend = AerSimulator(noise_model=noise_model, shots=shots)

    estimator = BackendEstimatorV2(backend=backend)

    return estimator, coupling, basis


def transpile_for_hardware(ansatz, observable, coupling, basis):
    pm = generate_preset_pass_manager(
        optimization_level=3, basis_gates=basis, coupling_map=coupling
    )

    transpiled_ansatz = pm.run(ansatz)
    transpiled_obs = observable.apply_layout(transpiled_ansatz.layout)

    return transpiled_ansatz, transpiled_obs


def get_optimizer(name: optimizerType, maxiter=40):
    if name == "COBYLA":
        return lambda fn, x0: minimize(
            fn, x0=x0, method="COBYLA", options={"maxiter": maxiter}
        )

    elif name == "SLSQP":
        return lambda fn, x0: SLSQP(maxiter=maxiter).minimize(fn, x0)

    elif name == "NFT":
        return lambda fn, x0: NFT(maxiter=maxiter).minimize(fn, x0)

    else:
        raise ValueError(name)


def run_vqe(
    estimator,
    ansatz,
    observable,
    nuclear_shift,
    initial_point,
    optimizer_name: optimizerType,
    maxiter=50,
):
    history = []

    def cost_fn(params):
        pub = (ansatz.assign_parameters(params), observable)

        res = estimator.run([pub]).result()[0]

        energy = float(np.atleast_1d(res.data.evs)[0]) + nuclear_shift

        history.append(energy)

        return energy

    optimizer = get_optimizer(optimizer_name, maxiter=maxiter)

    result = optimizer(cost_fn, initial_point)

    return result, history


def apply_zne(estimator, circuit, observable, nuclear_shift):
    def mitiq_executor(circ):
        pub = (circ, observable)

        res = estimator.run([pub]).result()[0]

        return float(np.atleast_1d(res.data.evs)[0])

    mitigated = execute_with_zne(
        circuit=circuit,
        executor=mitiq_executor,
        factory=LinearFactory(scale_factors=[1.0, 1.5, 2.0]),
        scale_noise=fold_gates_at_random,
    )

    return mitigated + nuclear_shift


def run_experiment(
    molecule="H2",
    ansatz_type: ansatzType = "UCCSD",
    optimizers=("COBYLA", "SLSQP", "NFT"),
    atom="H 0 0 0; H 0 0 0.735",
    basis="sto3g",
    mapper_name: mapperType = "jw",
    hea_layers: int = 2,
    ucc_excitations: str = "d",
    fci_energy: float | None = None,
    seed: int = 42,
    maxiter: int = 40,
    p1q_error: float = 0.001,
    p2q_error: float = 0.03,
    shots: int = 8192,
):
    problem, qubit_op, mapper, nuclear_shift = build_molecule(
        atom=atom,
        basis=basis,
        mapper_name=mapper_name,
    )

    ansatz = build_ansatz(
        problem,
        mapper,
        ansatz_type,
        hea_layers=hea_layers,
        ucc_excitations=ucc_excitations,
    )

    np.random.seed(seed)
    initial_point = np.random.uniform(-np.pi, np.pi, ansatz.num_parameters)

    ideal_estimator = BackendEstimatorV2(backend=AerSimulator(shots=shots))

    ion_estimator, ion_coupling, ion_basis = build_hardware_backend(
        "ion_trap",
        num_qubits=qubit_op.num_qubits,
        p1q_error=p1q_error,
        p2q_error=p2q_error,
        shots=shots,
    )

    sc_estimator, sc_coupling, sc_basis = build_hardware_backend(
        "superconducting",
        num_qubits=qubit_op.num_qubits,
        p1q_error=p1q_error,
        p2q_error=p2q_error,
        shots=shots,
    )

    ion_ansatz, ion_obs = transpile_for_hardware(
        ansatz, qubit_op, ion_coupling, ion_basis
    )
    sc_ansatz, sc_obs = transpile_for_hardware(ansatz, qubit_op, sc_coupling, sc_basis)

    _, ideal_history = run_vqe(
        ideal_estimator,
        ion_ansatz,
        ion_obs,
        nuclear_shift,
        initial_point,
        "COBYLA",
        maxiter=maxiter,
    )

    results = {"SC": {}, "Ion": {}}

    for opt in optimizers:
        print(f"Running ion_trap with {opt}")

        ion_result, ion_history = run_vqe(
            ion_estimator,
            ion_ansatz,
            ion_obs,
            nuclear_shift,
            initial_point,
            opt,
            maxiter=maxiter,
        )
        ion_zne = apply_zne(
            ion_estimator,
            ion_ansatz.assign_parameters(ion_result.x),
            ion_obs,
            nuclear_shift,
        )
        results["Ion"][opt] = {"hist": ion_history, "zne": ion_zne}

        print(f"Running superconducting with {opt}")

        sc_result, sc_history = run_vqe(
            sc_estimator,
            sc_ansatz,
            sc_obs,
            nuclear_shift,
            initial_point,
            opt,
            maxiter=maxiter,
        )
        sc_zne = apply_zne(
            sc_estimator,
            sc_ansatz.assign_parameters(sc_result.x),
            sc_obs,
            nuclear_shift,
        )
        results["SC"][opt] = {"hist": sc_history, "zne": sc_zne}

    exact_total_energy = exact_ground_energy(qubit_op) + nuclear_shift

    return {
        "fci_energy": fci_energy if fci_energy is not None else exact_total_energy,
        "ideal_history": ideal_history,
        "sc_data": results["SC"],
        "ion_data": results["Ion"],
        "depths": {"Ion": ion_ansatz.depth(), "SC": sc_ansatz.depth()},
    }


def run_active_space_experiment(
    molecule="LiH",
    atom="Li 0 0 0; H 0 0 1.5",
    basis="sto3g",
    mapper_name: mapperType = "parity",
    active_electrons: int = 2,
    active_orbitals: int = 3,
    ansatz_type: ansatzType = "UCCSD",
    optimizers=("COBYLA", "SLSQP", "NFT"),
    fci_energy: float | None = None,
    hea_layers: int = 2,
    ucc_excitations: str = "d",
    seed: int = 42,
    maxiter: int = 50,
    p1q_error: float = 0.001,
    p2q_error: float = 0.03,
    shots: int = 8192,
):
    problem, qubit_op, mapper, nuclear_shift = build_molecule(
        atom=atom,
        basis=basis,
        mapper_name=mapper_name,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
    )

    ansatz = build_ansatz(
        problem,
        mapper,
        ansatz_type,
        hea_layers=hea_layers,
        ucc_excitations=ucc_excitations,
    )

    np.random.seed(seed)
    initial_point = np.random.uniform(-np.pi, np.pi, ansatz.num_parameters)

    ideal_estimator = BackendEstimatorV2(backend=AerSimulator(shots=shots))

    ion_estimator, ion_coupling, ion_basis = build_hardware_backend(
        "ion_trap",
        num_qubits=qubit_op.num_qubits,
        p1q_error=p1q_error,
        p2q_error=p2q_error,
        shots=shots,
    )

    sc_estimator, sc_coupling, sc_basis = build_hardware_backend(
        "superconducting",
        num_qubits=qubit_op.num_qubits,
        p1q_error=p1q_error,
        p2q_error=p2q_error,
        shots=shots,
    )

    ion_ansatz, ion_obs = transpile_for_hardware(
        ansatz, qubit_op, ion_coupling, ion_basis
    )
    sc_ansatz, sc_obs = transpile_for_hardware(ansatz, qubit_op, sc_coupling, sc_basis)

    _, ideal_history = run_vqe(
        ideal_estimator,
        ion_ansatz,
        ion_obs,
        nuclear_shift,
        initial_point,
        "COBYLA",
        maxiter=maxiter,
    )

    results = {"SC": {}, "Ion": {}}

    for optimizer_name in optimizers:
        ion_result, ion_history = run_vqe(
            ion_estimator,
            ion_ansatz,
            ion_obs,
            nuclear_shift,
            initial_point,
            optimizer_name,
            maxiter=maxiter,
        )
        ion_zne = apply_zne(
            ion_estimator,
            ion_ansatz.assign_parameters(ion_result.x),
            ion_obs,
            nuclear_shift,
        )
        results["Ion"][optimizer_name] = {"hist": ion_history, "zne": ion_zne}

        sc_result, sc_history = run_vqe(
            sc_estimator,
            sc_ansatz,
            sc_obs,
            nuclear_shift,
            initial_point,
            optimizer_name,
            maxiter=maxiter,
        )
        sc_zne = apply_zne(
            sc_estimator,
            sc_ansatz.assign_parameters(sc_result.x),
            sc_obs,
            nuclear_shift,
        )
        results["SC"][optimizer_name] = {"hist": sc_history, "zne": sc_zne}

    return {
        "fci_energy": fci_energy,
        "ideal_history": ideal_history,
        "sc_data": results["SC"],
        "ion_data": results["Ion"],
        "depths": {"Ion": ion_ansatz.depth(), "SC": sc_ansatz.depth()},
    }


def export_results(results, molecule="H2", ansatz="UCCSD", filename: str | None = None):
    folder = Path(f"data/{molecule}")
    folder.mkdir(parents=True, exist_ok=True)

    path = (
        folder / filename
        if filename
        else folder / f"{ansatz}_{molecule}_analytics.json"
    )

    with open(path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Saved results to {path}")
    return path
