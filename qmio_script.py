import logging
import time
from datetime import datetime

import numpy as np
from qiskit import transpile
from qiskit.circuit.library import TwoLocal
from qiskit_algorithms.minimum_eigensolvers import (
    QAOA,
    NumPyMinimumEigensolver,
    SamplingVQE,
)
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qmiotools.integrations.qiskitqmio import QmioBackend

from route_optimizer_functions import (
    build_qubo,
    default_locations,
    enrich_with_routes,
    make_backend_sampler,
    print_summary,
    save_json,
)

# Set up logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

# Defining data
gear_level = {"Urban": 0, "Trail": 1, "Mountain": 2, "Snow": 3}
pois = [4, 5, 6]
scenic = {4: 4.0, 5: 3.5, 6: 8.0}
season = "summer"
user_gear = gear_level["Mountain"]
travel_time = np.array(
    [
        [0.00, 1.20, 0.45, 2.50],
        [1.20, 0.00, 1.30, 3.00],
        [0.45, 1.30, 0.00, 1.42],
        [2.50, 3.00, 1.42, 0.00],
    ],
    dtype=float,
)
visit_time = {4: 1.0, 5: 0.5, 6: 2.0}
penalties = {
    "A": 12.0,
    "B": 10.0,
    "C": 6.0,
    "D": 8.0,
    "E": 5.0,
    "F": 50.0,
    "T_MAX": 24.0,
    "T_MAX_REFUGIO": 48.0,
    "MAX_ALT_GAIN": 800,
}

qubo_problem = build_qubo(
    locations=default_locations(),
    base_altitude=1135,
    base_camp=0,
    gear_level=gear_level,
    pois=pois,
    scenic=scenic,
    season=season,
    user_gear=user_gear,
    travel_time=travel_time,
    visit_time=visit_time,
    n_slots=2,
    penalties=penalties,
)

op, _ = qubo_problem.to_ising()
num_qubits = op.num_qubits

# Set up quantum backend and sampler
backend = QmioBackend(reservation_name="Benasque_QPU")
shots = 1000
sampler = make_backend_sampler(backend, shots=shots)

results = {}

# NumPy exact solver (classical benchmark)
log.info("Running exact NumPy solver …")
t0 = time.time()
exact_result = MinimumEigenOptimizer(NumPyMinimumEigensolver()).solve(qubo_problem)
results["exact"] = {
    "fval": float(exact_result.fval),
    "x": exact_result.x.tolist(),
    "elapsed_s": round(time.time() - t0, 3),
}
log.info(
    "  Exact cost = %.4f  (%.1f s)",
    results["exact"]["fval"],
    results["exact"]["elapsed_s"],
)

# QAOA on real hardware
qaoa_reps = 3
qaoa_maxiter = 300

log.info("Running QAOA (reps=%d, maxiter=%d) …", qaoa_reps, qaoa_maxiter)
t0 = time.time()
qaoa_solver = QAOA(
    sampler=sampler,
    optimizer=COBYLA(maxiter=qaoa_maxiter),
    reps=qaoa_reps,
)
qaoa_result = MinimumEigenOptimizer(qaoa_solver).solve(qubo_problem)
results["qaoa"] = {
    "fval": float(qaoa_result.fval),
    "x": qaoa_result.x.tolist(),
    "reps": qaoa_reps,
    "maxiter": qaoa_maxiter,
    "elapsed_s": round(time.time() - t0, 3),
}
log.info(
    "  QAOA cost = %.4f  (%.1f s)",
    results["qaoa"]["fval"],
    results["qaoa"]["elapsed_s"],
)

# VQE with TwoLocal ansatz on real hardware
vqe_reps = 3
vqe_maxiter = 800

log.info("Running VQE/TwoLocal (reps=%d, maxiter=%d) …", vqe_reps, vqe_maxiter)
t0 = time.time()
ansatz_twolocal = TwoLocal(
    num_qubits=num_qubits,
    rotation_blocks=["ry", "rz"],
    entanglement_blocks="cz",
    reps=vqe_reps,
)
ansatz_twolocal_t = transpile(ansatz_twolocal, backend=backend, optimization_level=2)
vqe_solver = SamplingVQE(
    sampler=sampler,
    ansatz=ansatz_twolocal_t,
    optimizer=COBYLA(maxiter=vqe_maxiter),
)
vqe_result = MinimumEigenOptimizer(vqe_solver).solve(qubo_problem)
results["vqe_twolocal"] = {
    "fval": float(vqe_result.fval),
    "x": vqe_result.x.tolist(),
    "reps": vqe_reps,
    "maxiter": vqe_maxiter,
    "ansatz_depth": ansatz_twolocal_t.depth(),
    "elapsed_s": round(time.time() - t0, 3),
}
log.info(
    "  VQE/TwoLocal cost = %.4f  depth=%d  (%.1f s)",
    results["vqe_twolocal"]["fval"],
    results["vqe_twolocal"]["ansatz_depth"],
    results["vqe_twolocal"]["elapsed_s"],
)

# VQE with custom HEA ansatz on real hardware
# hea_layers = 4
# hea_maxiter = 800
#
# log.info("Running VQE/HEA (layers=%d, maxiter=%d) …", hea_layers, hea_maxiter)
# t0 = time.time()
# custom_ansatz = create_hea_ansatz(num_qubits, layers=hea_layers)
# custom_ansatz_t = transpile(custom_ansatz, backend=backend, optimization_level=2)
# custom_vqe_solver = SamplingVQE(
#     sampler=sampler,
#     ansatz=custom_ansatz_t,
#     optimizer=COBYLA(maxiter=hea_maxiter),
# )
# results["vqe_hea"] = {
#     "fval": float(custom_vqe_result.fval),
#     "x": custom_vqe_result.x.tolist(),
#     "layers": hea_layers,
#     "maxiter": hea_maxiter,
#     "ansatz_depth": custom_ansatz_t.depth(),
#     "elapsed_s": round(time.time() - t0, 3),
# }
# log.info(
#     "  VQE/HEA cost = %.4f  depth=%d  (%.1f s)",
#     results["vqe_hea"]["fval"],
#     results["vqe_hea"]["ansatz_depth"],
#     results["vqe_hea"]["elapsed_s"],
# )

enrich_with_routes(results, pois=pois, slots=2)
print_summary(results)

output = {
    "meta": {
        "timestamp": datetime.now().isoformat(),
        "pois": pois,
        "n_slots": 2,
    },
    "results": results,
}

output_path = "qmio_results.json"
save_json(output, output_path)
log.info("Results saved to %s", output_path)
