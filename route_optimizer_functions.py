import itertools
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import BackendSamplerV2
from qiskit_optimization import QuadraticProgram

log = logging.getLogger(__name__)
GEAR_LEVEL = {"Urban": 0, "Trail": 1, "Mountain": 2, "Snow": 3}


def default_locations():
    return {
        1: ("Pico Aneto", "Peak", 3404, "Snow", "Snow"),
        2: ("Tuca Alba", "Peak", 3122, "Snow", "Mountain"),
        3: ("Benasque", "Town", 1135, "Urban", "Urban"),
        4: ("Cerler", "Town", 1530, "Urban", "Urban"),
        5: ("Ancils", "Town", 1140, "Urban", "Urban"),
        6: ("Portillón de Benás", "Landmark", 2444, "Mountain", "Trail"),
        7: ("Ski resort", "Snow", 1530, "Snow", "Trail"),
        8: ("Baños de Benasque", "Resting area", 1550, "Urban", "Urban"),
        9: ("Hospital de Benasque", "Resting area", 1747, "Urban", "Urban"),
        10: ("Forau d'aiguallut", "Landmark", 2020, "Mountain", "Trail"),
        11: ("Tres Cascadas", "Landmark", 1900, "Mountain", "Trail"),
        12: ("Salvaguardia", "Peak", 2736, "Snow", "Mountain"),
        13: ("Tuca Maladeta", "Peak", 3312, "Snow", "Mountain"),
        14: ("Cap Llauset", "Refugio", 2425, "Snow", "Mountain"),
        15: ("Ibón Cregüeña", "Lake", 2632, "Snow", "Mountain"),
        16: ("Batisielles", "Lake", 2216, "Mountain", "Trail"),
        17: ("Eriste", "Town", 1089, "Urban", "Urban"),
        18: ("Ibón Eriste", "Lake", 2407, "Snow", "Mountain"),
        19: ("Tempestades", "Peak", 3289, "Snow", "Snow"),
        20: ("La Besurta", "Resting area", 1860, "Trail", "Trail"),
        21: ("La Renclusa", "Refugio", 2160, "Snow", "Trail"),
        22: ("Escaleta", "Lake", 2630, "Snow", "Mountain"),
        23: ("Mulleres", "Peak", 3013, "Snow", "Snow"),
        24: ("Salterillo", "Lake", 2460, "Snow", "Mountain"),
        25: ("Tres Barrancos", "Landmark", 1460, "Trail", "Trail"),
    }


def default_problem_data():
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

    return {
        "locations": default_locations(),
        "base_altitude": 1135,
        "base_camp": 0,
        "gear_level": gear_level,
        "pois": pois,
        "scenic": scenic,
        "season": season,
        "user_gear": user_gear,
        "travel_time": travel_time,
        "visit_time": visit_time,
        "n_slots": 2,
        "penalties": penalties,
    }


def gear_required(loc_id: int, season: str = "summer") -> int:
    """Return the numeric gear level required to visit a location."""
    col = 3 if season == "summer" else 4  # index into the tuple
    return GEAR_LEVEL[default_locations()[loc_id][col]]


def altitude(loc_id, locations):
    return locations[loc_id][2]


def is_refugio(loc_id, locations):
    return locations[loc_id][1] == "Refugio"


def make_distance_fn(travel_time, pois, base_camp=0):
    poi_to_dist_idx = {poi: k + 1 for k, poi in enumerate(pois)}

    def distance(a, b):
        ai = 0 if a == base_camp else poi_to_dist_idx[a]
        bi = 0 if b == base_camp else poi_to_dist_idx[b]
        return travel_time[ai][bi]

    return distance


def build_qubo(
    locations,
    base_altitude,
    base_camp,
    gear_level,
    pois,
    scenic,
    season,
    user_gear,
    travel_time,
    visit_time,
    n_slots,
    penalties,
):
    A = penalties["A"]
    B = penalties["B"]
    C = penalties["C"]
    D = penalties["D"]
    E = penalties["E"]
    F = penalties["F"]
    t_max = penalties["T_MAX"]
    t_max_refugio = penalties["T_MAX_REFUGIO"]
    max_alt_gain = penalties["MAX_ALT_GAIN"]

    slots = list(range(1, n_slots + 1))
    d = make_distance_fn(travel_time, pois, base_camp=base_camp)

    qp = QuadraticProgram("benasque_hiking_route")
    for i in pois:
        for p in slots:
            qp.binary_var(name=f"x_{i}_{p}")

    linear = defaultdict(float)
    quadratic = defaultdict(float)
    constant = 0.0

    # Objective: scenic value and travel distances
    for i in pois:
        linear[f"x_{i}_1"] += d(base_camp, i) - scenic[i]
        linear[f"x_{i}_{n_slots}"] += d(i, base_camp) - scenic[i]
        for p in slots[1:-1]:
            linear[f"x_{i}_{p}"] += -scenic[i]

    for p, q in zip(slots, slots[1:]):
        for i in pois:
            for j in pois:
                quadratic[(f"x_{i}_{p}", f"x_{j}_{q}")] += d(i, j)

    # Penalty A: exactly one POI per slot
    for p in slots:
        constant += A
        for i in pois:
            linear[f"x_{i}_{p}"] += -A
        for i, j in itertools.combinations(pois, 2):
            quadratic[(f"x_{i}_{p}", f"x_{j}_{p}")] += 2 * A

    # Penalty B: no POI visited in more than one slot
    for i in pois:
        for p, q in itertools.combinations(slots, 2):
            quadratic[(f"x_{i}_{p}", f"x_{i}_{q}")] += B

    # Penalty C: altitude gain threshold
    for i in pois:
        gain_base_i = max(0.0, altitude(i, locations) - base_altitude)
        excess_base_i = max(0.0, gain_base_i - max_alt_gain)
        if excess_base_i > 0:
            linear[f"x_{i}_1"] += C * excess_base_i**2

        for j in pois:
            gain_ij = max(0.0, altitude(j, locations) - altitude(i, locations))
            excess_ij = max(0.0, gain_ij - max_alt_gain)
            if excess_ij > 0:
                for p, q in zip(slots, slots[1:]):
                    quadratic[(f"x_{i}_{p}", f"x_{j}_{q}")] += C * excess_ij**2

    # Penalty D: gear mismatch
    for i in pois:
        required = gear_required(i, locations, gear_level, season=season)
        deficit = max(0, required - user_gear)
        if deficit > 0:
            for p in slots:
                linear[f"x_{i}_{p}"] += D * deficit

    # Penalty E: time budget
    for i in pois:
        for j in pois:
            budget = (
                t_max_refugio
                if (is_refugio(i, locations) or is_refugio(j, locations))
                else t_max
            )
            for p, q in zip(slots, slots[1:]):
                leg_time = (
                    d(base_camp, i) + visit_time[i] + d(i, j)
                    if p == slots[0]
                    else visit_time[i] + d(i, j)
                )
                if q == slots[-1]:
                    leg_time += visit_time[j] + d(j, base_camp)
                excess = max(0.0, leg_time - budget)
                if excess > 0:
                    quadratic[(f"x_{i}_{p}", f"x_{j}_{q}")] += E * excess**2

    # Penalty F: hard Snow↔Mountain ban (cross-type adjacency)
    snow_gear = gear_level["Snow"]
    mount_gear = gear_level["Mountain"]
    for i in pois:
        for j in pois:
            gi = gear_required(i, locations, gear_level, season=season)
            gj = gear_required(j, locations, gear_level, season=season)
            if {gi, gj} == {snow_gear, mount_gear}:
                for p, q in zip(slots, slots[1:]):
                    quadratic[(f"x_{i}_{p}", f"x_{j}_{q}")] += F

    qp.minimize(constant=constant, linear=dict(linear), quadratic=dict(quadratic))
    return qp


def create_hea_ansatz(num_qubits, layers):
    qc = QuantumCircuit(num_qubits)
    thetas = ParameterVector("theta", num_qubits * layers)

    for layer in range(layers):
        for qubit in range(num_qubits):
            qc.ry(thetas[layer * num_qubits + qubit], qubit)
        for qubit in range(num_qubits - 1):
            qc.cx(qubit, qubit + 1)
        if num_qubits > 1:
            qc.cx(num_qubits - 1, 0)
        if layer < layers - 1:
            qc.barrier()

    return qc


def decode_solution(xvec, pois, slots, base_camp=0):
    var_names = [f"x_{i}_{p}" for i in pois for p in slots]
    sol = {name: int(round(val)) for name, val in zip(var_names, xvec)}

    route = [base_camp]
    for p in slots:
        chosen = next((i for i in pois if sol.get(f"x_{i}_{p}", 0) == 1), None)
        route.append(chosen)
    route.append(base_camp)
    return route


def total_hike_time(route, visit_time, distance_fn, base_camp=0):
    stops = route[1:-1]
    if None in stops:
        return float("nan")
    total = 0.0
    prev = base_camp
    for stop in stops:
        total += distance_fn(prev, stop) + visit_time[stop]
        prev = stop
    total += distance_fn(prev, base_camp)
    return total


def effective_budget(route, t_max, t_max_refugio, locations):
    stops = route[1:-1]
    if any(s is not None and is_refugio(s, locations) for s in stops):
        return t_max_refugio
    return t_max


def route_info(route, locations, base_camp, base_altitude, season):
    lines = []
    for idx, node in enumerate(route):
        if node == base_camp:
            lines.append(f"  Base camp (Benasque, {base_altitude} m)")
        else:
            name = locations[node][0]
            alt = altitude(node, locations)
            req = locations[node][3 if season == "summer" else 4]
            refugio = (
                " [REFUGIO — overnight possible]" if is_refugio(node, locations) else ""
            )
            lines.append(
                f"  Slot {idx}: POI {node}: {name} ({alt} m)"
                f" — needs {req} gear{refugio}"
            )
    return "\n".join(lines)


def print_summary(solver_results: dict) -> None:
    print("\n" + "═" * 60)
    print("  SUMMARY")
    print("═" * 60)
    for solver, res in solver_results.items():
        route_str = " → ".join(
            "Base" if n == 0 else str(n) for n in res.get("route", [])
        )
        print(
            f"  {solver:<16} cost={res['fval']:>10.4f}  "
            f"time={res['elapsed_s']:>7.1f}s  route=[{route_str}]"
        )
    print("═" * 60)


def print_configuration(data):
    penalties = data["penalties"]
    gear_name = [k for k, v in data["gear_level"].items() if v == data["user_gear"]][0]
    print(f"\n{'─' * 55}")
    print("Configuration:")
    print(f"  N_SLOTS               = {data['n_slots']}")
    print(f"  Season                = {data['season']}")
    print(f"  User gear             = {data['user_gear']} ({gear_name})")
    print("Penalty weights used:")
    print(f"  A (one POI per slot)  = {penalties['A']}")
    print(f"  B (no revisit)        = {penalties['B']}")
    print(
        f"  C (altitude gain)     = {penalties['C']}  [threshold: {penalties['MAX_ALT_GAIN']} m]"
    )
    print(f"  D (gear mismatch)     = {penalties['D']}")
    print(
        f"  E (time budget)       = {penalties['E']}"
        f"  [normal: {penalties['T_MAX']} h, refugio: {penalties['T_MAX_REFUGIO']} h]"
    )
    print(f"  F (Snow↔Mountain ban) = {penalties['F']}  [hard constraint]")


def save_json(data: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def enrich_with_routes(solver_results: dict, pois: list, slots: list) -> dict:
    """Decode and attach the route list to every entry in solver_results."""
    for key, res in solver_results.items():
        res["route"] = decode_solution(res["x"], pois=pois, slots=slots, base_camp=0)
    return solver_results


def make_backend_sampler(backend, shots: int) -> BackendSamplerV2:
    sampler = BackendSamplerV2(backend=backend)
    sampler.options.default_shots = shots
    return sampler
