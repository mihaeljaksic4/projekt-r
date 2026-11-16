import json
import math
import sys

from sage.all import CDF, RDF, DiGraph, identity_matrix, matrix, vector


def analyze(data: dict) -> dict:
    states = data["states"]
    mask = data["mask"]
    means = data["means"]
    variances = data["variances"]
    features = data["feature_names"]
    cadence = float(data["cadence_seconds"])
    n = len(states)
    a = matrix(RDF, data["transitions"])

    # Graph theory concerns allowed edges, not the size of learned probabilities.
    graph = DiGraph(
        {i: [j for j in range(n) if mask[i][j]] for i in range(n)},
        format="dict_of_lists",
        loops=True,
    )
    shortest_paths = [[int(graph.distance(i, j)) for j in range(n)] for i in range(n)]

    # Solve pi*A = pi, sum(pi)=1 (ergodic, row-stochastic chain).
    system = a.transpose() - identity_matrix(RDF, n)
    for j in range(n):
        system[n - 1, j] = 1.0
    stationary = system.solve_right(vector(RDF, [0] * (n - 1) + [1]))
    p = [max(0.0, float(v)) for v in stationary]
    p = [v / sum(p) for v in p]

    eigenvalues = sorted(
        (complex(v) for v in a.change_ring(CDF).eigenvalues()),
        key=abs,
        reverse=True,
    )
    second_modulus = abs(eigenvalues[1]) if n > 1 else 0.0
    spectral_gap = 1 - second_modulus
    relaxation = -cadence / math.log(second_modulus) if 0 < second_modulus < 1 else None

    # Expected first-passage times, and the stationary-target Kemeny constant.
    hitting = [[0.0] * n for _ in range(n)]
    for target in range(n):
        other = [i for i in range(n) if i != target]
        q = matrix(RDF, [[a[i, j] for j in other] for i in other])
        h = (identity_matrix(RDF, n - 1) - q).solve_right(vector(RDF, [1] * (n - 1)))
        for i, value in zip(other, h):
            hitting[i][target] = max(0.0, float(value)) * cadence
    kemeny = [sum(p[j] * hitting[i][j] for j in range(n)) for i in range(n)]

    # Long-run inter-state events per hour. Do not include self-transitions.
    stationary_flow = [
        [float(p[i] * a[i, j]) * 3600 / cadence if i != j else 0.0 for j in range(n)]
        for i in range(n)
    ]
    entry_rates = {
        states[j]: sum(stationary_flow[i][j] for i in range(n)) for j in range(n)
    }
    entropy = {
        states[i]: -sum(
            float(a[i, j]) * math.log(float(a[i, j])) for j in range(n) if a[i, j] > 0
        )
        for i in range(n)
    }
    dwell = {
        states[i]: cadence / (1.0 - float(a[i, i])) if float(a[i, i]) < 1 else None
        for i in range(n)
    }

    # Worst-case total-variation distance to the stationary distribution.
    # This is more interpretable than the near-unity spectrum for persistent states.
    horizons = [0, 30, 60, 120, 300, 600, 1200, 2400]
    convergence = []
    for seconds in horizons:
        steps = int(round(seconds / cadence))
        power = a**steps
        worst_tv = max(
            0.5 * sum(abs(float(power[i, j]) - p[j]) for j in range(n))
            for i in range(n)
        )
        convergence.append(
            {"seconds": steps * cadence, "worst_total_variation": worst_tv}
        )

    # Bhattacharyya distance decomposes by feature for diagonal Gaussians.
    distances = [[0.0] * n for _ in range(n)]
    adjacent_explanations = []
    for i in range(n):
        for j in range(i + 1, n):
            contributions = []
            for d in range(len(features)):
                vi, vj = variances[i][d], variances[j][d]
                avg = (vi + vj) / 2
                distance = (means[i][d] - means[j][d]) ** 2 / (8 * avg)
                distance += 0.5 * math.log(avg / math.sqrt(vi * vj))
                contributions.append(float(max(0, distance)))
            total = sum(contributions)
            distances[i][j] = distances[j][i] = total
            if mask[i][j] or mask[j][i]:
                adjacent_explanations.append(
                    {
                        "states": [states[i], states[j]],
                        "distance": total,
                        "contributions": dict(zip(features, contributions)),
                    }
                )

    # Nontrivial LEFT eigenvectors describe slow probability redistribution.
    modes = []
    for eigenvalue, vectors, _ in sorted(
        a.transpose().change_ring(CDF).eigenvectors_right(),
        key=lambda item: abs(complex(item[0])),
        reverse=True,
    ):
        lam = complex(eigenvalue)
        if abs(lam - 1) < 1e-7 or lam.imag < -1e-8:
            continue  # omit the stationary mode and duplicate conjugate vectors
        amplitudes = [abs(complex(value)) for value in vectors[0]]
        maximum = max(amplitudes)
        modes.append(
            {
                "eigenvalue": {"real": lam.real, "imag": lam.imag},
                "amplitude": dict(zip(states, [v / maximum for v in amplitudes])),
                "timescale_seconds": -cadence / math.log(abs(lam))
                if 0 < abs(lam) < 1
                else None,
            }
        )
        if len(modes) == 3:
            break

    # Exact first-passage probabilities for a target SET, not just a mean time.
    horizons = [0, 10, 30, 60, 120, 300, 600, 1200]
    first_passage = {}
    for label, targets in (
        (
            "thermal",
            [
                i
                for i, state in enumerate(states)
                if state in ("THERMAL_RESPONSE", "THERMAL_STEADY")
            ],
        ),
        ("idle", [states.index("IDLE")]),
    ):
        other = [i for i in range(n) if i not in targets]
        q = matrix(RDF, [[a[i, j] for j in other] for i in other])
        curves = {state: [] for state in states}
        for seconds in horizons:
            power = q ** int(round(seconds / cadence))
            for i in range(n):
                survival = (
                    sum(float(power[other.index(i), j]) for j in range(len(other)))
                    if i in other
                    else 0.0
                )
                curves[states[i]].append(max(0.0, min(1.0, 1 - survival)))
        first_passage[label] = {
            "targets": [states[i] for i in targets],
            "horizons_seconds": horizons,
            "probability_by_start": curves,
        }

    # Committor: probability of hitting a thermal state before reaching IDLE.
    hot = [states.index(name) for name in ("THERMAL_RESPONSE", "THERMAL_STEADY")]
    idle = states.index("IDLE")
    middle = [i for i in range(n) if i not in hot and i != idle]
    q_middle = matrix(RDF, [[a[i, j] for j in middle] for i in middle])
    rhs = vector(RDF, [sum(a[i, j] for j in hot) for i in middle])
    solved = (identity_matrix(RDF, len(middle)) - q_middle).solve_right(rhs)
    committor = {
        state: 1.0
        if i in hot
        else 0.0
        if i == idle
        else max(0.0, min(1.0, float(solved[middle.index(i)])))
        for i, state in enumerate(states)
    }

    # Antisymmetric stationary currents: J_ij = pi_i*A_ij - pi_j*A_ji.
    currents = [
        [
            0.0
            if i == j
            else (p[i] * float(a[i, j]) - p[j] * float(a[j, i])) * 3600 / cadence
            for j in range(n)
        ]
        for i in range(n)
    ]

    # Analytic stationary sensitivity to reallocating 0.0001 probability
    # from self-loop i->i to an allowed outgoing transition i->j.
    sensitivity = []
    for i in range(n):
        for j in range(n):
            if i == j or not mask[i][j]:
                continue
            system = a.transpose() - identity_matrix(RDF, n)
            rhs = vector(RDF, [0] * n)
            rhs[i] = p[i]
            rhs[j] = -p[i]
            for col in range(n):
                system[n - 1, col] = 1.0
            rhs[n - 1] = 0.0
            derivative = system.solve_right(rhs)
            sensitivity.append(
                {
                    "edge": [states[i], states[j]],
                    "delta_transition_probability": 0.0001,
                    "delta_stationary_probability": {
                        states[k]: 0.0001 * float(derivative[k]) for k in range(n)
                    },
                }
            )

    return {
        "topology": {
            "strongly_connected": bool(graph.is_strongly_connected()),
            "components": [
                [states[i] for i in component]
                for component in graph.strongly_connected_components()
            ],
            "allowed_edges": int(graph.size()),
            "out_degree": {states[i]: int(graph.out_degree(i)) for i in range(n)},
            "shortest_allowed_paths_steps": shortest_paths,
        },
        "markov_chain": {
            "stationary_probabilities": dict(zip(states, p)),
            "expected_dwell_seconds": dwell,
            "expected_hitting_seconds": hitting,
            "transition_entropy_nats": entropy,
            "entropy_rate_nats_per_step": sum(
                p[i] * entropy[states[i]] for i in range(n)
            ),
            "eigenvalues": [{"real": v.real, "imag": v.imag} for v in eigenvalues],
            "spectral_gap": spectral_gap,
            "relaxation_seconds": relaxation,
            "kemeny_seconds": sum(kemeny) / n,
            "kemeny_source_spread_seconds": max(kemeny) - min(kemeny),
            "stationary_offdiagonal_flow_per_hour": stationary_flow,
            "stationary_entries_per_hour": entry_rates,
            "worst_case_convergence": convergence,
            "slow_modes": modes,
            "first_passage": first_passage,
            "thermal_before_idle_committor": committor,
            "stationary_currents_per_hour": currents,
            "stationary_sensitivity": sensitivity,
            "first_exit_probability_30s": {
                states[i]: 1 - float(a[i, i]) ** int(round(30 / cadence))
                for i in range(n)
            },
        },
        "emissions": {
            "bhattacharyya_distance": distances,
            "adjacent_pair_feature_contributions": adjacent_explanations,
        },
    }


print(json.dumps(analyze(json.load(sys.stdin))))
