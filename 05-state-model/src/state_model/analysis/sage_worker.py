import json
import math
import sys
from fractions import Fraction
from copy import copy

from sage.all import (
    CDF,
    QQ,
    RDF,
    DiGraph,
    PolynomialRing,
    identity_matrix,
    matrix,
    vector,
)


def rational_chain(rows):
    result = []
    for i, row in enumerate(rows):
        values = []
        for value in row:
            fraction = Fraction(f"{float(value):.12f}")
            values.append(QQ(fraction.numerator) / fraction.denominator)
        values[i] = 1 - sum(value for j, value in enumerate(values) if j != i)
        result.append(values)
    return matrix(QQ, result)


def tree_weights(a):
    laplacian = identity_matrix(a.base_ring(), a.nrows()) - a
    return [
        laplacian.matrix_from_rows_and_columns(
            [j for j in range(a.nrows()) if j != i],
            [j for j in range(a.nrows()) if j != i],
        ).det()
        for i in range(a.nrows())
    ]


def passage_analysis(a, exact, states, targets, cadence):
    n = len(states)
    other = [i for i in range(n) if i not in targets]
    q = a.matrix_from_rows_and_columns(other, other)
    horizons = [0, 10, 30, 60, 120, 300, 600, 1200]
    curves = {state: [] for state in states}

    for seconds in horizons:
        survival = q ** int(round(seconds / cadence)) * vector(RDF, [1] * len(other))
        for i, state in enumerate(states):
            curves[state].append(
                1.0
                if i in targets
                else max(0.0, min(1.0, 1 - float(survival[other.index(i)])))
            )

    # A single rational function F(z) encodes the complete first-hitting-time law.
    ring = PolynomialRing(QQ, "z")
    field = ring.fraction_field()
    z = field.gen()
    q_exact = exact.matrix_from_rows_and_columns(other, other).change_ring(field)
    arrivals = exact.matrix_from_rows_and_columns(other, targets) * vector(
        QQ, [1] * len(targets)
    )
    generating = (identity_matrix(field, len(other)) - z * q_exact).solve_right(
        z * vector(field, arrivals)
    )
    mean = {state: 0.0 for state in states}
    std = {state: 0.0 for state in states}
    for i, function in zip(other, generating):
        first = function.derivative()(1)
        second = function.derivative().derivative()(1)
        variance = second + first - first**2
        mean[states[i]] = float(first) * cadence
        std[states[i]] = math.sqrt(max(0.0, float(variance))) * cadence

    return {
        "targets": [states[i] for i in targets],
        "horizons_seconds": horizons,
        "probability_by_start": curves,
        "mean_seconds_by_start": mean,
        "std_seconds_by_start": std,
    }


def symbolic_sensitivity(a, stationary, states, mask):
    ring = PolynomialRing(QQ, "eps")
    eps = ring.fraction_field().gen()
    n = len(states)
    fundamental = (
        identity_matrix(QQ, n) - a + matrix(QQ, [list(stationary)] * n)
    ).inverse()
    results = []
    for i, source in enumerate(states):
        for j, destination in enumerate(states):
            if i == j or not mask[i][j]:
                continue
            direction = fundamental.row(j) - fundamental.row(i)
            factor = eps * stationary[i] / (1 - eps * direction[i])
            slopes = [
                float((stationary[k] + factor * direction[k]).derivative()(0))
                for k in range(n)
            ]
            results.append(
                {
                    "edge": [source, destination],
                    "delta_transition_probability": 0.0001,
                    "delta_stationary_probability": {
                        state: 0.0001 * slope for state, slope in zip(states, slopes)
                    },
                }
            )
    return results


def analyze(data: dict) -> dict:
    states = data["states"]
    mask = data["mask"]
    n = len(states)
    cadence = float(data["cadence_seconds"])
    a = matrix(RDF, data["transitions"])
    exact = rational_chain(data["transitions"])

    graph = DiGraph(matrix(QQ, mask), loops=True)
    topology = {
        "simple_cycles": [
            [states[i] for i in cycle]
            for cycle in graph.all_simple_cycles()
            if len(cycle) > 2
        ],
        "strongly_connected": bool(graph.is_strongly_connected()),
        "components": [
            [states[i] for i in comp] for comp in graph.strongly_connected_components()
        ],
        "allowed_edges": int(graph.size()),
        "out_degree": {
            state: int(graph.out_degree(i)) for i, state in enumerate(states)
        },
        "shortest_allowed_paths_steps": [
            [int(distance) for distance in row]
            for row in graph.distance_matrix(vertices=list(range(n))).rows()
        ],
    }

    trees = tree_weights(exact)
    stationary = vector(QQ, trees) / sum(trees)
    p = [float(x) for x in stationary]

    hitting = matrix(QQ, n, n)
    for target in range(n):
        other = [i for i in range(n) if i != target]
        q = exact.matrix_from_rows_and_columns(other, other)
        steps = (identity_matrix(QQ, len(other)) - q).solve_right(
            vector(QQ, [1] * len(other))
        )
        for i, t in zip(other, steps):
            hitting[i, target] = t
    kemeny = [sum(stationary[j] * hitting[i, j] for j in range(n)) for i in range(n)]

    flux = matrix(
        RDF, [[p[i] * a[i, j] * 3600 / cadence for j in range(n)] for i in range(n)]
    )
    flow = copy(flux)
    for i in range(n):
        flow[i, i] = 0
    currents = flux - flux.transpose()
    equilibrium = matrix(RDF, [p] * n)
    convergence = []
    for seconds in (0, 30, 60, 120, 300, 600, 1200, 2400):
        steps = int(round(seconds / cadence))
        difference = a**steps - equilibrium
        tv = 0.5 * max(sum(abs(v) for v in row) for row in difference.rows())
        convergence.append(
            {"seconds": steps * cadence, "worst_total_variation": float(tv)}
        )

    eig = sorted(
        (complex(v) for v in a.change_ring(CDF).eigenvalues()), key=abs, reverse=True
    )
    modulus = abs(eig[1]) if n > 1 else 0.0
    modes = []
    for eigenvalue, vectors, _ in sorted(
        a.change_ring(CDF).eigenvectors_left(),
        key=lambda item: abs(complex(item[0])),
        reverse=True,
    ):
        lam = complex(eigenvalue)
        if abs(lam - 1) < 1e-7 or lam.imag < -1e-8:
            continue
        amplitudes = [abs(complex(v)) for v in vectors[0]]
        scale = max(amplitudes)
        modes.append(
            {
                "eigenvalue": {"real": lam.real, "imag": lam.imag},
                "amplitude": dict(zip(states, [v / scale for v in amplitudes])),
                "timescale_seconds": -cadence / math.log(abs(lam))
                if 0 < abs(lam) < 1
                else None,
            }
        )
        if len(modes) == 3:
            break

    thermal = [states.index(s) for s in ("THERMAL_RESPONSE", "THERMAL_STEADY")]
    idle = states.index("IDLE")
    first_passage = {
        "thermal": passage_analysis(a, exact, states, thermal, cadence),
        "idle": passage_analysis(a, exact, states, [idle], cadence),
    }

    middle = [i for i in range(n) if i not in thermal and i != idle]
    q = exact.matrix_from_rows_and_columns(middle, middle)
    arrivals = exact.matrix_from_rows_and_columns(middle, thermal) * vector(
        QQ, [1] * len(thermal)
    )
    solved = (identity_matrix(QQ, len(middle)) - q).solve_right(arrivals)
    committor = {
        states[i]: 1.0
        if i in thermal
        else 0.0
        if i == idle
        else float(solved[middle.index(i)])
        for i in range(n)
    }

    entropy = {
        states[i]: -sum(float(v) * math.log(float(v)) for v in a.row(i) if v > 0)
        for i in range(n)
    }
    dwell = {
        states[i]: cadence / (1 - float(a[i, i])) if a[i, i] < 1 else None
        for i in range(n)
    }

    means, variances, features = data["means"], data["variances"], data["feature_names"]
    distances = [[0.0] * n for _ in range(n)]
    explanations = []
    for i in range(n):
        for j in range(i + 1, n):
            contributions = []
            for mi, mj, vi, vj in zip(means[i], means[j], variances[i], variances[j]):
                avg = (vi + vj) / 2
                contribution = (mi - mj) ** 2 / (8 * avg)
                contribution += 0.5 * math.log(avg / math.sqrt(vi * vj))
                contributions.append(max(0.0, contribution))
            total = sum(contributions)
            distances[i][j] = distances[j][i] = total
            if mask[i][j] or mask[j][i]:
                explanations.append(
                    {
                        "states": [states[i], states[j]],
                        "distance": total,
                        "contributions": dict(zip(features, contributions)),
                    }
                )

    return {
        "topology": topology,
        "markov_chain": {
            "stationary_probabilities": dict(zip(states, p)),
            "stationary_tree_weights": dict(zip(states, [float(t) for t in trees])),
            "characteristic_polynomial_coefficients_ascending": [
                float(c) for c in exact.charpoly().list()
            ],
            "expected_dwell_seconds": dwell,
            "expected_hitting_seconds": [
                [float(t) * cadence for t in row] for row in hitting.rows()
            ],
            "transition_entropy_nats": entropy,
            "entropy_rate_nats_per_step": sum(
                p[i] * entropy[states[i]] for i in range(n)
            ),
            "eigenvalues": [{"real": v.real, "imag": v.imag} for v in eig],
            "spectral_gap": 1 - modulus,
            "relaxation_seconds": -cadence / math.log(modulus)
            if 0 < modulus < 1
            else None,
            "kemeny_seconds": float(sum(kemeny) / n) * cadence,
            "kemeny_source_spread_seconds": float(max(kemeny) - min(kemeny)) * cadence,
            "stationary_offdiagonal_flow_per_hour": [
                [float(v) for v in row] for row in flow.rows()
            ],
            "stationary_entries_per_hour": {
                states[j]: float(sum(flow.column(j))) for j in range(n)
            },
            "worst_case_convergence": convergence,
            "slow_modes": modes,
            "first_passage": first_passage,
            "thermal_before_idle_committor": committor,
            "stationary_currents_per_hour": [
                [float(v) for v in row] for row in currents.rows()
            ],
            "stationary_sensitivity": symbolic_sensitivity(
                exact, stationary, states, mask
            ),
            "first_exit_probability_30s": {
                states[i]: 1 - float(a[i, i]) ** int(round(30 / cadence))
                for i in range(n)
            },
        },
        "emissions": {
            "bhattacharyya_distance": distances,
            "adjacent_pair_feature_contributions": explanations,
        },
    }


print(json.dumps(analyze(json.load(sys.stdin))))
