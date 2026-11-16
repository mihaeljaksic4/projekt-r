import json
import math
import sys

from sage.all import CDF, RDF, DiGraph, identity_matrix, matrix, vector


def analyze(data: dict) -> dict:
    states = data["states"]
    mask = data["mask"]
    means = data["means"]
    variances = data["variances"]
    n = len(states)
    a = matrix(RDF, data["transitions"])

    # Structural properties come from the allowed-transition mask, not from the
    # fitted probabilities, which may be near zero for unseen transitions.
    graph = DiGraph(
        {i: [j for j in range(n) if mask[i][j]] for i in range(n)},
        format="dict_of_lists",
        loops=True,
    )

    # Solve (A^T - I) pi = 0 subject to sum(pi) = 1.
    system = a.transpose() - identity_matrix(RDF, n)
    for j in range(n):
        system[n - 1, j] = 1.0
    stationary = system.solve_right(vector(RDF, [0] * (n - 1) + [1]))
    p = [max(0.0, float(v)) for v in stationary]
    total = sum(p)
    p = [v / total for v in p]

    eigenvalues = sorted(
        (complex(v) for v in a.change_ring(CDF).eigenvalues()),
        key=lambda v: abs(v),
        reverse=True,
    )
    second_modulus = abs(eigenvalues[1]) if n > 1 else 0.0

    # Expected first passage times E_i[time to enter j], with j fixed.
    hitting = [[0.0] * n for _ in range(n)]
    for target in range(n):
        other = [i for i in range(n) if i != target]
        q = matrix(RDF, [[a[i, j] for j in other] for i in other])
        h = (identity_matrix(RDF, n - 1) - q).solve_right(vector(RDF, [1] * (n - 1)))
        for i, value in zip(other, h):
            hitting[i][target] = max(0.0, float(value))

    # Pairwise Bhattacharyya distances between *full-dimensional* diagonal
    # Gaussian emissions, not just the 2D slices used in the 3D plots.
    distances = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            distance = 0.0
            for d in range(len(means[i])):
                vi, vj = variances[i][d], variances[j][d]
                avg = (vi + vj) / 2.0
                distance += (means[i][d] - means[j][d]) ** 2 / (8.0 * avg)
                distance += 0.5 * math.log(avg / math.sqrt(vi * vj))
            distances[i][j] = distances[j][i] = distance

    entropy = {
        states[i]: -sum(
            float(a[i, j]) * math.log(float(a[i, j])) for j in range(n) if a[i, j] > 0
        )
        for i in range(n)
    }
    dwell = {
        states[i]: (1.0 / (1.0 - float(a[i, i])) if float(a[i, i]) < 1 else None)
        for i in range(n)
    }
    return {
        "topology": {
            "strongly_connected": bool(graph.is_strongly_connected()),
            "components": [
                [states[i] for i in component]
                for component in graph.strongly_connected_components()
            ],
            "allowed_edges": int(graph.size()),
            "out_degree": {states[i]: int(graph.out_degree(i)) for i in range(n)},
        },
        "markov_chain": {
            "stationary_probabilities": dict(zip(states, p)),
            "expected_dwell_seconds": dwell,
            "expected_hitting_seconds": hitting,
            "transition_entropy_nats": entropy,
            "eigenvalues": [{"real": v.real, "imag": v.imag} for v in eigenvalues],
            "spectral_gap": 1.0 - second_modulus,
        },
        "emissions": {"bhattacharyya_distance": distances},
    }


print(json.dumps(analyze(json.load(sys.stdin))))
