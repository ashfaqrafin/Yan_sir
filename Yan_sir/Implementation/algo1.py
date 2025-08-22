import json
import numpy as np

def load_data(path):
    with open(path, 'r') as f:
        return json.load(f)

def normalize_rows(M):
    """
    Row-normalize matrix M so each row sums to 1 (if its sum > 0).
    """
    row_sums = M.sum(axis=1, keepdims=True)
    nonzero = row_sums[:, 0] > 0
    M[nonzero] = M[nonzero] / row_sums[nonzero]
    return M

def power_iter_with_const_norm(M, R_init, eps=1e-6, max_iter=1000):
    """
    Power iteration that preserves the norm of R between steps.
    """
    R = R_init.copy()
    norm_prev = np.linalg.norm(R)
    for _ in range(max_iter):
        R_next = M.dot(R)
        norm_next = np.linalg.norm(R_next)
        # adjust so ||R_next|| == norm_prev
        d = norm_next - norm_prev
        R_next = R_next - d * R
        delta = np.linalg.norm(R_next - R)
        if delta < eps:
            return R_next
        R, norm_prev = R_next, np.linalg.norm(R_next)
    return R

def incremental_ranking(data, alpha=0.6, beta=0.4, eps=1e-6):
    # Seed vector
    R_prev = np.array(data['seed'], dtype=float)
    R_prev = R_prev / np.linalg.norm(R_prev)

    # Process each year in order
    for year in sorted(data['years'], key=int):
        entry = data['years'][year]
        C = np.array(entry['C'], dtype=float)
        K = np.array(entry['K'], dtype=float)

        # row-normalize citation and collaboration matrices
        C = normalize_rows(C)
        K = normalize_rows(K)

        # build vote matrix
        M = alpha * C + beta * K

        # run power iteration seeded from last year's result
        R_new = power_iter_with_const_norm(M, R_prev, eps=eps)

        # print intermediate results
        print(f"Year {year} ranking:", R_new.round(4))

        R_prev = R_new

    return R_prev

if __name__ == "__main__":
    data = load_data("matrix_data1.json")
    final = incremental_ranking(data)
    print("Final scores:", np.round(final, 4))