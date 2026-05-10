import sys
import numpy as np
from pagerank import read_matrix_market_coo, build_out_lists, pagerank_google_sparse


def select_targets(pi: np.ndarray):
    n = len(pi)
    order = np.argsort(-pi)

    high_idx = max(1, n // 100)
    low_idx  = max(1, n // 100)

    high   = order[high_idx]
    medium = order[n // 2]
    low    = order[-low_idx]

    return high, medium, low


def add_complete_graph(out, deg, n_attackers, target):
    out = [list(row) for row in out]
    deg = deg.copy()

    N = len(out)
    new_nodes = list(range(N, N + n_attackers))

    for i in new_nodes:
        neighbors = [j for j in new_nodes if j != i] + [target]
        out.append(neighbors)
        deg = np.append(deg, float(len(neighbors)))

    return out, deg


def add_ring(out, deg, n_attackers, target):
    out = [list(row) for row in out]
    deg = deg.copy()

    N = len(out)
    new_nodes = list(range(N, N + n_attackers))

    for k in range(len(new_nodes)):
        if k < len(new_nodes) - 1:
            out.append([new_nodes[k + 1]])
        else:
            out.append([target])
        deg = np.append(deg, 1.0)

    if new_nodes:
        out[target].append(new_nodes[0])
        deg[target] += 1.0

    return out, deg


def add_isolated_attackers(out, deg, n_attackers, target):
    out = [list(row) for row in out]
    deg = deg.copy()

    for _ in range(n_attackers):
        out.append([target])
        deg = np.append(deg, 1.0)

    return out, deg


def run_attack(out_base, deg_base, attack_fn, n_attackers, target, alpha, eps, max_iter):
    out_att, deg_att = attack_fn(out_base, deg_base, n_attackers, target)
    pi_att, iters, _ = pagerank_google_sparse(
        out_att, deg_att, alpha=alpha, eps=eps, max_iter=max_iter, verbose=False
    )
    return pi_att[target], iters


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <fichier.mtx>")
        sys.exit(1)

    path = sys.argv[1]
    alpha = 0.85
    eps = 1e-6
    max_iter = 10000

    nrows, ncols, _, entries = read_matrix_market_coo(path)
    if nrows != ncols:
        raise ValueError(f"Matrice non carrée: {nrows}x{ncols}")
    N = nrows

    out_base, deg_base = build_out_lists(N, entries, treat_as_unweighted=True, transpose=False)

    print("Calcul du PageRank de base...")
    pi_base, iters_base, _ = pagerank_google_sparse(
        out_base, deg_base, alpha=alpha, eps=eps, max_iter=max_iter, verbose=False
    )
    print(f"Convergé en {iters_base} itérations\n")

    high, medium, low = select_targets(pi_base)
    order_base = np.argsort(-pi_base)

    targets = {
        "forte":   (high,   pi_base[high]),
        "moyenne": (medium, pi_base[medium]),
        "faible":  (low,    pi_base[low]),
    }

    print("Cibles sélectionnées :")
    for label, (idx, score) in targets.items():
        rang = int(np.where(order_base == idx)[0][0]) + 1
        print(f"  {label:8s}: sommet {idx+1:6d}  score={score:.8f}  rang={rang}/{N}")
    print()

    attack_fns = {
        "complet": add_complete_graph,
        "anneau":  add_ring,
        "isolés":  add_isolated_attackers,
    }

    attacker_sizes = [1, 5, 10, 50, 100]

    # results[target_label][atk_name] = liste de (n, delta)
    results = {t: {a: [] for a in attack_fns} for t in targets}

    for target_label, (target_idx, base_score) in targets.items():
        print(f"=== Cible {target_label} (sommet {target_idx+1}, score de base={base_score:.8f}) ===")
        for atk_name, atk_fn in attack_fns.items():
            print(f"  [{atk_name}]")
            for n_att in attacker_sizes:
                score_att, iters_att = run_attack(
                    out_base, deg_base, atk_fn, n_att, target_idx, alpha, eps, max_iter
                )
                delta = score_att - base_score
                results[target_label][atk_name].append((n_att, delta))
                print(f"    n={n_att:4d}  Δ={delta:+.8f}  iters={iters_att}")
        print()

    # ===== RÉSUMÉ =====
    print("=" * 60)
    print("RÉSUMÉ : meilleure attaque par cible")
    print("=" * 60)
    print(f"{'Cible':<10} {'Structure':<10} {'n':>5}  {'Δ max':>12}")
    print("-" * 60)
    for target_label in targets:
        best_atk, best_n, best_delta = None, None, -np.inf
        for atk_name, entries in results[target_label].items():
            for n_att, delta in entries:
                if delta > best_delta:
                    best_delta = delta
                    best_atk = atk_name
                    best_n = n_att
        print(f"{target_label:<10} {best_atk:<10} {best_n:>5}  {best_delta:>+12.8f}")
    print()

    print("=" * 60)
    print("RÉSUMÉ : Δ par structure et par n (toutes cibles confondues)")
    print("=" * 60)
    header = f"{'':>10}" + "".join(f"  n={n:>4}" for n in attacker_sizes)
    print(header)
    print("-" * 60)
    for atk_name in attack_fns:
        for target_label in targets:
            deltas = [d for _, d in results[target_label][atk_name]]
            row = f"{atk_name+'/'+target_label:<10}" + "".join(f"  {d:>+8.2e}" for d in deltas)
            print(row)
        print()


if __name__ == "__main__":
    main()
