import sys
import time
import numpy as np
from typing import List, Tuple


def read_matrix_market_coo(path: str) -> Tuple[int, int, int, List[Tuple[int, int, float]]]:
    """
    Lit un fichier .mtx (Matrix Market) au format coordinate.

    - Ignore les lignes commençant par '%'
    - Attend une ligne : nRows nCols nnz
    - Puis des lignes : i j [val]
      * indices 1-based -> convertis en 0-based
      * si val absent -> val = 1.0
    """
    entries: List[Tuple[int, int, float]] = []
    nrows = ncols = nnz = -1

    with open(path, "r", encoding="utf-8") as f:
        # Lire la ligne des dimensions
        for line in f:
            s = line.strip()
            if not s or s.startswith("%"):
                continue
            parts = s.split()
            if len(parts) < 3:
                raise ValueError(f"Ligne dimensions invalide: {s}")
            nrows, ncols, nnz = map(int, parts[:3])
            break

        if nrows <= 0 or ncols <= 0 or nnz < 0:
            raise ValueError("Impossible de lire les dimensions du fichier .mtx")

        # Lire les entrées non nulles
        for line in f:
            s = line.strip()
            if not s or s.startswith("%"):
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            i = int(parts[0]) - 1
            j = int(parts[1]) - 1
            val = float(parts[2]) if len(parts) >= 3 else 1.0
            entries.append((i, j, val))

    return nrows, ncols, nnz, entries

def read_txt(path: str) -> Tuple[int, int, int, list[tuple[int, int, float]]]:
    entries: List[Tuple[int, int, float]] = []

    with open(path, "r", encoding="utf-8") as f:
        first = f.readline().strip().split()

        if len(first) == 1:
            # G1000 format: n seul sur la première ligne, nnz sur la deuxième
            n = int(first[0])
            nnz_header = int(f.readline().strip())
        elif len(first) == 2:
            # Stanford format: n et nnz sur la même ligne
            n = int(first[0])
            nnz_header = int(first[1])
        else:
            raise ValueError(f"Première ligne invalide: {' '.join(first)}")

        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            
            i = int(parts[0]) - 1
            deg = int(parts[1])
            excepted_len = 2 + 2*deg
            if len(parts) < excepted_len:
                raise ValueError(f"Ligne invalide pour le sommet {i+1}")
            for k in range(deg):
                j = int(parts[2 + 2*k]) - 1
                val = float(parts[3 + 2*k])
                entries.append((i, j, val))
    return n, n, nnz_header, entries

def read_file(path: str) -> Tuple[int, int, int, list[tuple[int, int, float]]]:
    if path.endswith(".mtx"):
        return read_matrix_market_coo(path)
    elif path.endswith(".txt"):
        return read_txt(path)
    else:
        raise ValueError(f"Format de fichier non supporté: {path}")
            
def build_out_lists(
    n: int,
    entries: List[Tuple[int, int, float]],
    treat_as_unweighted: bool = True,
    transpose: bool = False,
) -> Tuple[List[List[int]], np.ndarray]:
    """
    Construit une représentation creuse :
    - out[i] = liste des voisins j tels que i -> j
    - deg[i] = degré sortant d⁺(i)
    """
    out: List[List[int]] = [[] for _ in range(n)]
    deg = np.zeros(n, dtype=float)

    for i, j, val in entries:
        if not (0 <= i < n and 0 <= j < n):
            continue

        if transpose:
            i, j = j, i

        out[i].append(j)
        deg[i] += 1.0 if treat_as_unweighted else float(val)

    return out, deg


def pagerank_google_sparse(
    out: List[List[int]],
    deg: np.ndarray,
    alpha: float = 0.85,
    eps: float = 1e-6,
    max_iter: int = 10000,
) -> Tuple[np.ndarray, int, float]:
    """
    Implémentation du cours 3 :

        x(0)   = (1/N)e
        x(k+1) = α x(k)P + α(1/N)(x(k)f^t)e + (1-α)(1/N)e

    où :
      - P est stockée en creux via out et deg
      - f[i] = 1 si deg[i] = 0, sinon 0

    Affichage à chaque itération :
        norme k = ...
    Retour :
        (vecteur final, nombre d'itérations, temps total)
    """
    n = len(out)
    e_uniform = np.ones(n, dtype=float) / n
    x = e_uniform.copy()                   # x(0)
    f = (deg == 0.0).astype(float)         # vecteur f

    start_time = time.time()

    for k in range(1, max_iter + 1):
        y = np.zeros(n, dtype=float)

        # 1) On initialise y avec (1 - alpha) / N
        y += (1.0 - alpha) * e_uniform

        # 2) On calcule xf, puis on ajoute alpha * (xf / N) à toutes les composantes
        xf = float(np.dot(x, f))
        y += alpha * xf * e_uniform

        # 3) On calcule alpha * xP en creux et on l'ajoute à y
        for i in range(n):
            if deg[i] == 0.0:
                continue
            contrib = alpha * x[i] / deg[i]
            for j in out[i]:
                y[j] += contrib

        # Normalisation de sécurité
        s = y.sum()
        if s != 0.0:
            y /= s

        # Norme L1
        norme = np.sum(np.abs(y - x))

        # Affichage comme demandé
        print(f"norme {k} = {norme:.10f}")

        if norme <= eps:
            elapsed_time = time.time() - start_time
            return y, k, elapsed_time

        x = y

    elapsed_time = time.time() - start_time
    return x, max_iter, elapsed_time


def main():
    # ===== PARAMÈTRES =====
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <fichier.mtx>")
        sys.exit(1)
    path = sys.argv[1]
    alpha = 0.85
    eps = 1e-6
    max_iter = 10000

    # ===== LECTURE DU FICHIER =====
    nrows, ncols, nnz_header, entries = read_file(path)
    if nrows != ncols:
        raise ValueError(f"Matrice non carrée: {nrows}x{ncols} (PageRank exige NxN)")

    N = nrows

    # ===== CONSTRUCTION DE LA STRUCTURE CREUSE =====
    # transpose=False : on suppose que chaque ligne 'i j' signifie arc i -> j
    out, deg = build_out_lists(
        N,
        entries,
        treat_as_unweighted=True,
        transpose=False
    )

    # ===== CALCUL PAGERANK =====
    pi, iters, elapsed_time = pagerank_google_sparse(
        out,
        deg,
        alpha=alpha,
        eps=eps,
        max_iter=max_iter
    )

    # ===== AFFICHAGE FINAL =====
    print("\n--- Résultat final ---")
    print("N =", N)
    print("nnz(en-tête) =", nnz_header)
    print("nnz(lu) =", len(entries))
    print("alpha =", alpha)
    print("Nombre d'itérations =", iters)
    print(f"Temps total d'itération = {elapsed_time:.4f} secondes")
    print("Somme des composantes =", pi.sum())

    topk = 10
    order = np.argsort(-pi)[:topk]
    print(f"\nTop {topk} sommets (index 1-based) :")
    for r, idx in enumerate(order, start=1):
        print(f"{r:2d}) sommet {idx+1:4d}  score = {pi[idx]:.8f}")


if __name__ == "__main__":
    main()