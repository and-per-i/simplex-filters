#!/usr/bin/env python3
"""
grassmann_baseline.py — Baseline Monte Carlo per varianza geodesica su Gr(2,128).

Genera N_planes piani random sulla Grassmanniana Gr(2, dim),
calcola varianza geodesica via Frechet mean, ripete per stima stabile.

Usage:
    python scripts/grassmann_baseline.py                     # default: d=128, N=320, runs=10
    python scripts/grassmann_baseline.py --dim 64 --runs 20
"""

import argparse
import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np

from src.geometry.grassmann import frechet_mean_planes, geodesic_variance


def random_planes(num_planes: int, dim: int = 128, device: str = "cpu") -> torch.Tensor:
    """
    Genera num_planes piani casuali su Gr(2, dim) campionando vettori Gaussiani
    e ortogonalizzando con QR.

    Args:
        num_planes: numero di piani
        dim: dimensione dello spazio ambiente
        device: "cpu" o "cuda"

    Returns:
        U_list: basi ortonormali [num_planes, dim, 2]
    """
    U_list = torch.zeros(num_planes, dim, 2, device=device)
    
    for i in range(num_planes):
        # Campiona 2 vettori Gaussiani in R^dim
        A = torch.randn(dim, 2, device=device)
        # QR decomposition → base ortonormale
        Q, R = torch.linalg.qr(A)
        # Q ha colonne ortogonali (con segno casuale)
        U_list[i] = Q[:, :2]
    
    return U_list


def main():
    parser = argparse.ArgumentParser(
        description="Monte Carlo baseline per varianza geodesica su Gr(2,d)"
    )
    parser.add_argument("--dim", type=int, default=64,
                        help="Dimensione dello spazio ambiente (default: 64)")
    parser.add_argument("--n-planes", type=int, default=320,
                        help="Numero di piani per run (default: 320)")
    parser.add_argument("--runs", type=int, default=10,
                        help="Numero di run Monte Carlo (default: 10)")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"],
                        help="Device per il calcolo (default: cpu)")
    args = parser.parse_args()

    print(f"\n{'='*65}")
    print(f"  BASELINE MONTE CARLO — Gr({args.dim}, 2)")
    print(f"  Piani per run: {args.n_planes}")
    print(f"  Run: {args.runs}")
    print(f"  Device: {args.device}")
    print(f"{'='*65}\n")

    all_variances = []
    all_distances = []

    for run in range(args.runs):
        print(f"  Run {run+1}/{args.runs}...", end=" ", flush=True)

        # 1. Genera piani random
        U_list = random_planes(args.n_planes, args.dim, args.device)

        # 2. Media di Frechet (10 iterazioni)
        U_mean, P_mean = frechet_mean_planes(U_list, n_iter=10)

        # 3. Varianza geodesica
        var_g, distances = geodesic_variance(U_list, U_mean)

        all_variances.append(var_g.item())
        all_distances.append(distances.cpu())

        print(f"var={var_g.item():.4f}")

    # Statistiche aggregate
    all_variances = np.array(all_variances)
    all_dists = torch.cat(all_distances, dim=0).numpy()

    mean_var = all_variances.mean()
    std_var = all_variances.std()
    ci95 = 1.96 * std_var / math.sqrt(args.runs)

    mean_dist = all_dists.mean()
    std_dist = all_dists.std()

    print(f"\n{'='*65}")
    print(f"  RISULTATI")
    print(f"{'='*65}")
    print(f"  Varianza geodesica media:     {mean_var:.4f} ± {std_var:.4f}")
    print(f"  IC 95%%:                       [{mean_var - ci95:.4f}, {mean_var + ci95:.4f}]")
    print(f"  Distanza geodesica media:     {mean_dist:.4f} ± {std_dist:.4f}")
    print(f"  Distanza massima teorica:     {math.pi * math.sqrt(2) / 2:.4f}")
    print(f"{'='*65}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())