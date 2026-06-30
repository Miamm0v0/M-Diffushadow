"""
Generate multimodal XXZ training data for train_multi.py.

The script writes two JSON files with the layouts expected by the current
multimodal model:

    single/seq modality:
        [Delta, P1, ..., PN, b1, ..., bN]

    pair/C modality:
        [Delta, r1_1, r2_1, c1, ..., r1_N, r2_N, cN]

Pauli tokens follow the existing project convention:

    2 = X, 3 = Y, 4 = Z

Outcomes are encoded as:

    0 = -1, 1 = +1

For the pair/C modality, item i is the nearest-neighbor periodic bond
(i, i+1).  By default, pair bases are diagonal XX/YY/ZZ, matching the XXZ
Hamiltonian terms.  Use --pair-basis-mode full to sample all 9 Pauli products.

Example:

    python generate_xxz_multimodal_dataset.py \
        --num-qubits 10 \
        --samples-per-delta 10000 \
        --deltas -2 -1.5 -1 -0.5 0 0.5 1 1.5 2 \
        --seq-json-out data/xxz_train_seq.json \
        --pair-json-out data/xxz_train_phys.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
from itertools import product
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm


PAULI_TOKENS = (2, 3, 4)  # X, Y, Z in the data files.
DTYPE = torch.complex64
REAL_DTYPE = torch.float64

H_MEASURE = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=DTYPE) / math.sqrt(2.0)
Y_MATRIX = torch.tensor([[0.0, -1.0j], [1.0j, 0.0]], dtype=DTYPE)
_, Y_EVECS = torch.linalg.eigh(Y_MATRIX)
Y_MEASURE = Y_EVECS[:, [1, 0]].contiguous()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate multimodal XXZ training data.")
    parser.add_argument("--num-qubits", type=int, default=10)
    parser.add_argument("--samples-per-delta", type=int, default=10000)
    parser.add_argument("--j-xy", type=float, default=1.0)

    parser.add_argument("--deltas", type=float, nargs="+", default=None)
    parser.add_argument("--delta-min", type=float, default=-2.0)
    parser.add_argument("--delta-max", type=float, default=2.0)
    parser.add_argument("--delta-length", type=int, default=9)

    parser.add_argument(
        "--pair-basis-mode",
        choices=["diagonal", "full"],
        default="diagonal",
        help="diagonal samples XX/YY/ZZ only; full samples all 9 Pauli products.",
    )
    parser.add_argument("--seq-json-out", type=str, required=True)
    parser.add_argument("--pair-json-out", type=str, required=True)
    parser.add_argument("--seq-pt-out", type=str, default="")
    parser.add_argument("--pair-pt-out", type=str, default="")
    parser.add_argument("--meta-json-out", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def get_delta_values(args: argparse.Namespace) -> list[float]:
    if args.deltas is not None:
        return [float(value) for value in args.deltas]
    return np.linspace(args.delta_min, args.delta_max, args.delta_length).astype(float).tolist()


def site_bit_mask(num_qubits: int, site: int) -> int:
    return 1 << (num_qubits - 1 - (site % num_qubits))


def build_xxz_hamiltonian_pbc(
    num_qubits: int,
    j_xy: float,
    delta: float,
    device: torch.device,
) -> torch.Tensor:
    """Dense XXZ Hamiltonian in the Pauli convention used by eval_utils.py."""
    if num_qubits < 2:
        raise ValueError("--num-qubits must be at least 2")

    dim = 1 << num_qubits
    ham = torch.zeros((dim, dim), dtype=REAL_DTYPE, device=device)

    for basis in range(dim):
        for site in range(num_qubits):
            next_site = (site + 1) % num_qubits
            mask_i = site_bit_mask(num_qubits, site)
            mask_j = site_bit_mask(num_qubits, next_site)

            bit_i = 1 if (basis & mask_i) else 0
            bit_j = 1 if (basis & mask_j) else 0
            z_i = -1.0 if bit_i else 1.0
            z_j = -1.0 if bit_j else 1.0
            zz_value = z_i * z_j
            flipped = basis ^ mask_i ^ mask_j

            # X_i X_j flips both bits.
            ham[flipped, basis] += j_xy

            # Y_i Y_j flips both bits with the Pauli-Y phase product.
            ham[flipped, basis] += j_xy * (-zz_value)

            # Z_i Z_j is diagonal.
            ham[basis, basis] += j_xy * delta * zz_value

    return ham


def ground_state_from_hamiltonian(ham: torch.Tensor) -> torch.Tensor:
    ham = (ham + ham.T.conj()) / 2
    _eigenvalues, eigenvectors = torch.linalg.eigh(ham)
    psi = eigenvectors[:, 0].to(dtype=DTYPE).contiguous()
    return psi / psi.norm()


def sample_single_shadow_nseq(psi: torch.Tensor, num_qubits: int) -> tuple[list[int], list[int]]:
    """Sequentially sample one single-site Pauli shadow in nseq order."""
    h_measure = H_MEASURE.to(device=psi.device)
    y_measure = Y_MEASURE.to(device=psi.device)

    bases: list[int] = []
    outcomes: list[int] = []
    psi_work = psi.reshape(-1).to(dtype=DTYPE).clone()

    for _site in range(num_qubits):
        pauli_token = int(torch.randint(2, 5, (1,), device=psi.device).item())
        measure_basis = pauli_token - 2  # 0: X, 1: Y, 2: Z

        left = psi_work.numel() // 2
        matrix = psi_work.view(2, left)
        if measure_basis == 0:
            matrix = h_measure @ matrix
        elif measure_basis == 1:
            matrix = y_measure.conj().T @ matrix

        positive_prob = (matrix[0].conj() * matrix[0]).real.sum().clamp(0.0, 1.0)
        outcome = int(torch.bernoulli(positive_prob).item())

        branch = matrix[0] if outcome == 1 else matrix[1]
        psi_work = (branch / (branch.norm() + 1e-30)).contiguous()

        bases.append(pauli_token)
        outcomes.append(outcome)

    return bases, outcomes


def apply_two_pauli_product(
    psi: torch.Tensor,
    num_qubits: int,
    site_i: int,
    token_i: int,
    site_j: int,
    token_j: int,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Apply sigma_i^token_i sigma_j^token_j to psi without building the matrix."""
    target = indices.clone()
    phase = torch.ones(psi.numel(), dtype=psi.dtype, device=psi.device)

    for site, token in ((site_i, token_i), (site_j, token_j)):
        bit_mask = site_bit_mask(num_qubits, site)
        bits_are_one = (indices & bit_mask) != 0

        if token == 2:  # X
            target = torch.bitwise_xor(target, bit_mask)
        elif token == 3:  # Y
            target = torch.bitwise_xor(target, bit_mask)
            phase = phase * torch.where(
                bits_are_one,
                torch.tensor(-1.0j, dtype=psi.dtype, device=psi.device),
                torch.tensor(1.0j, dtype=psi.dtype, device=psi.device),
            )
        elif token == 4:  # Z
            phase = phase * torch.where(
                bits_are_one,
                torch.tensor(-1.0, dtype=psi.dtype, device=psi.device),
                torch.tensor(1.0, dtype=psi.dtype, device=psi.device),
            )
        else:
            raise ValueError(f"Unsupported Pauli token: {token}")

    result = torch.empty_like(psi)
    result[target] = phase * psi
    return result


def pauli_product_expectation(
    psi: torch.Tensor,
    num_qubits: int,
    site_i: int,
    token_i: int,
    site_j: int,
    token_j: int,
    indices: torch.Tensor,
) -> float:
    operated = apply_two_pauli_product(
        psi=psi,
        num_qubits=num_qubits,
        site_i=site_i,
        token_i=token_i,
        site_j=site_j,
        token_j=token_j,
        indices=indices,
    )
    value = torch.vdot(psi, operated).real.item()
    return float(max(-1.0, min(1.0, value)))


def pair_basis_choices(mode: str) -> list[tuple[int, int]]:
    if mode == "diagonal":
        return [(token, token) for token in PAULI_TOKENS]
    if mode == "full":
        return list(product(PAULI_TOKENS, PAULI_TOKENS))
    raise ValueError(f"Unknown pair basis mode: {mode}")


def precompute_pair_product_probs(
    psi: torch.Tensor,
    num_qubits: int,
    mode: str,
) -> dict[tuple[int, int, int], float]:
    dim = psi.numel()
    indices = torch.arange(dim, dtype=torch.long, device=psi.device)
    choices = pair_basis_choices(mode)
    probs: dict[tuple[int, int, int], float] = {}

    for site in range(num_qubits):
        next_site = (site + 1) % num_qubits
        for token_i, token_j in choices:
            expectation = pauli_product_expectation(
                psi=psi,
                num_qubits=num_qubits,
                site_i=site,
                token_i=token_i,
                site_j=next_site,
                token_j=token_j,
                indices=indices,
            )
            probs[(site, token_i, token_j)] = 0.5 * (1.0 + expectation)

    return probs


Number = Union[float, int]


def sample_pair_product_row(
    delta: float,
    num_qubits: int,
    mode: str,
    plus_probs: dict[tuple[int, int, int], float],
    device: torch.device,
) -> List[Number]:
    row: List[Number] = [float(delta)]
    choices = pair_basis_choices(mode)

    for site in range(num_qubits):
        token_i, token_j = random.choice(choices)
        plus_prob = plus_probs[(site, token_i, token_j)]
        outcome = int((torch.rand((), device=device).item()) < plus_prob)
        row.extend([token_i, token_j, outcome])

    return row


def generate_for_deltas(
    *,
    num_qubits: int,
    j_xy: float,
    deltas: list[float],
    samples_per_delta: int,
    pair_basis_mode: str,
    device: torch.device,
) -> Tuple[List[List[Number]], List[List[Number]]]:
    seq_rows: List[List[Number]] = []
    pair_rows: List[List[Number]] = []

    for delta in tqdm(deltas, desc="XXZ Delta values", ncols=100):
        ham = build_xxz_hamiltonian_pbc(
            num_qubits=num_qubits,
            j_xy=j_xy,
            delta=float(delta),
            device=device,
        )
        psi = ground_state_from_hamiltonian(ham)
        del ham

        plus_probs = precompute_pair_product_probs(
            psi=psi,
            num_qubits=num_qubits,
            mode=pair_basis_mode,
        )

        sample_iter = tqdm(
            range(samples_per_delta),
            desc=f"Delta={float(delta):.6g}",
            leave=False,
            ncols=100,
        )
        for _ in sample_iter:
            bases, outcomes = sample_single_shadow_nseq(psi, num_qubits)
            seq_rows.append([float(delta), *bases, *outcomes])
            pair_rows.append(
                sample_pair_product_row(
                    delta=float(delta),
                    num_qubits=num_qubits,
                    mode=pair_basis_mode,
                    plus_probs=plus_probs,
                    device=device,
                )
            )

    return seq_rows, pair_rows


def ensure_parent_dir(path: str) -> None:
    parent = Path(path).parent
    if str(parent):
        parent.mkdir(parents=True, exist_ok=True)


def save_json(rows: List[List[Number]], path: str) -> None:
    ensure_parent_dir(path)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(rows, f)
    print(f"saved json: {path}")


def save_pt(rows: List[List[Number]], path: str) -> None:
    ensure_parent_dir(path)
    torch.save(torch.tensor(rows, dtype=torch.float32), path)
    print(f"saved pt: {path}")


def save_meta(args: argparse.Namespace, deltas: list[float], path: str) -> None:
    if not path:
        return
    ensure_parent_dir(path)
    meta = {
        "model": "XXZ",
        "num_qubits": args.num_qubits,
        "J_xy": args.j_xy,
        "deltas": deltas,
        "samples_per_delta": args.samples_per_delta,
        "pair_basis_mode": args.pair_basis_mode,
        "single_format": "[Delta, P1, ..., PN, b1, ..., bN]",
        "pair_format": "[Delta, r1_1, r2_1, c1, ..., r1_N, r2_N, cN]",
        "pauli_tokens": {"X": 2, "Y": 3, "Z": 4},
        "outcome_tokens": {"minus_one": 0, "plus_one": 1},
        "pair_site_rule": "bond i is (i, i+1) with periodic boundary conditions",
    }
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"saved meta: {path}")


def main() -> None:
    args = parse_args()
    if args.samples_per_delta <= 0:
        raise ValueError("--samples-per-delta must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    deltas = get_delta_values(args)

    print("XXZ multimodal generation arguments:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print(f"  resolved_deltas: {deltas}")
    if args.num_qubits != 10:
        print("Warning: Qmultimodel.py/train_multi.py are currently hard-coded for 10 qubits.")
    if args.samples_per_delta != 10000:
        print("Warning: train_multi.py groups training rows in fixed blocks of 10000 samples.")

    seq_rows, pair_rows = generate_for_deltas(
        num_qubits=args.num_qubits,
        j_xy=args.j_xy,
        deltas=deltas,
        samples_per_delta=args.samples_per_delta,
        pair_basis_mode=args.pair_basis_mode,
        device=device,
    )

    expected_seq_width = 1 + 2 * args.num_qubits
    expected_pair_width = 1 + 3 * args.num_qubits
    if seq_rows and len(seq_rows[0]) != expected_seq_width:
        raise RuntimeError(f"Expected seq width {expected_seq_width}, got {len(seq_rows[0])}")
    if pair_rows and len(pair_rows[0]) != expected_pair_width:
        raise RuntimeError(f"Expected pair width {expected_pair_width}, got {len(pair_rows[0])}")
    if len(seq_rows) != len(pair_rows):
        raise RuntimeError("single and pair row counts differ")

    save_json(seq_rows, args.seq_json_out)
    save_json(pair_rows, args.pair_json_out)
    if args.seq_pt_out:
        save_pt(seq_rows, args.seq_pt_out)
    if args.pair_pt_out:
        save_pt(pair_rows, args.pair_pt_out)
    save_meta(args, deltas, args.meta_json_out)

    print(f"dataset rows: {len(seq_rows)}")
    print(f"single width: {expected_seq_width}")
    print(f"pair width: {expected_pair_width}")
    print("Done.")


if __name__ == "__main__":
    main()
