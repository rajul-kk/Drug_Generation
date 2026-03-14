import argparse
import os
import sys
from typing import List, Optional, Set, Tuple

import numpy as np
from rdkit import DataStructs
from rdkit.Chem import AllChem

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.chemistry import canonicalize_smiles, is_valid_molecule, smiles_to_mol
from core.scoring import get_scorer


def parse_topk(topk_arg: str) -> List[int]:
    values = []
    for part in topk_arg.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    return sorted(set(v for v in values if v > 0))


def parse_line(line: str, delimiter: Optional[str]) -> Tuple[Optional[str], List[str]]:
    line = line.strip()
    if not line:
        return None, []
    if delimiter:
        cols = [c.strip() for c in line.split(delimiter)]
    else:
        cols = line.replace(",", " ").split()
    if not cols:
        return None, []
    return cols[0], cols


def load_reference(path: str, delimiter: Optional[str]) -> Set[str]:
    reference: Set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            smiles, _ = parse_line(line, delimiter)
            if not smiles:
                continue
            can = canonicalize_smiles(smiles)
            if can:
                reference.add(can)
    return reference


def internal_diversity(canonical_smiles: List[str], max_pairs: int = 50000) -> float:
    mols = [smiles_to_mol(s) for s in canonical_smiles]
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048) for m in mols if m is not None]
    n = len(fps)
    if n < 2:
        return 0.0

    total_possible_pairs = (n * (n - 1)) // 2
    step = 1
    if total_possible_pairs > max_pairs:
        step = max(1, total_possible_pairs // max_pairs)

    total_similarity = 0.0
    pair_count = 0
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if idx % step == 0:
                total_similarity += DataStructs.TanimotoSimilarity(fps[i], fps[j])
                pair_count += 1
            idx += 1

    if pair_count == 0:
        return 0.0

    return float(max(0.0, 1.0 - (total_similarity / pair_count)))


def format_table(rows: List[Tuple[str, str]]) -> str:
    key_width = max(len(k) for k, _ in rows)
    lines = []
    for key, value in rows:
        lines.append(f"{key.ljust(key_width)} : {value}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute GuacaMol-style summary metrics from generated SMILES"
    )
    parser.add_argument("--input", required=True, help="Input file with generated SMILES")
    parser.add_argument("--reference-file", default=None, help="Optional reference SMILES file for novelty")
    parser.add_argument("--delimiter", default=None, help="Optional delimiter for both input and reference files")
    parser.add_argument("--score-column", type=int, default=None, help="Optional score column index (0-based)")
    parser.add_argument("--scorer", default=None, help="Scorer name used to compute objective if score column is not provided")
    parser.add_argument("--topk", default="1,10,50", help="Comma-separated top-k list, e.g. 1,10,50")
    parser.add_argument("--max-diversity-pairs", type=int, default=50000, help="Max pair comparisons for diversity")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    scorer = get_scorer(args.scorer) if args.scorer else None
    topk_values = parse_topk(args.topk)
    if not topk_values:
        raise ValueError("--topk must contain at least one positive integer")

    reference_set: Set[str] = set()
    if args.reference_file:
        if not os.path.exists(args.reference_file):
            raise FileNotFoundError(f"Reference file not found: {args.reference_file}")
        reference_set = load_reference(args.reference_file, args.delimiter)

    total = 0
    valid = 0
    valid_canonical: List[str] = []
    objective_scores: List[float] = []

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            smiles, cols = parse_line(line, args.delimiter)
            if not smiles:
                continue

            total += 1
            if not is_valid_molecule(smiles):
                continue

            can = canonicalize_smiles(smiles)
            if not can:
                continue

            valid += 1
            valid_canonical.append(can)

            score = None
            if args.score_column is not None and args.score_column < len(cols):
                try:
                    score = float(cols[args.score_column])
                except ValueError:
                    score = None
            elif scorer is not None:
                score = float(scorer.score(can))

            if score is not None:
                objective_scores.append(score)

    valid_set = set(valid_canonical)

    validity = (valid / total) if total else 0.0
    uniqueness = (len(valid_set) / valid) if valid else 0.0

    novelty = None
    if reference_set:
        novel_valid = sum(1 for s in valid_set if s not in reference_set)
        novelty = (novel_valid / len(valid_set)) if valid_set else 0.0

    diversity = internal_diversity(list(valid_set), max_pairs=max(1, args.max_diversity_pairs))

    rows: List[Tuple[str, str]] = [
        ("Total molecules", str(total)),
        ("Valid molecules", str(valid)),
        ("Validity", f"{validity:.4f}"),
        ("Uniqueness", f"{uniqueness:.4f}"),
    ]

    if novelty is not None:
        rows.append(("Novelty", f"{novelty:.4f}"))
    else:
        rows.append(("Novelty", "N/A (no reference file)"))

    rows.append(("Internal diversity", f"{diversity:.4f}"))

    if objective_scores:
        sorted_scores = sorted(objective_scores, reverse=True)
        rows.append(("Objective mean", f"{float(np.mean(objective_scores)):.4f}"))
        for k in topk_values:
            k_eff = min(k, len(sorted_scores))
            topk_mean = float(np.mean(sorted_scores[:k_eff])) if k_eff > 0 else 0.0
            rows.append((f"Top-{k} objective", f"{topk_mean:.4f}"))
    else:
        rows.append(("Objective", "N/A (provide --score-column or --scorer)"))

    print(format_table(rows))


if __name__ == "__main__":
    main()
