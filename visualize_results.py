"""
2D grid and 3D conformer visualization for generated molecules.

Standalone workflow:
    python evaluate_agent.py ... --save-smiles generated.smi
    python visualize_results.py --input generated.smi --scorer qed --top 20

Or import draw_molecule_grid / draw_3d_molecule from evaluate_agent.py.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from rdkit import Chem
from rdkit.Chem import Draw, AllChem, Descriptors

from core.chemistry import canonicalize_smiles, smiles_to_mol
from core.scoring import get_scorer


# ---------------------------------------------------------------------------
# Atom styling
# ---------------------------------------------------------------------------

_ATOM_COLORS = {
    1:  "#DDDDDD",  # H
    6:  "#404040",  # C
    7:  "#3050F8",  # N
    8:  "#FF0D0D",  # O
    9:  "#90E050",  # F
    15: "#FF8000",  # P
    16: "#FFFF30",  # S
    17: "#1FF01F",  # Cl
    35: "#A62929",  # Br
    53: "#940094",  # I
}
_DEFAULT_COLOR = "#BBBBBB"


def _atom_color(atom_num: int) -> str:
    return _ATOM_COLORS.get(atom_num, _DEFAULT_COLOR)


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def load_smiles_file(path: str) -> list:
    """Load a line-delimited SMILES file.

    Accepts lines of the form:
        SMILES
        SMILES score
        SMILES score id
    Returns list of (smiles, score_or_None) tuples.
    """
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            smiles = parts[0]
            score = float(parts[1]) if len(parts) > 1 else None
            entries.append((smiles, score))
    return entries


# ---------------------------------------------------------------------------
# 2D grid
# ---------------------------------------------------------------------------

def draw_molecule_grid(
    smiles_scores: list,
    output_path: str,
    mol_size: tuple = (300, 300),
    mols_per_row: int = 5,
) -> None:
    """Draw a 2D molecular grid and save to output_path.

    Args:
        smiles_scores: list of (smiles, score) tuples; score may be None.
        output_path:   path to write the PNG.
        mol_size:      (width, height) in pixels per molecule.
        mols_per_row:  columns in the grid.
    """
    mols, legends = [], []
    for smiles, score in smiles_scores:
        mol = smiles_to_mol(smiles)
        if mol is None:
            continue
        AllChem.Compute2DCoords(mol)
        mols.append(mol)
        label = f"Score: {score:.3f}" if score is not None else (canonicalize_smiles(smiles) or smiles)[:24]
        legends.append(label)

    if not mols:
        print("[visualize] No valid molecules — skipping 2D grid.")
        return

    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=mols_per_row,
        subImgSize=mol_size,
        legends=legends,
        returnPNG=False,
    )
    img.save(output_path)
    print(f"[visualize] 2D grid saved → {output_path}  ({len(mols)} molecules)")


# ---------------------------------------------------------------------------
# 3D conformer
# ---------------------------------------------------------------------------

def draw_3d_molecule(
    smiles: str,
    output_path: str,
    score: float = None,
    title: str = None,
) -> None:
    """Render a 3D MMFF-optimised conformer and save to output_path.

    Falls back gracefully if conformer generation fails.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"[visualize] Cannot parse SMILES for 3D render: {smiles!r}")
        return

    mol_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    ok = AllChem.EmbedMolecule(mol_h, params)
    if ok == -1:
        # Fallback: random distance geometry
        ok = AllChem.EmbedMolecule(mol_h, randomSeed=42)
    if ok == -1:
        print(f"[visualize] Could not generate 3D conformer for: {smiles!r}")
        return

    AllChem.MMFFOptimizeMolecule(mol_h, maxIters=2000)
    mol_h = Chem.RemoveHs(mol_h)

    conf = mol_h.GetConformer()
    pos = conf.GetPositions()          # shape (n_atoms, 3)

    atom_nums = [atom.GetAtomicNum() for atom in mol_h.GetAtoms()]
    colors = [_atom_color(n) for n in atom_nums]
    sizes  = [120 + 60 * n / 6 for n in atom_nums]

    fig = plt.figure(figsize=(8, 7), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#F8F8F8")

    # Bonds
    for bond in mol_h.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        lw = 2.5 if bond.GetBondTypeAsDouble() >= 2 else 1.8
        ax.plot(
            [pos[i, 0], pos[j, 0]],
            [pos[i, 1], pos[j, 1]],
            [pos[i, 2], pos[j, 2]],
            color="#888888", linewidth=lw, alpha=0.9, zorder=1,
        )

    # Atoms
    ax.scatter(
        pos[:, 0], pos[:, 1], pos[:, 2],
        c=colors, s=sizes,
        edgecolors="#222222", linewidths=0.6,
        depthshade=True, zorder=2,
    )

    # Label heteroatoms
    for atom in mol_h.GetAtoms():
        if atom.GetAtomicNum() != 6:
            i = atom.GetIdx()
            ax.text(
                pos[i, 0], pos[i, 1], pos[i, 2] + 0.15,
                atom.GetSymbol(), fontsize=7.5, ha="center", va="bottom",
                fontweight="bold", color="#111111", zorder=3,
            )

    can = canonicalize_smiles(smiles) or smiles
    head = title or can
    if score is not None:
        head += f"\nScore: {score:.4f}"
    ax.set_title(head, fontsize=10, pad=6)
    ax.set_xlabel("X (Å)", fontsize=7)
    ax.set_ylabel("Y (Å)", fontsize=7)
    ax.set_zlabel("Z (Å)", fontsize=7)
    ax.tick_params(labelsize=6)

    # Element legend (unique atoms present)
    seen_nums = sorted({a.GetAtomicNum() for a in mol_h.GetAtoms()})
    pt = Chem.GetPeriodicTable()
    legend_patches = [
        Patch(color=_atom_color(n), label=pt.GetElementSymbol(n))
        for n in seen_nums
    ]
    ax.legend(handles=legend_patches, loc="upper left", fontsize=8, framealpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[visualize] 3D render saved  → {output_path}")


# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------

def print_stats(smiles_scores: list) -> None:
    """Print validity, score, and MW statistics."""
    valid = [(s, sc) for s, sc in smiles_scores if smiles_to_mol(s) is not None]
    n_total = len(smiles_scores)
    n_valid = len(valid)

    scores = [sc for _, sc in valid if sc is not None]
    mws    = [Descriptors.MolWt(smiles_to_mol(s)) for s, _ in valid]
    has_c  = [
        any(a.GetAtomicNum() == 6 for a in smiles_to_mol(s).GetAtoms())
        for s, _ in valid
    ]

    print(f"\n{'─'*52}")
    print(f"  Molecules total      : {n_total}")
    print(f"  Valid (RDKit)        : {n_valid}  ({100*n_valid/max(n_total,1):.1f}%)")
    print(f"  Contains carbon      : {sum(has_c)}  ({100*sum(has_c)/max(n_valid,1):.1f}% of valid)")
    if scores:
        print(f"  Score  mean ± std    : {np.mean(scores):.3f} ± {np.std(scores):.3f}")
        print(f"  Score  min / max     : {min(scores):.3f} / {max(scores):.3f}")
    if mws:
        print(f"  MW     mean ± std    : {np.mean(mws):.1f} ± {np.std(mws):.1f} Da")
    print(f"{'─'*52}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize generated molecules in 2D and 3D.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True,
                        help="Line-delimited SMILES file (e.g. from evaluate_agent.py --save-smiles).")
    parser.add_argument("--scorer", default=None,
                        help="Re-score with this scorer (qed, sa, logp, mw). Uses file scores if omitted.")
    parser.add_argument("--top", type=int, default=20,
                        help="Number of top-scoring molecules to display.")
    parser.add_argument("--output-dir", default=".",
                        help="Directory to write PNG files.")
    parser.add_argument("--mol-size", type=int, default=300,
                        help="Per-molecule pixel size for the 2D grid.")
    parser.add_argument("--mols-per-row", type=int, default=5,
                        help="Grid columns.")
    parser.add_argument("--no-3d", action="store_true",
                        help="Skip 3D conformer rendering.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    entries = load_smiles_file(args.input)

    if args.scorer:
        scorer = get_scorer(args.scorer)
        entries = [(s, scorer.score(s)) for s, _ in entries]

    # Keep only RDKit-valid entries
    valid_entries = [
        (s, sc) for s, sc in entries
        if smiles_to_mol(s) is not None and sc is not None
    ]
    valid_entries.sort(key=lambda x: x[1], reverse=True)

    print_stats(valid_entries)

    top = valid_entries[: args.top]
    if not top:
        print("No valid scored molecules found. Exiting.")
        return

    grid_path = os.path.join(args.output_dir, "molecules_2d_grid.png")
    draw_molecule_grid(top, grid_path, mol_size=(args.mol_size, args.mol_size), mols_per_row=args.mols_per_row)

    if not args.no_3d:
        best_smiles, best_score = top[0]
        three_d_path = os.path.join(args.output_dir, "best_molecule_3d.png")
        draw_3d_molecule(best_smiles, three_d_path, score=best_score)


if __name__ == "__main__":
    main()
