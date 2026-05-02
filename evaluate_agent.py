import argparse
import os
import sys
import numpy as np
from typing import List, Set

from rdkit import DataStructs
from rdkit.Chem import AllChem

# Ensure imports work regardless of where script is run
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.scoring import get_scorer
from envs.molecule_env import MoleculeEnv
from core.chemistry import smiles_to_mol, canonicalize_smiles


def load_smiles_file(path: str) -> Set[str]:
    """Load a line-delimited SMILES file and return canonicalized unique molecules."""
    molecules: Set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Accept either "SMILES" or "SMILES<whitespace>id" formats.
            smiles = line.split()[0]
            can = canonicalize_smiles(smiles)
            if can:
                molecules.add(can)
    return molecules


def compute_internal_diversity(canonical_smiles: List[str], max_pairs: int = 50000) -> float:
    """
    Compute internal diversity as 1 - mean pairwise Tanimoto similarity.
    Returns 0.0 if there are fewer than 2 valid molecules.
    """
    mols = [smiles_to_mol(s) for s in canonical_smiles]
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048) for m in mols if m is not None]
    n = len(fps)
    if n < 2:
        return 0.0

    total_sim = 0.0
    pair_count = 0
    step = 1

    total_possible_pairs = (n * (n - 1)) // 2
    if total_possible_pairs > max_pairs:
        # Uniformly subsample pair indices by stepping through comparisons.
        step = max(1, total_possible_pairs // max_pairs)

    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if idx % step == 0:
                total_sim += DataStructs.TanimotoSimilarity(fps[i], fps[j])
                pair_count += 1
            idx += 1

    if pair_count == 0:
        return 0.0

    mean_similarity = total_sim / pair_count
    return float(max(0.0, 1.0 - mean_similarity))

def main():
    parser = argparse.ArgumentParser(description="Evaluate a Trained DrugGen Agent")
    parser.add_argument("--agent", type=str, choices=["ppo", "sac"], required=True, help="Agent type (ppo or sac)")
    parser.add_argument("--scorer", type=str, default="qed", help="Scoring function used during training")
    parser.add_argument("--model", type=str, required=True, help="Path to the model .zip file")
    parser.add_argument("--episodes", type=int, default=10, help="Number of molecules to generate")
    parser.add_argument("--stochastic", action="store_true", help="Sample actions stochastically (default is deterministic)")
    parser.add_argument("--mask-actions", action="store_true", help="Enable action masking (MaskablePPO for PPO, env masking for SAC)")
    parser.add_argument("--reference-file", type=str, default=None, help="Optional line-delimited reference SMILES file for novelty")
    parser.add_argument("--duplicate-penalty", type=float, default=1.0, help="Duplicate molecule penalty in [0,1]; 1.0 disables penalty")
    parser.add_argument("--novelty-bonus", type=float, default=0.0, help="Bonus added to novel molecules when reference file is set")
    parser.add_argument("--max-diversity-pairs", type=int, default=50000, help="Max pairwise comparisons for internal diversity")
    parser.add_argument("--save-smiles", type=str, default=None, help="Write generated SMILES + scores to this file (one per line)")
    parser.add_argument("--save-img", type=str, default=None, help="Directory to save 2D grid and 3D render PNGs")
    args = parser.parse_args()
    
    print(f"Loading {args.agent.upper()} model from {args.model}...")
    
    is_continuous = (args.agent == "sac")
    scorer = get_scorer(args.scorer)

    reference_smiles: Set[str] = set()
    if args.reference_file:
        if not os.path.exists(args.reference_file):
            raise FileNotFoundError(f"Reference file not found: {args.reference_file}")
        reference_smiles = load_smiles_file(args.reference_file)
        print(f"Loaded {len(reference_smiles)} canonical reference molecules for novelty.")

    env = MoleculeEnv(
        scorer=scorer,
        max_steps=60,
        continuous_actions=is_continuous,
        enable_action_masking=args.mask_actions,
        duplicate_penalty=args.duplicate_penalty,
        novelty_bonus=args.novelty_bonus,
        reference_smiles=reference_smiles if reference_smiles else None,
    )
    
    if args.agent == "ppo":
        from agents.PPO import PPOAgent
        agent = PPOAgent(env=env, verbose=0, use_action_mask=args.mask_actions).load(args.model)
    else:
        from agents.SAC import SACAgent
        agent = SACAgent(env=env, verbose=0).load(args.model)
        
    print(f"\nGenerating {args.episodes} molecules...")
    print("-" * 60)
    
    valid_count = 0
    total_reward = 0.0
    deterministic = not args.stochastic
    generated_canonical = []
    valid_canonical = []
    scored_molecules = []   # (canonical_smiles, score) for every generated molecule
    
    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        
        # LSTM states for PPO
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)
        
        smiles = ""
        reward = 0.0
        info = {}
        
        while not done:
            if args.agent == "ppo":
                action, lstm_states = agent.predict(
                    obs, state=lstm_states, episode_start=episode_starts, deterministic=deterministic
                )
                episode_starts = np.zeros((1,), dtype=bool)
            else:
                action = agent.predict(obs, deterministic=deterministic)
                
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
        smiles = info.get('smiles', '')
        canonical = info.get('canonical_smiles', '')
        # Basic validation check using RDKit proxy via score, or info valid flag
        is_valid = info.get('is_valid', False)
        
        # Let's clean up terminal output for very long broken smiles
        display_smiles = smiles if len(smiles) < 80 else smiles[:77] + "..."
        
        valid_mark = "[VALID]" if is_valid else "[INVALID]"
        if is_valid:
            valid_count += 1
            if canonical:
                valid_canonical.append(canonical)

        if canonical:
            generated_canonical.append(canonical)
            scored_molecules.append((canonical, reward))

        total_reward += reward
        print(f"[{ep+1:02d}/{args.episodes:02d}] {valid_mark} Score: {reward:.3f}  |  {display_smiles}")

    print("-" * 60)
    validity = (valid_count / args.episodes) if args.episodes > 0 else 0.0
    avg_score = (total_reward / args.episodes) if args.episodes > 0 else 0.0

    unique_generated = len(set(generated_canonical))
    uniqueness_all = (unique_generated / len(generated_canonical)) if generated_canonical else 0.0
    unique_valid = len(set(valid_canonical))
    uniqueness_valid = (unique_valid / len(valid_canonical)) if valid_canonical else 0.0

    novelty = None
    if reference_smiles:
        novel_valid = sum(1 for s in valid_canonical if s not in reference_smiles)
        novelty = (novel_valid / len(valid_canonical)) if valid_canonical else 0.0

    internal_diversity = compute_internal_diversity(valid_canonical, max_pairs=max(1, args.max_diversity_pairs))

    print(f"Summary: Generated {valid_count}/{args.episodes} valid molecules ({validity*100:.1f}%).")
    print(f"Average Score: {avg_score:.3f}")
    print(f"Uniqueness (all canonical): {uniqueness_all*100:.1f}%")
    print(f"Uniqueness (valid canonical): {uniqueness_valid*100:.1f}%")
    if novelty is not None:
        print(f"Novelty (valid vs reference): {novelty*100:.1f}%")
    print(f"Internal Diversity (1 - mean Tanimoto): {internal_diversity:.3f}")

    # ------------------------------------------------------------------ #
    # Optional: save SMILES file                                           #
    # ------------------------------------------------------------------ #
    if args.save_smiles:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_smiles)), exist_ok=True)
        with open(args.save_smiles, "w", encoding="utf-8") as f:
            for smi, score in scored_molecules:
                f.write(f"{smi} {score:.6f}\n")
        print(f"SMILES saved to {args.save_smiles}")

    # ------------------------------------------------------------------ #
    # Optional: render 2D grid + 3D best molecule                         #
    # ------------------------------------------------------------------ #
    if args.save_img:
        from visualize_results import draw_molecule_grid, draw_3d_molecule
        os.makedirs(args.save_img, exist_ok=True)
        top_scored = sorted(scored_molecules, key=lambda x: x[1], reverse=True)
        top20 = top_scored[:20]
        grid_path = os.path.join(args.save_img, "molecules_2d_grid.png")
        draw_molecule_grid(top20, grid_path)
        if top20:
            best_smi, best_score = top20[0]
            draw_3d_molecule(best_smi, os.path.join(args.save_img, "best_molecule_3d.png"), score=best_score)

if __name__ == "__main__":
    main()
