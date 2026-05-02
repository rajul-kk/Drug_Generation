"""
Unified training entry point for DrugGen-RL.

Dispatches to PPO (RecurrentPPO / MaskablePPO) or SAC depending on --agent.

Usage examples
--------------
# PPO with LSTM, QED objective
python train_agent.py --agent ppo --scorer qed --timesteps 600000

# PPO with action masking (MaskablePPO)
python train_agent.py --agent ppo --scorer qed --timesteps 600000 --mask-actions

# SAC, synthetic-accessibility objective
python train_agent.py --agent sac --scorer sa --timesteps 300000

# Resume from a checkpoint
python train_agent.py --agent ppo --scorer qed \
    --resume checkpoints/ppo_qed/best_model/best_model.zip

# With duplicate penalty + novelty bonus
python train_agent.py --agent ppo --scorer qed --timesteps 600000 \
    --duplicate-penalty 0.3 --novelty-bonus 0.1 \
    --reference-file data/chembl_reference.smi
"""

import argparse
import glob
import os
import re
import sys

# Ensure project root is on the path when called from any working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.chemistry import canonicalize_smiles
from core.scoring import get_scorer
from envs.molecule_env import MoleculeEnv


def find_latest_checkpoint(checkpoint_dir: str) -> tuple:
    """Return (model_zip_path, replay_buffer_path_or_None) for the most recent checkpoint.

    Search order:
      1. *_steps.zip files written by SB3 CheckpointCallback — picks highest step count.
      2. best_model/best_model.zip written by EvalCallback.
      3. interrupted_model.zip / final_model.zip (named saves).
    Returns (None, None) if the directory doesn't exist or contains no checkpoints.
    """
    if not os.path.isdir(checkpoint_dir):
        return None, None

    # 1. Step-numbered checkpoints
    candidates = []
    for path in glob.glob(os.path.join(checkpoint_dir, "*.zip")):
        m = re.search(r"_(\d+)_steps\.zip$", os.path.basename(path))
        if m:
            candidates.append((int(m.group(1)), path))

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        model_path = candidates[0][1]
        rb_path = model_path.replace(".zip", "_replay_buffer.pkl")
        return model_path, (rb_path if os.path.exists(rb_path) else None)

    # 2. best_model from EvalCallback
    best = os.path.join(checkpoint_dir, "best_model", "best_model.zip")
    if os.path.exists(best):
        return best, None

    # 3. Named saves
    for name in ("interrupted_model", "final_model"):
        path = os.path.join(checkpoint_dir, f"{name}.zip")
        if os.path.exists(path):
            rb_path = path.replace(".zip", "_replay_buffer.pkl")
            return path, (rb_path if os.path.exists(rb_path) else None)

    return None, None


def load_reference_smiles(path: str) -> set:
    loaded = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            smiles = line.strip().split()[0] if line.strip() else ""
            can = canonicalize_smiles(smiles)
            if can:
                loaded.add(can)
    return loaded


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a DrugGen-RL agent (PPO or SAC) for molecule generation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Agent selection
    parser.add_argument(
        "--agent", type=str, choices=["ppo", "sac"], required=True,
        help="RL algorithm to use.",
    )

    # Objective
    parser.add_argument(
        "--scorer", type=str, default="qed",
        help="Molecular scoring function. One of: qed, sa, logp, mw, tanimoto, multi.",
    )

    # Training duration
    parser.add_argument(
        "--timesteps", type=int, default=600000,
        help="Total environment steps to train for.",
    )
    parser.add_argument(
        "--max-steps", type=int, default=60,
        help="Maximum tokens per episode (SMILES length limit).",
    )

    # Checkpointing
    parser.add_argument(
        "--checkpoint-dir", type=str, default=None,
        help="Directory to save checkpoints. Defaults to checkpoints/<agent>_<scorer>.",
    )
    parser.add_argument(
        "--checkpoint-freq", type=int, default=50000,
        help="Save a checkpoint every N timesteps.",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to a .zip checkpoint to resume training from.",
    )
    parser.add_argument(
        "--continue", dest="auto_resume", action="store_true",
        help="Auto-detect and load the latest checkpoint from the checkpoint directory "
             "(highest step count). For SAC, also loads the replay buffer if present. "
             "Ignored if --resume is also given.",
    )

    # Action masking (PPO only)
    parser.add_argument(
        "--mask-actions", action="store_true",
        help="Use MaskablePPO (PPO) or env-side argmax masking (SAC).",
    )

    # Reward shaping
    parser.add_argument(
        "--duplicate-penalty", type=float, default=0.3,
        help="Scale factor applied to reward for duplicate molecules (1.0 = no penalty).",
    )
    parser.add_argument(
        "--novelty-bonus", type=float, default=0.0,
        help="Additive bonus for molecules not in the reference set.",
    )
    parser.add_argument(
        "--reference-file", type=str, default=None,
        help="Line-delimited SMILES file used for novelty calculation.",
    )
    parser.add_argument(
        "--step-penalty", type=float, default=0.0,
        help="Deduct step_penalty * (steps_used / max_steps) from terminal reward. "
             "Encourages shorter, decisive molecules. Try 0.05–0.1.",
    )
    parser.add_argument(
        "--validity-bonus", type=float, default=0.0,
        help="Flat bonus added for any RDKit-valid molecule before objective scoring. "
             "Bootstraps valid-SMILES learning early in training. Try 0.02–0.05.",
    )
    parser.add_argument(
        "--max-seen-smiles", type=int, default=10_000,
        help="Cap on the duplicate-penalty memory set. Cleared when full to prevent "
             "unbounded memory growth over long training runs.",
    )

    # Evaluation during training
    parser.add_argument(
        "--eval-freq", type=int, default=10000,
        help="Run evaluation callback every N timesteps.",
    )
    parser.add_argument(
        "--n-eval-episodes", type=int, default=10,
        help="Number of episodes per evaluation callback.",
    )

    # Misc
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level (0/1).")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Load optional reference set
    reference_smiles = None
    if args.reference_file:
        if not os.path.exists(args.reference_file):
            raise FileNotFoundError(f"Reference file not found: {args.reference_file}")
        reference_smiles = load_reference_smiles(args.reference_file)
        print(f"Loaded {len(reference_smiles)} canonical reference molecules.")

    checkpoint_dir = args.checkpoint_dir or f"./checkpoints/{args.agent}_{args.scorer}"

    # Resolve which checkpoint to load (if any)
    resume_path = args.resume
    replay_buffer_path = None
    if args.auto_resume and not args.resume:
        resume_path, replay_buffer_path = find_latest_checkpoint(checkpoint_dir)
        if resume_path:
            print(f"Auto-resuming from:   {resume_path}")
            if replay_buffer_path:
                print(f"Replay buffer found:  {replay_buffer_path}")
        else:
            print(f"No checkpoint found in {checkpoint_dir} — starting fresh.")

    scorer = get_scorer(args.scorer)
    is_continuous = args.agent == "sac"

    train_env = MoleculeEnv(
        scorer=scorer,
        max_steps=args.max_steps,
        continuous_actions=is_continuous,
        enable_action_masking=args.mask_actions,
        duplicate_penalty=args.duplicate_penalty,
        novelty_bonus=args.novelty_bonus,
        reference_smiles=reference_smiles,
        step_penalty=args.step_penalty,
        validity_bonus=args.validity_bonus,
        max_seen_smiles=args.max_seen_smiles,
    )
    eval_env = MoleculeEnv(
        scorer=scorer,
        max_steps=args.max_steps,
        continuous_actions=is_continuous,
        enable_action_masking=args.mask_actions,
        duplicate_penalty=1.0,   # no shaping during eval — clean objective signal
        novelty_bonus=0.0,
        reference_smiles=reference_smiles,
        step_penalty=0.0,
        validity_bonus=0.0,
        max_seen_smiles=0,       # disable memory during eval
    )

    tensorboard_log = f"./logs/{args.agent}_{args.scorer}"

    if args.agent == "ppo":
        from agents.PPO import PPOAgent

        agent = PPOAgent(
            env=train_env,
            use_action_mask=args.mask_actions,
            tensorboard_log=tensorboard_log,
            verbose=args.verbose,
            seed=args.seed,
        )
        if resume_path:
            print(f"Loading weights from {resume_path} ...")
            agent.load(resume_path)

        try:
            agent.train(
                total_timesteps=args.timesteps,
                checkpoint_freq=args.checkpoint_freq,
                checkpoint_path=checkpoint_dir,
                eval_env=eval_env,
                eval_freq=args.eval_freq,
                n_eval_episodes=args.n_eval_episodes,
            )
        except KeyboardInterrupt:
            print(f"\nInterrupted. Saving to {checkpoint_dir}/interrupted_model ...")
            agent.save(f"{checkpoint_dir}/interrupted_model")
            sys.exit(0)

        agent.save(f"{checkpoint_dir}/final_model")

    elif args.agent == "sac":
        from agents.SAC import SACAgent

        agent = SACAgent(
            env=train_env,
            use_action_mask=args.mask_actions,
            tensorboard_log=tensorboard_log,
            verbose=args.verbose,
            seed=args.seed,
        )
        if resume_path:
            print(f"Loading weights from {resume_path} ...")
            agent.load(resume_path, load_replay_buffer=bool(replay_buffer_path))

        try:
            agent.train(
                total_timesteps=args.timesteps,
                checkpoint_freq=args.checkpoint_freq,
                checkpoint_path=checkpoint_dir,
                eval_env=eval_env,
                eval_freq=args.eval_freq,
                n_eval_episodes=args.n_eval_episodes,
            )
        except KeyboardInterrupt:
            print(f"\nInterrupted. Saving to {checkpoint_dir}/interrupted_model ...")
            agent.save(f"{checkpoint_dir}/interrupted_model")
            sys.exit(0)

        agent.save(f"{checkpoint_dir}/final_model")

    print(f"\nTraining complete. Final model: {checkpoint_dir}/final_model.zip")


if __name__ == "__main__":
    main()
