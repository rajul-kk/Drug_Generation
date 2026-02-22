import argparse
import os
import sys
import numpy as np

# Ensure imports work regardless of where script is run
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.scoring import get_scorer
from envs.molecule_env import MoleculeEnv

def main():
    parser = argparse.ArgumentParser(description="Evaluate a Trained DrugGen Agent")
    parser.add_argument("--agent", type=str, choices=["ppo", "sac"], required=True, help="Agent type (ppo or sac)")
    parser.add_argument("--scorer", type=str, default="qed", help="Scoring function used during training")
    parser.add_argument("--model", type=str, required=True, help="Path to the model .zip file")
    parser.add_argument("--episodes", type=int, default=10, help="Number of molecules to generate")
    parser.add_argument("--stochastic", action="store_true", help="Sample actions stochastically (default is deterministic)")
    args = parser.parse_args()
    
    print(f"Loading {args.agent.upper()} model from {args.model}...")
    
    is_continuous = (args.agent == "sac")
    scorer = get_scorer(args.scorer)
    env = MoleculeEnv(scorer=scorer, max_steps=60, continuous_actions=is_continuous)
    
    if args.agent == "ppo":
        from agents.PPO import PPOAgent
        agent = PPOAgent(env=env, verbose=0).load(args.model)
    else:
        from agents.SAC import SACAgent
        agent = SACAgent(env=env, verbose=0).load(args.model)
        
    print(f"\nGenerating {args.episodes} molecules...")
    print("-" * 60)
    
    valid_count = 0
    total_reward = 0.0
    deterministic = not args.stochastic
    
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
        # Basic validation check using RDKit proxy via score, or info valid flag
        is_valid = info.get('is_valid', False)
        
        # Let's clean up terminal output for very long broken smiles
        display_smiles = smiles if len(smiles) < 80 else smiles[:77] + "..."
        
        valid_mark = "✅" if is_valid else "❌"
        if is_valid:
            valid_count += 1
            
        total_reward += reward
        print(f"[{ep+1:02d}/{args.episodes:02d}] {valid_mark} Score: {reward:.3f}  |  {display_smiles}")

    print("-" * 60)
    print(f"Summary: Generated {valid_count}/{args.episodes} valid molecules ({(valid_count/args.episodes)*100:.1f}%).")
    print(f"Average Score: {total_reward/args.episodes:.3f}")

if __name__ == "__main__":
    main()
