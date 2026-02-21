"""
PPO Agent with LSTM Policy for Drug Molecule Generation

This module implements a Proximal Policy Optimization (PPO) agent with LSTM
(Long Short-Term Memory) policy network for sequential decision-making in
molecular generation tasks.

The LSTM architecture is crucial for molecule generation because:
- It maintains memory of previously added atoms/bonds
- It can learn sequential patterns in valid molecular structures
- It handles variable-length molecule building sequences
"""

import os
from typing import Optional, Union, Dict, Any, Callable
import numpy as np
import torch as th
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
    BaseCallback,
)
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
import gymnasium as gym


class MetricPrinterCallback(BaseCallback):
    """Callback to print training metrics periodically while keeping the progress bar."""
    def __init__(self, print_freq: int = 2048, verbose: int = 0):
        super().__init__(verbose)
        self.print_freq = print_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.print_freq == 0:
            if hasattr(self.model, "ep_info_buffer") and len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
                print(f"\r[Step {self.num_timesteps}] Rolling Mean Reward: {mean_reward:.3f}")
        return True


class PPOAgent:
    """PPO Agent with LSTM policy for molecular generation."""
    
    def __init__(
        self,
        env: Union[gym.Env, str],
        lstm_hidden_size: int = 128,
        policy_layers: tuple = (128, 128),
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 128,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        verbose: int = 1,
        tensorboard_log: Optional[str] = "./logs/ppo_lstm",
        device: str = "auto",
        seed: Optional[int] = None,
    ):
        # Initialize PPO agent with LSTM policy using sb3_contrib RecurrentPPO
        # Create environment if string provided
        if isinstance(env, str):
            self.env = gym.make(env)
        else:
            self.env = env
        
        # Wrap in DummyVecEnv for compatibility
        self.vec_env = DummyVecEnv([lambda: self.env])
        
        # Store configuration
        self.config = {
            "lstm_hidden_size": lstm_hidden_size,
            "policy_layers": policy_layers,
            "learning_rate": learning_rate,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "clip_range": clip_range,
            "ent_coef": ent_coef,
            "vf_coef": vf_coef,
            "max_grad_norm": max_grad_norm,
        }
        
        # Policy kwargs for LSTM and network architecture
        policy_kwargs = {
            "lstm_hidden_size": lstm_hidden_size,
            "n_lstm_layers": 1,
            "net_arch": {
                "pi": list(policy_layers),  # Policy network layers
                "vf": list(policy_layers),  # Value function network layers
            },
            "activation_fn": th.nn.ReLU,
            "enable_critic_lstm": True,  # Use LSTM for both actor and critic
        }
        
        # Initialize RecurrentPPO model
        self.model = RecurrentPPO(
            policy="MlpLstmPolicy",
            env=self.vec_env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            verbose=verbose,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            device=device,
            seed=seed,
        )
        
        self.verbose = verbose
        
    def train(
        self,
        total_timesteps: int,
        checkpoint_freq: int = 10000,
        checkpoint_path: str = "./checkpoints/ppo_lstm",
        eval_env: Optional[gym.Env] = None,
        eval_freq: int = 5000,
        n_eval_episodes: int = 5,
        callback: Optional[Any] = None,
    ) -> "PPOAgent":
        """Train the PPO agent for a specified number of timesteps."""
        # Create checkpoint directory
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # Setup callbacks
        callbacks = []
        
        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=checkpoint_path,
            name_prefix="ppo_lstm_model",
            save_replay_buffer=False,
            save_vecnormalize=False,
        )
        callbacks.append(checkpoint_callback)
        
        # Evaluation callback
        if eval_env is not None:
            eval_vec_env = DummyVecEnv([lambda: eval_env])
            eval_callback = EvalCallback(
                eval_vec_env,
                best_model_save_path=os.path.join(checkpoint_path, "best_model"),
                log_path=os.path.join(checkpoint_path, "eval_logs"),
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                deterministic=True,
                render=False,
            )
            callbacks.append(eval_callback)
            
        # Metric Printer Callback
        callbacks.append(MetricPrinterCallback(print_freq=2048))
        
        # Add custom callback if provided
        if callback is not None:
            callbacks.append(callback)
        
        # Create callback list
        callback_list = CallbackList(callbacks)
        
        # Train the model
        if self.verbose >= 1:
            print(f"Starting PPO training for {total_timesteps} timesteps...")
            print(f"LSTM hidden size: {self.config['lstm_hidden_size']}")
            print(f"Checkpoints will be saved to: {checkpoint_path}")
        
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback_list,
            progress_bar=True,
        )
        
        if self.verbose >= 1:
            print("Training completed!")
        
        return self
    
    def predict(
        self,
        obs: np.ndarray,
        state: Optional[tuple] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> tuple:
        """
        Predict action using the trained policy.
        
        Args:
            obs: Current observation
            state: LSTM hidden state (if None, uses zeros)
            episode_start: Whether this is the start of an episode
            deterministic: Whether to use deterministic actions
            
        Returns:
            action: Predicted action
            state: Updated LSTM hidden state
        """
        if episode_start is None:
            episode_start = np.array([False])
        
        action, state = self.model.predict(
            obs,
            state=state,
            episode_start=episode_start,
            deterministic=deterministic,
        )
        
        return action, state
    
    def save(self, path: str) -> None:
        """Save the trained model."""
        self.model.save(path)
        if self.verbose >= 1:
            print(f"Model saved to {path}.zip")
    
    def load(self, path: str) -> "PPOAgent":
        """Load a trained model."""
        self.model = RecurrentPPO.load(path, env=self.vec_env)
        if self.verbose >= 1:
            print(f"Model loaded from {path}.zip")
        return self
    
    def evaluate(
        self,
        n_eval_episodes: int = 10,
        deterministic: bool = True,
        render: bool = False,
    ) -> Dict[str, float]:
        """Evaluate the trained agent and return metrics."""
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(n_eval_episodes):
            obs = self.vec_env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            # Reset LSTM states
            lstm_states = None
            episode_starts = np.ones((1,), dtype=bool)
            
            while not done:
                action, lstm_states = self.model.predict(
                    obs,
                    state=lstm_states,
                    episode_start=episode_starts,
                    deterministic=deterministic,
                )
                obs, reward, done, info = self.vec_env.step(action)
                
                episode_reward += reward[0]
                episode_length += 1
                episode_starts = np.zeros((1,), dtype=bool)
                
                if render:
                    self.vec_env.render()
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        results = {
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "mean_length": np.mean(episode_lengths),
            "std_length": np.std(episode_lengths),
        }
        
        if self.verbose >= 1:
            print(f"Evaluation over {n_eval_episodes} episodes:")
            print(f"  Mean reward: {results['mean_reward']:.2f} +/- {results['std_reward']:.2f}")
            print(f"  Mean length: {results['mean_length']:.2f} +/- {results['std_length']:.2f}")
        
        return results
    
    def get_config(self) -> Dict[str, Any]:
        """Get the agent configuration."""
        return self.config.copy()


# Example usage
if __name__ == "__main__":
    import argparse
    import sys
    import os
    
    # Needs to be imported here to prevent circular imports if scoring/env imports PPO
    # Add project root to sys.path so we can import core and envs correctly
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.scoring import get_scorer
    from envs.molecule_env import MoleculeEnv
    
    parser = argparse.ArgumentParser(description="Train PPO Agent for Molecule Generation")
    parser.add_argument("--scorer", type=str, default="qed", help="Scoring function")
    parser.add_argument("--timesteps", type=int, default=600000, help="Total training timesteps")
    parser.add_argument("--max_steps", type=int, default=60, help="Max tokens per molecule")
    args = parser.parse_args()
    
    print(f"--- Training PPO Agent ({args.timesteps} timesteps) ---")
    scorer = get_scorer(args.scorer)
    env = MoleculeEnv(scorer=scorer, max_steps=args.max_steps, continuous_actions=False)
    eval_env = MoleculeEnv(scorer=scorer, max_steps=args.max_steps, continuous_actions=False)
    
    agent = PPOAgent(
        env=env,
        tensorboard_log=f"./logs/ppo_{args.scorer}"
    )
    
    checkpoint_path = f"./checkpoints/ppo_{args.scorer}"
    
    try:
        agent.train(
            total_timesteps=args.timesteps,
            checkpoint_freq=50000,
            checkpoint_path=checkpoint_path,
            eval_env=eval_env,
            eval_freq=10000,
        )
        print(f"✅ PPO Training completed!")
    except KeyboardInterrupt:
        print(f"⚠️ Training interrupted. Saving to {checkpoint_path}/interrupted_model")
        agent.save(f"{checkpoint_path}/interrupted_model")
    
    print("\nSaving agent...")
    agent.save("./checkpoints/ppo_lstm/final_model")
