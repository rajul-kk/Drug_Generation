"""
SAC Agent for Drug Molecule Generation

This module implements a Soft Actor-Critic (SAC) agent for molecular generation tasks.
SAC is an off-policy algorithm that is well-suited for continuous control and can be
adapted for discrete action spaces.

Key advantages of SAC for molecular generation:
- Sample efficiency through off-policy learning (reuses past experiences)
- Automatic entropy tuning for exploration-exploitation balance
- Stable training through twin critic networks
- Works well with continuous molecular property optimization
"""

import os
from typing import Optional, Union, Dict, Any, Callable
import numpy as np
import torch as th
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
    BaseCallback,
)
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.noise import NormalActionNoise
import gymnasium as gym


class MetricPrinterCallback(BaseCallback):
    """Callback to print training metrics periodically while keeping the progress bar."""
    def __init__(self, print_freq: int = 2000, verbose: int = 0):
        super().__init__(verbose)
        self.print_freq = print_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.print_freq == 0:
            if hasattr(self.model, "ep_info_buffer") and len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
                print(f"\r[Step {self.num_timesteps}] Rolling Mean Reward: {mean_reward:.3f}")
        return True


class SACAgent:
    """Soft Actor-Critic (SAC) Agent for continuous molecular generation tasks."""
    
    def __init__(
        self,
        env: Union[gym.Env, str],
        policy_layers: tuple = (256, 256),
        learning_rate: float = 3e-4,
        buffer_size: int = 100_000,
        learning_starts: int = 10000,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, tuple] = 1,
        gradient_steps: int = 1,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        use_action_noise: bool = False,
        action_noise_std: float = 0.1,
        verbose: int = 1,
        tensorboard_log: Optional[str] = "./logs/sac",
        device: str = "auto",
        seed: Optional[int] = None,
    ):
        # Initialize SAC agent with stable-baselines3 SAC algorithm
        # Create environment if string provided
        if isinstance(env, str):
            self.env = gym.make(env)
        else:
            self.env = env
        
        # Wrap in DummyVecEnv for compatibility
        self.vec_env = DummyVecEnv([lambda: self.env])
        
        # Store configuration
        self.config = {
            "policy_layers": policy_layers,
            "learning_rate": learning_rate,
            "buffer_size": buffer_size,
            "learning_starts": learning_starts,
            "batch_size": batch_size,
            "tau": tau,
            "gamma": gamma,
            "train_freq": train_freq,
            "gradient_steps": gradient_steps,
            "ent_coef": ent_coef,
            "target_update_interval": target_update_interval,
        }
        
        # Policy kwargs for network architecture
        policy_kwargs = {
            "net_arch": {
                "pi": list(policy_layers),  # Actor network layers
                "qf": list(policy_layers),  # Q-function (critic) network layers
            },
            "activation_fn": th.nn.ReLU,
        }
        
        # Add State Dependent Exploration if requested
        if use_sde:
            policy_kwargs["use_sde"] = True
        
        # Create action noise if requested
        action_noise = None
        if use_action_noise and hasattr(self.env.action_space, 'shape'):
            n_actions = self.env.action_space.shape[0]
            action_noise = NormalActionNoise(
                mean=np.zeros(n_actions),
                sigma=action_noise_std * np.ones(n_actions)
            )
        
        # Initialize SAC model
        self.model = SAC(
            policy="MlpPolicy",
            env=self.vec_env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            ent_coef=ent_coef,
            target_update_interval=target_update_interval,
            target_entropy=target_entropy,
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
        checkpoint_path: str = "./checkpoints/sac",
        eval_env: Optional[gym.Env] = None,
        eval_freq: int = 5000,
        n_eval_episodes: int = 5,
        callback: Optional[Any] = None,
    ) -> "SACAgent":
        """Train the SAC agent for a specified number of timesteps."""
        # Create checkpoint directory
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # Setup callbacks
        callbacks = []
        
        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=checkpoint_path,
            name_prefix="sac_model",
            save_replay_buffer=True,  # SAC benefits from saving replay buffer
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
        callbacks.append(MetricPrinterCallback(print_freq=2000))
        
        # Add custom callback if provided
        if callback is not None:
            callbacks.append(callback)
        
        # Create callback list
        callback_list = CallbackList(callbacks)
        
        # Train the model
        if self.verbose >= 1:
            print(f"Starting SAC training for {total_timesteps} timesteps...")
            print(f"Actor network: {self.config['policy_layers']}")
            print(f"Critic network: {self.config['policy_layers']}")
            print(f"Replay buffer size: {self.config['buffer_size']}")
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
        deterministic: bool = False,
    ) -> np.ndarray:
        """Predict action using the trained policy."""
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return action
    
    def save(self, path: str, save_replay_buffer: bool = True) -> None:
        """Save the trained model and optionally the replay buffer."""
        self.model.save(path)
        
        if save_replay_buffer and self.model.replay_buffer is not None:
            replay_buffer_path = f"{path}_replay_buffer.pkl"
            self.model.save_replay_buffer(replay_buffer_path)
            if self.verbose >= 1:
                print(f"Model and replay buffer saved to {path}")
        elif self.verbose >= 1:
            print(f"Model saved to {path}.zip")
    
    def load(
        self,
        path: str,
        load_replay_buffer: bool = False,
    ) -> "SACAgent":
        """Load a trained model and optionally the replay buffer."""
        self.model = SAC.load(path, env=self.vec_env)
        
        if load_replay_buffer:
            replay_buffer_path = f"{path}_replay_buffer.pkl"
            if os.path.exists(replay_buffer_path):
                self.model.load_replay_buffer(replay_buffer_path)
                if self.verbose >= 1:
                    print(f"Model and replay buffer loaded from {path}")
            else:
                if self.verbose >= 1:
                    print(f"Model loaded from {path}.zip (no replay buffer found)")
        elif self.verbose >= 1:
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
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=deterministic)
                obs, reward, done, info = self.vec_env.step(action)
                
                episode_reward += reward[0]
                episode_length += 1
                
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
    
    def get_replay_buffer_size(self) -> int:
        """Get the current size of the replay buffer."""
        if self.model.replay_buffer is not None:
            return self.model.replay_buffer.size()
        return 0


class DiscreteSACAgent(SACAgent):
    """SAC Agent adapted for discrete action spaces."""
    
    def __init__(self, *args, **kwargs):
        # Initialize Discrete SAC agent
        # Check if environment has discrete action space
        super().__init__(*args, **kwargs)
        
        if not isinstance(self.env.action_space, gym.spaces.Discrete):
            print("Warning: DiscreteSACAgent is designed for discrete action spaces.")
            print(f"Current action space: {self.env.action_space}")


# Example usage
if __name__ == "__main__":
    import argparse
    import sys
    import os
    
    # Needs to be imported here to prevent circular imports if scoring/env imports SAC
    # Add project root to sys.path so we can import core and envs correctly
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.scoring import get_scorer
    from envs.molecule_env import MoleculeEnv
    
    parser = argparse.ArgumentParser(description="Train SAC Agent for Molecule Generation")
    parser.add_argument("--scorer", type=str, default="qed", help="Scoring function")
    parser.add_argument("--timesteps", type=int, default=300000, help="Total training timesteps")
    parser.add_argument("--max_steps", type=int, default=60, help="Max tokens per molecule")
    args = parser.parse_args()
    
    print(f"--- Training SAC Agent ({args.timesteps} timesteps) ---")
    scorer = get_scorer(args.scorer)
    
    # SAC in SB3 requires continuous action space
    env = MoleculeEnv(scorer=scorer, max_steps=args.max_steps, continuous_actions=True)
    eval_env = MoleculeEnv(scorer=scorer, max_steps=args.max_steps, continuous_actions=True)
    
    agent = SACAgent(
        env=env,
        tensorboard_log=f"./logs/sac_{args.scorer}"
    )
    
    checkpoint_path = f"./checkpoints/sac_{args.scorer}"
    
    try:
        agent.train(
            total_timesteps=args.timesteps,
            checkpoint_freq=50000,
            checkpoint_path=checkpoint_path,
            eval_env=eval_env,
            eval_freq=10000,
        )
        print(f"✅ SAC Training completed!")
    except KeyboardInterrupt:
        print(f"⚠️ Training interrupted. Saving to {checkpoint_path}/interrupted_model")
        agent.save(f"{checkpoint_path}/interrupted_model")
