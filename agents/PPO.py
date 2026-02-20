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
)
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
import gymnasium as gym


class PPOAgent:
    """
    PPO Agent with LSTM policy for molecular generation.
    
    This class wraps the RecurrentPPO algorithm from sb3-contrib, which uses
    LSTM layers to handle sequential dependencies in the molecule building process.
    
    Key Features:
    - LSTM-based policy for handling sequential molecule construction
    - Automatic checkpoint saving and loading
    - TensorBoard logging for monitoring training
    - Customizable hyperparameters optimized for molecular tasks
    """
    
    def __init__(
        self,
        env: Union[gym.Env, str],
        lstm_hidden_size: int = 256,
        policy_layers: tuple = (256, 256),
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
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
        """
        Initialize PPO agent with LSTM policy.
        
        Args:
            env: Gymnasium environment or environment ID string
            lstm_hidden_size: Size of LSTM hidden state (default: 256)
            policy_layers: Tuple of layer sizes for policy/value networks
            learning_rate: Learning rate for optimizer (default: 3e-4)
            n_steps: Number of steps to collect per environment per update
            batch_size: Minibatch size for training
            n_epochs: Number of epochs for policy update
            gamma: Discount factor
            gae_lambda: GAE lambda parameter for advantage estimation
            clip_range: PPO clipping parameter
            ent_coef: Entropy coefficient for exploration
            vf_coef: Value function coefficient in loss
            max_grad_norm: Maximum gradient norm for clipping
            verbose: Verbosity level (0: no output, 1: info, 2: debug)
            tensorboard_log: Path for tensorboard logs
            device: Device to use ('auto', 'cpu', 'cuda')
            seed: Random seed for reproducibility
        """
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
        """
        Train the PPO agent.
        
        Args:
            total_timesteps: Total number of timesteps to train
            checkpoint_freq: Frequency (in timesteps) to save checkpoints
            checkpoint_path: Directory to save checkpoints
            eval_env: Environment for evaluation (if None, uses training env)
            eval_freq: Frequency of evaluation
            n_eval_episodes: Number of episodes for evaluation
            callback: Additional custom callbacks
            
        Returns:
            self: Returns the trained agent
        """
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
        """
        Save the trained model.
        
        Args:
            path: Path to save the model (without extension)
        """
        self.model.save(path)
        if self.verbose >= 1:
            print(f"Model saved to {path}.zip")
    
    def load(self, path: str) -> "PPOAgent":
        """
        Load a trained model.
        
        Args:
            path: Path to the saved model (without extension)
            
        Returns:
            self: Returns the agent with loaded model
        """
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
        """
        Evaluate the trained agent.
        
        Args:
            n_eval_episodes: Number of episodes to evaluate
            deterministic: Use deterministic actions
            render: Whether to render the environment
            
        Returns:
            Dictionary with evaluation metrics
        """
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
    # Example with CartPole (replace with molecular environment)
    print("Initializing PPO agent with LSTM policy...")
    agent = PPOAgent(
        env="CartPole-v1",
        lstm_hidden_size=256,
        learning_rate=3e-4,
        verbose=1,
    )
    
    print("\nTraining agent...")
    agent.train(
        total_timesteps=50000,
        checkpoint_freq=10000,
    )
    
    print("\nEvaluating agent...")
    results = agent.evaluate(n_eval_episodes=10)
    
    print("\nSaving agent...")
    agent.save("./checkpoints/ppo_lstm/final_model")
