"""
Molecular Generation Gym Environment.

This module provides a reinforcement learning environment compatible
with Stable Baselines3's PPO and SAC agents. The agent builds a SMILES
string token by token.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import logging
from typing import Optional, Dict, Any, Union

from core.scoring import MolecularScorer

logger = logging.getLogger(__name__)

# A simple vocabulary for basic drug-like SMILES
DEFAULT_VOCAB = [
    '<PAD>', '<STOP>', 
    'C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I',
    'c', 'n', 'o', 's', 'p',
    '1', '2', '3', '4', '5', '6',
    '(', ')', '[', ']', '=', '#', '-', '+', 
    '[nH]', '[O-]', '[N+]', '[nH+]'
]

class MoleculeEnv(gym.Env):
    """SMILES string building environment for RL agents."""
    metadata = {'render_modes': ['human']}

    def __init__(
        self, 
        scorer: MolecularScorer,
        vocab: list = DEFAULT_VOCAB,
        max_steps: int = 60,
        continuous_actions: bool = False
    ):
        # Initialize MoleculeEnv with scorer, vocabulary, and action space settings
        super().__init__()
        
        self.scorer = scorer
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.max_steps = max_steps
        self.continuous_actions = continuous_actions
        
        self.token_to_idx = {t: i for i, t in enumerate(self.vocab)}
        self.idx_to_token = {i: t for i, t in enumerate(self.vocab)}
        
        # Action space: Discrete (PPO) or Continuous (SAC -> Argmax inside step)
        if self.continuous_actions:
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(self.vocab_size,), dtype=np.float32
            )
        else:
            self.action_space = spaces.Discrete(self.vocab_size)
            
        # Observation space: flattened one-hot encoding of the sequence so far.
        # This is strictly required so standard MLP networks in SB3 can process it 
        # without crashing or treating token integer IDs as continuous magnitudes.
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, 
            shape=(self.max_steps * self.vocab_size,), 
            dtype=np.float32
        )
        
        # Internal state
        self.state_seq = np.zeros(self.max_steps, dtype=np.int32)
        self.current_step = 0
        self.smiles = ""

    def _get_obs(self) -> np.ndarray:
        """Returns the flattened one-hot encoded state sequence."""
        obs = np.zeros((self.max_steps, self.vocab_size), dtype=np.float32)
        for i in range(self.current_step):
            if i < self.max_steps:
                obs[i, self.state_seq[i]] = 1.0
        return obs.flatten()

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        """Resets the environment for a new molecule."""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.state_seq = np.zeros(self.max_steps, dtype=np.int32)
        self.smiles = ""
        
        return self._get_obs(), {}

    def step(self, action: Union[int, np.ndarray]):
        """Takes a step in the environment."""
        # Handle continuous actions from continuous SAC by taking the argmax
        if self.continuous_actions or isinstance(action, (np.ndarray, list)):
            action_idx = int(np.argmax(action))
        else:
            action_idx = int(action)
            
        token = self.idx_to_token.get(action_idx, '<PAD>')
        
        # Track the token in sequence memory
        if self.current_step < self.max_steps:
            self.state_seq[self.current_step] = action_idx
            
        self.current_step += 1
        
        terminated = False
        truncated = False
        reward = 0.0
        
        # Process the chosen token
        if token == '<STOP>':
            terminated = True
        elif token != '<PAD>':
            self.smiles += token
            
        # Check truncation limits
        if self.current_step >= self.max_steps:
            truncated = True
            
        # Terminal condition: calculate reward
        if terminated or truncated:
            if len(self.smiles) > 0:
                reward = self.scorer.score(self.smiles)
            else:
                reward = 0.0
                
        # Optional info dict
        info = {
            'smiles': self.smiles,
            'step': self.current_step,
            'is_valid': reward > 0.0 if (terminated or truncated) else False
        }
        
        return self._get_obs(), reward, terminated, truncated, info
        
    def render(self):
        """Console renderer for the environment."""
        print(f"Step {self.current_step}/{self.max_steps} | SMILES: {self.smiles}")
