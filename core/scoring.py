"""
Base scoring infrastructure for molecular generation.

This module provides the foundation for evaluating molecules,
defining the MolecularScorer base class and a factory pattern 
to manage and instantiate various scoring functions.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Type
import logging

from core.chemistry import is_valid_molecule

# Configure a logger for scoring
logger = logging.getLogger(__name__)

class MolecularScorer(ABC):
    """
    Abstract base class for all molecular scoring functions.
    
    Scorers evaluate a SMILES string and return a float value,
    typically normalized to the range [0.0, 1.0], where higher
    is better (representing a higher reward for the RL agent).
    """
    
    @abstractmethod
    def score(self, smiles: str) -> float:
        """
        Evaluate a single molecule.
        
        Args:
            smiles: The SMILES string of the molecule to score.
            
        Returns:
            A float score, typically in [0.0, 1.0]. A score of 0.0
            is typically returned for invalid molecules.
        """
        raise NotImplementedError()
        
    def score_batch(self, smiles_list: List[str]) -> List[float]:
        """
        Evaluate a batch of molecules.
        
        This default implementation calls score() sequentially.
        Derived classes may override this for vectorized performance.
        
        Args:
            smiles_list: A list of SMILES strings.
            
        Returns:
            A list of float scores corresponding to the input molecules.
        """
        return [self.score(smiles) for smiles in smiles_list]


# Global registry for scorer classes
_SCORER_REGISTRY: Dict[str, Type[MolecularScorer]] = {}

def register_scorer(name: str, scorer_class: Type[MolecularScorer]) -> None:
    """
    Register a new scorer class under a string identifier.
    
    Args:
        name: The identifier name (e.g., 'qed', 'tanimoto').
        scorer_class: The MolecularScorer subclass to register.
    """
    if name in _SCORER_REGISTRY:
        logger.warning(f"Overwriting existing scorer registered under '{name}'")
    _SCORER_REGISTRY[name] = scorer_class

def get_scorer(name: str, **kwargs: Any) -> MolecularScorer:
    """
    Factory function to instantiate a scorer by name.
    
    Args:
        name: The identifier name of the registered scorer.
        **kwargs: Configuration arguments passed to the scorer's __init__.
        
    Returns:
        An instantiated MolecularScorer.
        
    Raises:
        ValueError: If the requested scorer name is not registered.
    """
    scorer_class = _SCORER_REGISTRY.get(name)
    if scorer_class is None:
        raise ValueError(
            f"Unknown scorer '{name}'. Available scorers: {list(_SCORER_REGISTRY.keys())}"
        )
    return scorer_class(**kwargs)
