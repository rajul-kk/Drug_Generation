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


# ==========================================
# Basic Physicochemical Descriptors
# ==========================================

from rdkit.Chem import Descriptors, QED
from core.chemistry import smiles_to_mol


class QEDScorer(MolecularScorer):
    """
    Quantitative Estimate of Drug-likeness (QED).
    Returns a score in [0, 1] based on multiple drug-like properties.
    """
    def score(self, smiles: str) -> float:
        mol = smiles_to_mol(smiles)
        if mol is None:
            return 0.0
        try:
            return float(QED.qed(mol))
        except Exception:
            return 0.0


class SAScorer(MolecularScorer):
    """
    Synthetic Accessibility (SA) Score.
    Normal SA Score is [1, 10] (1 is easy, 10 is hard).
    This scorer inverts and normalizes it to [0, 1] where 1.0 is easiest.
    """
    def __init__(self):
        # We need the SA_Score module from RDKit contrib
        try:
            from rdkit.Chem import RDConfig
            import sys, os
            sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
            import sascorer
            self.sascorer = sascorer
        except ImportError:
            logger.warning("RDKit SA_Score contrib module missing. SAScorer will return 0.0")
            self.sascorer = None

    def score(self, smiles: str) -> float:
        if self.sascorer is None:
            return 0.0
            
        mol = smiles_to_mol(smiles)
        if mol is None:
            return 0.0
            
        try:
            sa_score = self.sascorer.calculateScore(mol)
            # Normalize from [1, 10] to [0, 1] (inverted)
            # 1.0 -> 1.0 (best)
            # 10.0 -> 0.0 (worst)
            normalized = max(0.0, min(1.0, 1.0 - (sa_score - 1.0) / 9.0))
            return float(normalized)
        except Exception:
            return 0.0


class LogPScorer(MolecularScorer):
    """
    LogP Scorer with a Gaussian modifier around a target range.
    Typically drug-like LogP is [0, 5].
    """
    def __init__(self, target_range: tuple = (1.0, 4.0), sigma: float = 1.0):
        self.min_target, self.max_target = target_range
        self.sigma = sigma

    def score(self, smiles: str) -> float:
        mol = smiles_to_mol(smiles)
        if mol is None:
            return 0.0
            
        try:
            logp = Descriptors.MolLogP(mol)
            
            # If within range, score is 1.0
            if self.min_target <= logp <= self.max_target:
                return 1.0
            
            # If outside range, apply Gaussian decay
            if logp < self.min_target:
                dist = self.min_target - logp
            else:
                dist = logp - self.max_target
                
            score = np.exp(-0.5 * (dist / self.sigma) ** 2)
            return float(score)
        except Exception:
            return 0.0


class MolecularWeightScorer(MolecularScorer):
    """
    Molecular Weight Scorer.
    Typically drug-like MW is [200, 600] Da.
    """
    def __init__(self, target_range: tuple = (250.0, 500.0), sigma: float = 50.0):
        self.min_target, self.max_target = target_range
        self.sigma = sigma

    def score(self, smiles: str) -> float:
        mol = smiles_to_mol(smiles)
        if mol is None:
            return 0.0
            
        try:
            mw = Descriptors.MolWt(mol)
            if self.min_target <= mw <= self.max_target:
                return 1.0
            
            if mw < self.min_target:
                dist = self.min_target - mw
            else:
                dist = mw - self.max_target
                
            score = np.exp(-0.5 * (dist / self.sigma) ** 2)
            return float(score)
        except Exception:
            return 0.0


# Register basic scorers
register_scorer("qed", QEDScorer)
register_scorer("sa", SAScorer)
register_scorer("logp", LogPScorer)
register_scorer("mw", MolecularWeightScorer)
