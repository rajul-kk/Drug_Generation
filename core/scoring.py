"""
Base scoring infrastructure for molecular generation.

This module provides the foundation for evaluating molecules,
defining the MolecularScorer base class and a factory pattern 
to manage and instantiate various scoring functions.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Type
import logging
import numpy as np

from core.chemistry import is_valid_molecule

# Configure a logger for scoring
logger = logging.getLogger(__name__)

class MolecularScorer(ABC):
    """Abstract base class for all molecular scoring functions."""
    
    @abstractmethod
    def score(self, smiles: str) -> float:
        """Evaluate a single molecule and return a float score in [0.0, 1.0]."""
        raise NotImplementedError()
        
    def score_batch(self, smiles_list: List[str]) -> List[float]:
        """Evaluate a batch of molecules sequentially."""
        return [self.score(smiles) for smiles in smiles_list]


# Global registry for scorer classes
_SCORER_REGISTRY: Dict[str, Type[MolecularScorer]] = {}

def register_scorer(name: str, scorer_class: Type[MolecularScorer]) -> None:
    """Register a new scorer class under a string identifier."""
    if name in _SCORER_REGISTRY:
        logger.warning(f"Overwriting existing scorer registered under '{name}'")
    _SCORER_REGISTRY[name] = scorer_class

def get_scorer(name: str, **kwargs: Any) -> MolecularScorer:
    """Factory function to instantiate a scorer by name."""
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


# ==========================================
# GuacaMol Benchmark Scorers
# ==========================================

from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem import inchi
from core.chemistry import canonicalize_smiles


class TanimotoScorer(MolecularScorer):
    """
    Tanimoto Similarity Scorer.
    Calculates the Tanimoto similarity between the generated molecule
    and a target molecule using Morgan fingerprints (ECFP-like).
    Used for goal-directed generation and analogue design.
    """
    def __init__(self, target_smiles: str, radius: int = 2, n_bits: int = 2048):
        self.target_smiles = target_smiles
        self.radius = radius
        self.n_bits = n_bits
        
        target_mol = smiles_to_mol(target_smiles)
        if target_mol is not None:
            self.target_fp = AllChem.GetMorganFingerprintAsBitVect(
                target_mol, radius, nBits=n_bits
            )
        else:
            self.target_fp = None

    def score(self, smiles: str) -> float:
        if self.target_fp is None:
            return 0.0
            
        mol = smiles_to_mol(smiles)
        if mol is None:
            return 0.0
            
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.n_bits)
            return float(DataStructs.TanimotoSimilarity(self.target_fp, fp))
        except Exception:
            return 0.0


class IsomerScorer(MolecularScorer):
    """
    Exact Isomer matching (InChI Key comparison).
    Returns 1.0 if the identical structural isomer is generated, otherwise 0.0.
    """
    def __init__(self, target_smiles: str):
        target_mol = smiles_to_mol(target_smiles)
        if target_mol is not None:
            self.target_inchi = inchi.MolToInchiKey(target_mol)
        else:
            self.target_inchi = None

    def score(self, smiles: str) -> float:
        if self.target_inchi is None:
            return 0.0
            
        mol = smiles_to_mol(smiles)
        if mol is None:
            return 0.0
            
        try:
            mol_inchi = inchi.MolToInchiKey(mol)
            return 1.0 if mol_inchi == self.target_inchi else 0.0
        except Exception:
            return 0.0


class RediscoveryScorer(MolecularScorer):
    """
    Rediscovery Scorer (SMILES exact match).
    Compares canonical SMILES to rediscover a specific molecule.
    Returns 1.0 on exact match, 0.0 otherwise.
    """
    def __init__(self, target_smiles: str):
        self.target_smiles = canonicalize_smiles(target_smiles)

    def score(self, smiles: str) -> float:
        if not self.target_smiles:
            return 0.0
            
        can_smiles = canonicalize_smiles(smiles)
        if not can_smiles:
            return 0.0
            
        return 1.0 if can_smiles == self.target_smiles else 0.0


# Register GuacaMol scorers
register_scorer("tanimoto", TanimotoScorer)
register_scorer("isomer", IsomerScorer)
register_scorer("rediscovery", RediscoveryScorer)


# ==========================================
# Multi-Objective Scoring
# ==========================================

class MultiObjectiveScorer(MolecularScorer):
    """
    Combines multiple scorers into a single scalar reward.
    
    Can use arithmetic mean (weighted average) or geometric mean.
    Geometric mean is often preferred for multi-parameter optimization
    as a score of 0 in any single objective forces the total score to 0.
    """
    def __init__(
        self, 
        scorers: List[Dict[str, Any]], 
        method: str = "geometric"
    ):
        """
        Args:
            scorers: List of dictionaries defining the component scorers:
                [{'name': 'qed', 'weight': 1.0}, {'name': 'logp', 'weight': 0.5, 'kwargs': {}}]
            method: 'arithmetic' or 'geometric'
        """
        self.method = method.lower()
        if self.method not in ["arithmetic", "geometric"]:
            raise ValueError(f"Unknown aggregation method: {self.method}")
            
        self.components = []
        self.weights = []
        
        for config in scorers:
            name = config['name']
            weight = config.get('weight', 1.0)
            kwargs = config.get('kwargs', {})
            
            scorer = get_scorer(name, **kwargs)
            self.components.append(scorer)
            self.weights.append(weight)
            
        # Normalize weights so they sum to 1.0
        total_weight = sum(self.weights)
        if total_weight > 0:
            self.weights = [w / total_weight for w in self.weights]
            
    def score(self, smiles: str) -> float:
        if not self.components:
            return 0.0
            
        scores = [scorer.score(smiles) for scorer in self.components]
        
        # If any essential score is exactly 0.0, a geometric mean is 0.
        if self.method == "geometric" and 0.0 in scores:
            return 0.0
            
        if self.method == "arithmetic":
            total = sum(w * s for w, s in zip(self.weights, scores))
            return float(total)
            
        elif self.method == "geometric":
            import numpy as np
            # Note: scores are assumed to be in [0, 1]
            log_scores = np.log(np.maximum(scores, 1e-10))
            weighted_log = sum(w * ls for w, ls in zip(self.weights, log_scores))
            return float(np.exp(weighted_log))
            
        return 0.0

register_scorer("multi", MultiObjectiveScorer)
