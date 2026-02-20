"""
Chemistry utility functions for molecular operations.

This module provides common wrappers around RDKit functionality
for validating, converting, and analyzing molecules built by RL agents.
"""

from typing import Optional
import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError as e:
    raise ImportError("RDKit is required. Run: pip install -r requirements.txt") from e

def smiles_to_mol(smiles: str) -> Optional[Chem.Mol]:
    """
    Convert a SMILES string to an RDKit Mol object.
    
    Args:
        smiles: The SMILES string of the molecule.
        
    Returns:
        The RDKit Mol object if valid, None otherwise.
    """
    if not smiles or not isinstance(smiles, str):
        return None
    
    try:
        # Disable RDKit logging for invalid molecules during RL exploration
        Chem.rdBase.DisableLog('rdApp.error')
        mol = Chem.MolFromSmiles(smiles)
        Chem.rdBase.EnableLog('rdApp.error')
        return mol
    except Exception:
        return None

def is_valid_molecule(smiles: str) -> bool:
    """
    Check if a SMILES string represents a valid molecule.
    
    Args:
        smiles: The SMILES string to check.
        
    Returns:
        True if the SMILES is valid chemically, False otherwise.
    """
    mol = smiles_to_mol(smiles)
    if mol is None:
        return False
        
    # Check for basic chemical validity (valence, etc)
    try:
        Chem.SanitizeMol(mol)
        return True
    except Exception:
        return False

def canonicalize_smiles(smiles: str) -> str:
    """
    Convert a SMILES string to its canonical form for consistent comparison.
    
    Args:
        smiles: The SMILES string to canonicalize.
        
    Returns:
        The canonical SMILES string, or an empty string if invalid.
    """
    mol = smiles_to_mol(smiles)
    if mol is None:
        return ""
        
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return ""

def get_morgan_fingerprint(mol: Chem.Mol, radius: int = 2, n_bits: int = 2048) -> Optional[np.ndarray]:
    """
    Generate a Morgan fingerprint (ECFP-like) for a molecule.
    
    Args:
        mol: The RDKit Mol object.
        radius: The Morgan fingerprint radius (default: 2, equivalent to ECFP4).
        n_bits: Number of bits in the fingerprint vector.
        
    Returns:
        A NumPy array containing the binary fingerprint, or None if generation fails.
    """
    if mol is None:
        return None
        
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((1,), dtype=np.int8)
        from rdkit import DataStructs
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except Exception:
        return None
