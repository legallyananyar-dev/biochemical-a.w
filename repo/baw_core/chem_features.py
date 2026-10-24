"""
chem_features.py
-----------------
Turns a molecule into two per-atom numeric arrays using RDKit:

  - bond_dissonance : how much each atom's bond angles deviate from the
                       "ideal" angle for its hybridization (a proxy for
                       geometric strain).
  - chirality        : R = +1.0, S = -1.0, non-chiral = 0.0 (CIP labels).

RDKit needs a SMILES string, not just a formula like "C8H10N4O2" (many
molecules share a formula), so PRESETS maps friendly ids to real SMILES.
"""

from functools import lru_cache

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms

# id -> (display name, SMILES, risk level shown in the UI)
PRESETS = {
    "H2O":       ("Water",       "O",                                  "Low"),
    "C9H8O4":    ("Aspirin",     "CC(=O)OC1=CC=CC=C1C(=O)O",           "High"),
    "C8H10N4O2": ("Caffeine",    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",       "Maximum"),
    "L-Ala":     ("L-Alanine",   "C[C@H](N)C(=O)O",                    "High"),
    "Ibuprofen": ("Ibuprofen",   "CC(C)Cc1ccc(cc1)C(C)C(=O)O",         "High"),
    "Paracetamol": ("Paracetamol", "CC(=O)Nc1ccc(O)cc1",               "Medium"),
}
MOLECULE_PRESETS = {k: v[1] for k, v in PRESETS.items()}  # backward compatible

MAX_HEAVY_ATOMS = 60  # trust boundary: RDKit 3D embedding cost grows fast


def resolve_to_smiles(molecule_input: str) -> str:
    """Accept either a known preset id or a raw SMILES string directly."""
    return MOLECULE_PRESETS.get(molecule_input, molecule_input)


def compute_bond_dissonance(mol) -> np.ndarray:
    """Per-atom mean |bond angle - ideal angle| (sp3 109.5, sp2 120, sp 180), scaled to [0, 1]."""
    conf = mol.GetConformer()
    dissonance = np.zeros(mol.GetNumAtoms(), dtype=np.float32)
    ideal = {
        Chem.HybridizationType.SP3: 109.5,
        Chem.HybridizationType.SP2: 120.0,
        Chem.HybridizationType.SP: 180.0,
    }
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        nbrs = [n.GetIdx() for n in atom.GetNeighbors()]
        if len(nbrs) < 2:
            continue
        target = ideal.get(atom.GetHybridization(), 109.5)
        devs = [
            abs(rdMolTransforms.GetAngleDeg(conf, nbrs[i], idx, nbrs[j]) - target)
            for i in range(len(nbrs)) for j in range(i + 1, len(nbrs))
        ]
        dissonance[idx] = float(np.mean(devs))
    peak = dissonance.max()
    return dissonance / peak if peak > 0 else dissonance


def compute_chirality(mol) -> np.ndarray:
    """R -> +1.0, S -> -1.0, no stereocenter -> 0.0."""
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    out = np.zeros(mol.GetNumAtoms(), dtype=np.float32)
    for atom in mol.GetAtoms():
        if atom.HasProp('_CIPCode'):
            out[atom.GetIdx()] = 1.0 if atom.GetProp('_CIPCode') == 'R' else -1.0
    return out


@lru_cache(maxsize=64)
def get_molecule_features(molecule_input: str):
    """molecule string -> (dissonance_array, chirality_array), one value per atom (H included)."""
    smiles = resolve_to_smiles(molecule_input.strip())
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse molecule input: {molecule_input!r}")
    if mol.GetNumHeavyAtoms() > MAX_HEAVY_ATOMS:
        raise ValueError(f"Molecule too large (max {MAX_HEAVY_ATOMS} heavy atoms)")

    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=42) != 0:
        raise ValueError(f"RDKit could not generate 3D coordinates for {molecule_input!r}")
    AllChem.MMFFOptimizeMolecule(mol)

    return compute_bond_dissonance(mol), compute_chirality(mol)
