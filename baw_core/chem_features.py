"""
chem_features.py
-----------------
Turns a molecule into two per-atom numeric arrays using RDKit:

  - bond_dissonance : how much each atom's bond angles deviate from the
                       "ideal" angle for its hybridization (a proxy for
                       geometric strain -- this is the real math behind
                       the "Bond Dissonance" feature).
  - chirality        : a numeric encoding of each atom's stereochemistry
                       (R = +1.0, S = -1.0, non-chiral = 0.0), from
                       RDKit's standard CIP (Cahn-Ingold-Prelog) labels.

RDKit needs a SMILES string (a structural notation: which atoms are
bonded to which), not just a molecular formula like "C8H10N4O2" --
a formula alone doesn't tell you connectivity, so many different
molecules can share the same formula. We keep a small preset table
mapping the formulas from config.json to real SMILES strings.
"""

from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

# Formula -> SMILES presets, matching config.json's molecule_presets
MOLECULE_PRESETS = {
    "H2O": "O",
    "C8H10N4O2": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # caffeine
    "C9H8O4": "CC(=O)OC1=CC=CC=C1C(=O)O",          # aspirin
}


def resolve_to_smiles(molecule_input: str) -> str:
    """Accept either a known formula preset or a raw SMILES string directly."""
    if molecule_input in MOLECULE_PRESETS:
        return MOLECULE_PRESETS[molecule_input]
    return molecule_input


def compute_bond_dissonance(mol) -> np.ndarray:
    """
    Per-atom dissonance = average deviation (in degrees) of this atom's
    bond angles from the ideal angle for its hybridization:
        sp3 ~ 109.5 deg, sp2 ~ 120 deg, sp ~ 180 deg
    Larger deviation = more strain = higher dissonance.
    Returned normalized to roughly [0, 1].
    """
    conf = mol.GetConformer()
    n_atoms = mol.GetNumAtoms()
    dissonance = np.zeros(n_atoms, dtype=np.float32)

    ideal_angle = {
        Chem.HybridizationType.SP3: 109.5,
        Chem.HybridizationType.SP2: 120.0,
        Chem.HybridizationType.SP: 180.0,
    }

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
        if len(neighbors) < 2:
            dissonance[idx] = 0.0
            continue

        target = ideal_angle.get(atom.GetHybridization(), 109.5)
        deviations = []
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                angle = Chem.rdMolTransforms.GetAngleDeg(conf, neighbors[i], idx, neighbors[j])
                deviations.append(abs(angle - target))

        dissonance[idx] = float(np.mean(deviations)) if deviations else 0.0

    max_val = dissonance.max()
    if max_val > 0:
        dissonance = dissonance / max_val
    return dissonance


def compute_chirality(mol) -> np.ndarray:
    """
    Per-atom chirality value from RDKit's CIP assignment:
        R  -> +1.0
        S  -> -1.0
        no stereocenter -> 0.0
    """
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    n_atoms = mol.GetNumAtoms()
    chirality = np.zeros(n_atoms, dtype=np.float32)

    for atom in mol.GetAtoms():
        if atom.HasProp('_CIPCode'):
            chirality[atom.GetIdx()] = 1.0 if atom.GetProp('_CIPCode') == 'R' else -1.0
    return chirality


def get_molecule_features(molecule_input: str):
    """
    Main entry point: molecule string -> (dissonance_array, chirality_array).
    Both arrays have length == number of atoms (hydrogens included,
    since bond-angle geometry needs the full explicit structure).
    """
    smiles = resolve_to_smiles(molecule_input)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse molecule input: {molecule_input!r}")

    mol = Chem.AddHs(mol)                       # add explicit hydrogens
    embed_result = AllChem.EmbedMolecule(mol, randomSeed=42)  # generate 3D coords
    if embed_result != 0:
        raise ValueError(f"RDKit could not generate 3D coordinates for {molecule_input!r}")
    AllChem.MMFFOptimizeMolecule(mol)            # relax to realistic geometry

    dissonance = compute_bond_dissonance(mol)
    chirality = compute_chirality(mol)
    return dissonance, chirality
