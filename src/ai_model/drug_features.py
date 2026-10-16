"""Convert drug SMILES strings into molecular graphs for the GNN drug encoder.

Uses RDKit when it is installed. Without RDKit a crude fallback builds a chain
graph over the SMILES atom tokens so the pipeline still runs end to end.
"""
import re

import numpy as np

MAX_ATOMS = 100
ELEMENTS = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B", "Pt", "other"]
ATOM_DIM = len(ELEMENTS) + 6 + 1 + 5 + 1 + 1  # element, degree, aromatic, Hs, charge, ring

try:
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    HAVE_RDKIT = True
except ImportError:  # pragma: no cover - depends on environment
    HAVE_RDKIT = False


def _one_hot(value, choices):
    vec = [0.0] * len(choices)
    vec[choices.index(value) if value in choices else len(choices) - 1] = 1.0
    return vec


def _rdkit_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    feats = []
    for atom in mol.GetAtoms():
        feats.append(
            _one_hot(atom.GetSymbol(), ELEMENTS)
            + _one_hot(min(atom.GetDegree(), 5), list(range(6)))
            + [float(atom.GetIsAromatic())]
            + _one_hot(min(atom.GetTotalNumHs(), 4), list(range(5)))
            + [float(atom.GetFormalCharge())]
            + [float(atom.IsInRing())]
        )
    edges = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    return np.array(feats, dtype=np.float32), edges


_TOKEN_RE = re.compile(r"Cl|Br|Pt|[BCNOSPFI]|[cnosp]")


def _fallback_graph(smiles):
    tokens = _TOKEN_RE.findall(smiles or "")
    if not tokens:
        tokens = ["C"]
    feats = []
    for tok in tokens:
        symbol = tok.upper() if len(tok) == 1 else tok
        row = _one_hot(symbol, ELEMENTS) + [0.0] * (ATOM_DIM - len(ELEMENTS))
        row[len(ELEMENTS) + 6] = float(tok.islower())  # aromatic flag
        feats.append(row)
    edges = [(i, i + 1) for i in range(len(tokens) - 1)]
    return np.array(feats, dtype=np.float32), edges


def smiles_to_graph(smiles, max_atoms=MAX_ATOMS):
    """Return (x [max_atoms, ATOM_DIM], a_hat [max_atoms, max_atoms], mask [max_atoms]).

    a_hat is the symmetric-normalised adjacency with self loops used by GCN layers.
    """
    graph = _rdkit_graph(smiles) if HAVE_RDKIT and smiles else None
    if graph is None:
        graph = _fallback_graph(smiles)
    feats, edges = graph
    n = min(len(feats), max_atoms)

    x = np.zeros((max_atoms, ATOM_DIM), dtype=np.float32)
    x[:n] = feats[:n]
    adj = np.zeros((max_atoms, max_atoms), dtype=np.float32)
    for i, j in edges:
        if i < n and j < n:
            adj[i, j] = adj[j, i] = 1.0
    adj[np.arange(n), np.arange(n)] = 1.0
    deg = adj.sum(1)
    inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(np.maximum(deg, 1e-9)), 0.0)
    a_hat = adj * inv_sqrt[:, None] * inv_sqrt[None, :]
    mask = np.zeros(max_atoms, dtype=np.float32)
    mask[:n] = 1.0
    return x, a_hat.astype(np.float32), mask


def build_drug_bank(smiles_list, max_atoms=MAX_ATOMS):
    """Stack graphs for a list of SMILES into arrays indexed by drug position."""
    graphs = [smiles_to_graph(s, max_atoms) for s in smiles_list]
    return (
        np.stack([g[0] for g in graphs]),
        np.stack([g[1] for g in graphs]),
        np.stack([g[2] for g in graphs]),
    )
