import sys
from rdkit import Chem

def canonicalize_smiles(smiles: str) -> str:
    
    if smiles is None:
        return "No smiles Input"
    smiles = smiles.strip()
    if not smiles:
        return "Input is not Smiles Structure"

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "Canonicalized smiles can not be produced"

    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


if __name__ == "__main__":

    if len(sys.argv) > 1:
        inp = " ".join(sys.argv[1:]).strip()
    else:
        inp = sys.stdin.read().strip()
    print(canonicalize_smiles(inp))
