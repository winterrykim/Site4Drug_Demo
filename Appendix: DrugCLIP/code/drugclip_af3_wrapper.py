from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Iterable, Sequence

import lmdb
import numpy as np
from Bio.PDB.MMCIFParser import MMCIFParser
from rdkit import Chem
from rdkit.Chem import AllChem
import rdkit.RDLogger as RDLogger

RDLogger.DisableLog('rdApp.*')


def parse_site_spec(site_spec: str) -> list[int]:
    residues = set()
    for token in site_spec.replace(';', ',').split(','):
        token = token.strip()
        if not token:
            continue
        if '-' in token:
            start, end = token.split('-', 1)
            start_i = int(start.strip())
            end_i = int(end.strip())
            if end_i < start_i:
                raise ValueError(f'Invalid site range: {token}')
            residues.update(range(start_i, end_i + 1))
        else:
            residues.add(int(token))
    if not residues:
        raise ValueError('No residues parsed from site_spec')
    return sorted(residues)


def _pick_single_path(candidates: Sequence[Path], suffix_hint: str) -> Path:
    if not candidates:
        raise FileNotFoundError(f'No {suffix_hint} found')
    preferred = [p for p in candidates if p.name.endswith(f'model_0.{suffix_hint}')]
    if len(preferred) == 1:
        return preferred[0]
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError('Multiple candidate structure files found. Please keep one structure per directory or adapt the selector.')


def extract_af3_cif(af3_dir: str | os.PathLike, out_dir: str | os.PathLike) -> str:
    af3_dir = Path(af3_dir)
    out_dir = Path(out_dir)
    cif_candidates = sorted(af3_dir.rglob('*.cif'))
    if cif_candidates:
        return str(_pick_single_path(cif_candidates, 'cif'))
    zip_candidates = sorted(af3_dir.rglob('*.zip'))
    if not zip_candidates:
        raise FileNotFoundError(f'No .cif or .zip found under {af3_dir}')
    zip_path = _pick_single_path(zip_candidates, 'zip')
    with zipfile.ZipFile(zip_path, 'r') as zf:
        names = [n for n in zf.namelist() if n.endswith('model_0.cif')]
        if not names:
            names = [n for n in zf.namelist() if n.endswith('.cif')]
        if not names:
            raise FileNotFoundError(f'No .cif found inside {zip_path}')
        if len(names) > 1:
            model0 = [n for n in names if n.endswith('model_0.cif')]
            if len(model0) == 1:
                names = model0
            else:
                raise ValueError(f'Multiple CIF files found inside {zip_path}; please keep one AF3 structure per zip.')
        member = names[0]
        extracted = Path(zf.extract(member, path=out_dir))
    return str(extracted)


def build_pocket_lmdb_from_af3(af3_dir: str | os.PathLike, site_spec: str, pocket_lmdb_path: str | os.PathLike,
                               pocket_name: str | None = None, scratch_dir: str | os.PathLike | None = None) -> dict:
    pocket_lmdb_path = Path(pocket_lmdb_path)
    scratch_dir = Path(scratch_dir or pocket_lmdb_path.parent / 'scratch')
    scratch_dir.mkdir(parents=True, exist_ok=True)
    cif_path = extract_af3_cif(af3_dir, scratch_dir)
    target_residues = set(parse_site_spec(site_spec))
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure('target', cif_path)
    pocket_atoms = []
    pocket_coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                resseq = residue.id[1]
                if resseq in target_residues:
                    for atom in residue:
                        pocket_atoms.append(atom.get_name())
                        pocket_coords.append(list(atom.get_coord().astype(float)))
        break
    if not pocket_atoms:
        raise ValueError(f'No atoms were collected for site {site_spec}. Check residue numbering in the AF3 mmCIF.')
    pocket_data = {
        'pocket': pocket_name or f'{Path(cif_path).stem}_site_{site_spec.replace(",", "_").replace("-", "to")}',
        'pocket_index': 0,
        'pocket_atoms': pocket_atoms,
        'pocket_coordinates': pocket_coords,
    }
    env = lmdb.open(str(pocket_lmdb_path), subdir=False, map_size=1099511627776)
    with env.begin(write=True) as txn:
        txn.put(b'0', pickle.dumps(pocket_data, protocol=5))
    env.close()
    return pocket_data


def _embed_smiles_to_record(name: str, smi: str, num_conf: int = 1, num_threads: int = 4) -> dict | None:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMultipleConfs(
            mol,
            numConfs=num_conf,
            numThreads=num_threads,
            pruneRmsThresh=1.0,
            maxAttempts=10000,
            useRandomCoords=False,
        )
        try:
            AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=num_threads)
        except Exception:
            pass
        mol = Chem.RemoveHs(mol)
    except Exception:
        return None
    if mol.GetNumConformers() == 0:
        return None
    coords = [np.array(mol.GetConformer(i).GetPositions()) for i in range(mol.GetNumConformers())]
    atom_types = [a.GetSymbol() for a in mol.GetAtoms()]
    return {
        'name': name,
        'atoms': atom_types,
        'coordinates': coords,
        'smi': smi,
        'mol': mol,
    }


def _read_ligand_library(ligand_library: str | os.PathLike | Sequence[dict]) -> list[dict]:
    if isinstance(ligand_library, (list, tuple)):
        out = []
        for i, item in enumerate(ligand_library):
            smi = item.get('smi') or item.get('smiles')
            if not smi:
                raise ValueError(f'Missing SMILES at item {i}')
            out.append({'name': item.get('name', f'ligand_{i}'), 'smi': smi})
        return out
    path = Path(ligand_library)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {'.json'}:
        data = json.loads(path.read_text())
        return _read_ligand_library(data)
    if path.suffix.lower() in {'.csv', '.tsv'}:
        sep = ',' if path.suffix.lower() == '.csv' else '\t'
        with path.open() as f:
            reader = csv.DictReader(f, delimiter=sep)
            rows = []
            for i, row in enumerate(reader):
                smi = row.get('smi') or row.get('smiles')
                if not smi:
                    raise ValueError(f'Missing smi/smiles column at row {i + 2}')
                rows.append({'name': row.get('name', f'ligand_{i}'), 'smi': smi})
            return rows
    if path.suffix.lower() in {'.smi', '.ism', '.txt'}:
        rows = []
        with path.open() as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                smi = parts[0]
                name = ' '.join(parts[1:]) if len(parts) > 1 else f'ligand_{i}'
                rows.append({'name': name, 'smi': smi})
        return rows
    raise ValueError(f'Unsupported ligand library format: {path.suffix}')


def build_mols_lmdb(ligand_library: str | os.PathLike | Sequence[dict], mols_lmdb_path: str | os.PathLike,
                    smiles_name_json: str | os.PathLike | None = None, num_conf: int = 1, num_threads: int = 4) -> dict[str, str]:
    ligands = _read_ligand_library(ligand_library)
    processed = []
    smiles_to_name = {}
    for item in ligands:
        rec = _embed_smiles_to_record(item['name'], item['smi'], num_conf=num_conf, num_threads=num_threads)
        if rec is None:
            continue
        processed.append(rec)
        smiles_to_name[rec['smi']] = item['name']
    if not processed:
        raise ValueError('No ligands were successfully converted to 3D conformers')
    env = lmdb.open(str(mols_lmdb_path), subdir=False, map_size=1099511627776)
    with env.begin(write=True) as txn:
        for i, rec in enumerate(processed):
            txn.put(str(i).encode('ascii'), pickle.dumps(rec, protocol=5))
    env.close()
    if smiles_name_json is not None:
        Path(smiles_name_json).write_text(json.dumps(smiles_to_name, ensure_ascii=False, indent=2))
    return smiles_to_name


def run_drugclip_retrieval(drugclip_root: str | os.PathLike, checkpoint_path: str | os.PathLike,
                          pocket_lmdb_path: str | os.PathLike, mols_lmdb_path: str | os.PathLike,
                          emb_dir: str | os.PathLike, data_dir: str | os.PathLike | None = None,
                          results_path: str | os.PathLike | None = None, batch_size: int = 8,
                          max_pocket_atoms: int = 256, cuda_visible_devices: str | None = '0',
                          fp16: bool = True, num_workers: int = 4, python_bin: str | None = None) -> str:
    python_bin = python_bin or sys.executable
    drugclip_root = Path(drugclip_root)
    retrieval_py = drugclip_root / 'unimol' / 'retrieval.py'
    if not retrieval_py.exists():
        raise FileNotFoundError(retrieval_py)
    data_dir = Path(data_dir or (drugclip_root / 'data'))
    results_path = Path(results_path or (Path(emb_dir).parent / 'results'))
    emb_dir = Path(emb_dir)
    emb_dir.mkdir(parents=True, exist_ok=True)
    results_path.mkdir(parents=True, exist_ok=True)
    cmd = [
        python_bin,
        "unimol/retrieval.py",
        "--user-dir", "unimol",
        str(data_dir),
        "--valid-subset", "test",
        "--results-path", str(results_path),
        "--num-workers", str(num_workers),
        "--ddp-backend", "c10d",
        "--batch-size", str(batch_size),
        "--task", "drugclip",
        "--loss", "in_batch_softmax",
        "--arch", "drugclip",
        "--max-pocket-atoms", str(max_pocket_atoms),
        "--seed", "1",
        "--path", str(checkpoint_path),
        "--log-interval", "100",
        "--log-format", "simple",
        "--mol-path", str(mols_lmdb_path),
        "--pocket-path", str(pocket_lmdb_path),
        "--emb-dir", str(emb_dir),
    ]
    if fp16:
        cmd.extend(['--fp16', '--fp16-init-scale', '4', '--fp16-scale-window', '256'])
    env = os.environ.copy()
    if cuda_visible_devices is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(cuda_visible_devices)
    subprocess.run(cmd, cwd=str(drugclip_root), env=env, check=True)
    ranked_path = emb_dir / 'ranked_compounds.txt'
    if not ranked_path.exists():
        raise FileNotFoundError(f'Expected ranking file not found: {ranked_path}')
    return str(ranked_path)


def read_ranked_compounds(ranked_path: str | os.PathLike, smiles_to_name: dict[str, str] | None = None, top_k: int | None = None) -> list[dict]:
    rows = []
    with Path(ranked_path).open() as f:
        for i, line in enumerate(f):
            line = line.rstrip('\n')
            if not line:
                continue
            smi, score = line.split('\t', 1)
            rows.append({
                'rank': i + 1,
                'name': (smiles_to_name or {}).get(smi, smi),
                'smi': smi,
                'score': float(score),
            })
            if top_k is not None and len(rows) >= top_k:
                break
    return rows


def recommend_ligands_from_af3_dir(af3_dir: str | os.PathLike, site_spec: str,
                                   ligand_library: str | os.PathLike | Sequence[dict],
                                   drugclip_root: str | os.PathLike, checkpoint_path: str | os.PathLike,
                                   work_dir: str | os.PathLike, data_dir: str | os.PathLike | None = None,
                                   target_name: str | None = None, top_k: int = 20,
                                   batch_size: int = 8, max_pocket_atoms: int = 256,
                                   cuda_visible_devices: str | None = '0', fp16: bool = True,
                                   num_workers: int = 4, num_conf: int = 1) -> list[dict]:
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    pocket_lmdb = work_dir / 'pocket.lmdb'
    mols_lmdb = work_dir / 'mols.lmdb'
    smiles_map_json = work_dir / 'smiles_to_name.json'
    emb_dir = work_dir / 'emb'
    build_pocket_lmdb_from_af3(
        af3_dir=af3_dir,
        site_spec=site_spec,
        pocket_lmdb_path=pocket_lmdb,
        pocket_name=target_name,
        scratch_dir=work_dir / 'scratch',
    )
    smiles_to_name = build_mols_lmdb(
        ligand_library=ligand_library,
        mols_lmdb_path=mols_lmdb,
        smiles_name_json=smiles_map_json,
        num_conf=num_conf,
        num_threads=num_workers,
    )
    ranked_path = run_drugclip_retrieval(
        drugclip_root=drugclip_root,
        checkpoint_path=checkpoint_path,
        pocket_lmdb_path=pocket_lmdb,
        mols_lmdb_path=mols_lmdb,
        emb_dir=emb_dir,
        data_dir=data_dir,
        results_path=work_dir / 'results',
        batch_size=batch_size,
        max_pocket_atoms=max_pocket_atoms,
        cuda_visible_devices=cuda_visible_devices,
        fp16=fp16,
        num_workers=num_workers,
    )
    return read_ranked_compounds(ranked_path, smiles_to_name=smiles_to_name, top_k=top_k)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--af3-dir', required=True)
    parser.add_argument('--site', required=True)
    parser.add_argument('--ligands', required=True)
    parser.add_argument('--drugclip-root', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--data-dir', default=None)
    parser.add_argument('--target-name', default=None)
    parser.add_argument('--top-k', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-pocket-atoms', type=int, default=256)
    parser.add_argument('--cuda-visible-devices', default='0')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--num-conf', type=int, default=1)
    args = parser.parse_args()
    results = recommend_ligands_from_af3_dir(
        af3_dir=args.af3_dir,
        site_spec=args.site,
        ligand_library=args.ligands,
        drugclip_root=args.drugclip_root,
        checkpoint_path=args.checkpoint,
        work_dir=args.work_dir,
        data_dir=args.data_dir,
        target_name=args.target_name,
        top_k=args.top_k,
        batch_size=args.batch_size,
        max_pocket_atoms=args.max_pocket_atoms,
        cuda_visible_devices=None if args.cpu else args.cuda_visible_devices,
        fp16=not args.cpu,
        num_workers=args.num_workers,
        num_conf=args.num_conf,
    )
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
