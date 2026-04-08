import argparse
import subprocess
from pathlib import Path
import sys
import yaml


def read_fasta(path: str) -> str:
    seq = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                continue
            seq.append(line)
    sequence = "".join(seq).replace(" ", "").upper()
    if not sequence:
        raise ValueError(f"No sequence found in FASTA: {path}")
    valid = set("ACDEFGHIKLMNPQRSTVWY")
    bad = sorted(set(sequence) - valid)
    if bad:
        raise ValueError(
            f"Invalid FASTA characters found: {''.join(bad)}. Only 20 standard amino acids are supported."
        )
    return sequence


def parse_epitope(spec: str, seq_len: int) -> list[int]:
    spec = spec.strip().replace("..", "-")
    if not spec:
        raise ValueError("Epitope specification is empty.")
    positions = set()
    for part in spec.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_s, end_s = token.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if start > end:
                start, end = end, start
            for i in range(start, end + 1):
                positions.add(i)
        else:
            positions.add(int(token))
    if not positions:
        raise ValueError("No epitope residues parsed.")
    if min(positions) < 1 or max(positions) > seq_len:
        raise ValueError(
            f"Epitope positions must be within 1..{seq_len}, but got {min(positions)}..{max(positions)}"
        )
    return sorted(positions)


def make_binding_types(seq_len: int, epitope_positions: list[int]) -> str:
    arr = ["u"] * seq_len
    for p in epitope_positions:
        arr[p - 1] = "B"
    return "".join(arr)


def build_design_spec(
    target_sequence: str,
    epitope_spec: str,
    peptide_length: str,
    target_chain_id: str = "A",
    peptide_chain_id: str = "P",
    cyclic: bool = False,
) -> dict:
    epitope_positions = parse_epitope(epitope_spec, len(target_sequence))
    binding_types = make_binding_types(len(target_sequence), epitope_positions)

    peptide_entity = {
        "protein": {
            "id": peptide_chain_id,
            "sequence": peptide_length,
        }
    }
    if cyclic:
        peptide_entity["protein"]["cyclic"] = True

    spec = {
        "entities": [
            {
                "protein": {
                    "id": target_chain_id,
                    "sequence": target_sequence,
                    "binding_types": binding_types,
                }
            },
            peptide_entity,
        ]
    }
    return spec


def write_yaml(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def run_command(cmd: list[str]) -> None:
    print("\n[RUN]", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a minimal BoltzGen design YAML from FASTA + epitope and run peptide design."
    )
    parser.add_argument("--fasta", required=True, help="Target FASTA file")
    parser.add_argument(
        "--epitope",
        required=True,
        help='1-based residue positions to target, e.g. "25-40" or "25,27,30-35"',
    )
    parser.add_argument("--outdir", required=True, help="BoltzGen output directory")
    parser.add_argument(
        "--yaml-path",
        default=None,
        help="Optional path for generated design YAML. Default: <outdir>/design_from_fasta_epitope.yaml",
    )
    parser.add_argument(
        "--peptide-length",
        default="12..18",
        help='Designed peptide length or range. Example: "14" or "12..18"',
    )
    parser.add_argument(
        "--protocol",
        default="peptide-anything",
        choices=[
            "protein-anything",
            "peptide-anything",
            "protein-small_molecule",
            "nanobody-anything",
            "antibody-anything",
        ],
    )
    parser.add_argument("--num-designs", type=int, default=100)
    parser.add_argument("--budget", type=int, default=20)
    parser.add_argument("--target-chain-id", default="A")
    parser.add_argument("--peptide-chain-id", default="P")
    parser.add_argument("--cyclic", action="store_true")
    parser.add_argument("--cache", default=None, help="Optional BoltzGen cache directory")
    parser.add_argument("--skip-check", action="store_true")
    parser.add_argument("--reuse", action="store_true")

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    yaml_path = Path(args.yaml_path) if args.yaml_path else outdir / "design_from_fasta_epitope.yaml"

    target_sequence = read_fasta(args.fasta)
    spec = build_design_spec(
        target_sequence=target_sequence,
        epitope_spec=args.epitope,
        peptide_length=args.peptide_length,
        target_chain_id=args.target_chain_id,
        peptide_chain_id=args.peptide_chain_id,
        cyclic=args.cyclic,
    )
    write_yaml(spec, yaml_path)

    print(f"Generated YAML: {yaml_path}")
    print(f"Target length: {len(target_sequence)} aa")
    print(f"Epitope: {args.epitope}")
    print(f"Protocol: {args.protocol}")
    print(f"Num designs: {args.num_designs}")
    print(f"Budget: {args.budget}")

    if not args.skip_check:
        check_cmd = ["boltzgen", "check", str(yaml_path)]
        if args.cache:
            check_cmd += ["--cache", args.cache]
        run_command(check_cmd)

    run_cmd = [
        "boltzgen",
        "run",
        str(yaml_path),
        "--output",
        str(outdir),
        "--protocol",
        args.protocol,
        "--num_designs",
        str(args.num_designs),
        "--budget",
        str(args.budget),
    ]
    if args.cache:
        run_cmd += ["--cache", args.cache]
    if args.reuse:
        run_cmd += ["--reuse"]
    run_command(run_cmd)

    print("\nDone.")
    print(f"Results directory: {outdir.resolve()}")
    print(f"Generated design spec: {yaml_path.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
