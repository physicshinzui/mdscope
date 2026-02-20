from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from Bio.Align import PairwiseAligner
from Bio.SeqUtils import seq1


def residues_from_selection(universe: Any, selection: str) -> list[Any]:
    atoms = universe.select_atoms(selection)
    return list(atoms.residues)


def residue_records(residues: list[Any]) -> list[tuple[int, str, str, int]]:
    return [(int(res.resid), str(res.resname), str(res.segid), int(res.resindex)) for res in residues]


def residue_sequence(records: list[tuple[int, str, str, int]]) -> str:
    return "".join(seq1(resname, custom_map={}, undef_code="X") if len(resname) <= 3 else "X" for _, resname, _, _ in records)


def strict_residue_mapping(
    mobile_records: list[tuple[int, str, str, int]],
    ref_records: list[tuple[int, str, str, int]],
) -> list[tuple[int, int]]:
    ref_by_resid = {(rid, segid): i for i, (rid, _, segid, _) in enumerate(ref_records)}
    mapping: list[tuple[int, int]] = []
    for i, (rid, _, segid, _) in enumerate(mobile_records):
        key = (rid, segid)
        if key in ref_by_resid:
            mapping.append((i, ref_by_resid[key]))
    return mapping


def user_map_mapping(
    mobile_records: list[tuple[int, str, str, int]],
    ref_records: list[tuple[int, str, str, int]],
    map_file: Path,
) -> list[tuple[int, int]]:
    mapping: list[tuple[int, int]] = []
    with map_file.open() as fh:
        reader = csv.DictReader(fh)
        mob_idx = {rid: i for i, (rid, _, _, _) in enumerate(mobile_records)}
        ref_idx = {rid: i for i, (rid, _, _, _) in enumerate(ref_records)}
        for row in reader:
            tr = int(row["target_resid"])
            rr = int(row["ref_resid"])
            if tr in mob_idx and rr in ref_idx:
                mapping.append((mob_idx[tr], ref_idx[rr]))
    return mapping


def align_residue_mapping(
    mobile_records: list[tuple[int, str, str, int]],
    ref_records: list[tuple[int, str, str, int]],
) -> list[tuple[int, int]]:
    mob_seq = residue_sequence(mobile_records)
    ref_seq = residue_sequence(ref_records)

    mapping: list[tuple[int, int]] = []
    if len(mob_seq) == len(ref_seq):
        identity = sum(1 for a, b in zip(mob_seq, ref_seq) if a == b) / max(len(mob_seq), 1)
        if identity >= 0.8:
            return [(i, i) for i in range(len(mob_seq))]

    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -10.0
    aligner.extend_gap_score = -0.5
    aln = aligner.align(mob_seq, ref_seq)[0]
    for mob_block, ref_block in zip(aln.aligned[0], aln.aligned[1]):
        mob_start, mob_end = mob_block
        ref_start, ref_end = ref_block
        for mob_pos, ref_pos in zip(range(mob_start, mob_end), range(ref_start, ref_end)):
            mapping.append((mob_pos, ref_pos))
    return mapping


def build_residue_mapping(
    mobile_universe: Any,
    reference_universe: Any,
    align_selection: str,
    map_mode: str,
    map_file: Path | None = None,
) -> tuple[list[Any], list[Any], list[tuple[int, int]], str]:
    mobile_residues = residues_from_selection(mobile_universe, align_selection)
    ref_residues = residues_from_selection(reference_universe, align_selection)
    mobile_records = residue_records(mobile_residues)
    ref_records = residue_records(ref_residues)

    if map_mode == "strict":
        mapping = strict_residue_mapping(mobile_records, ref_records)
        strategy = "strict"
    elif map_mode == "user_map" and map_file is not None:
        mapping = user_map_mapping(mobile_records, ref_records, map_file)
        strategy = "user_map"
    else:
        mapping = align_residue_mapping(mobile_records, ref_records)
        strategy = "align"
    return mobile_residues, ref_residues, mapping, strategy


def ligand_site_resindices(reference_universe: Any, ligand_selection: str, cutoff: float) -> set[int]:
    site_atoms = reference_universe.select_atoms(f"protein and around {cutoff} ({ligand_selection})")
    return {int(res.resindex) for res in site_atoms.residues}


def filter_mapping_to_reference_resindices(
    mapping: list[tuple[int, int]],
    ref_residues: list[Any],
    ref_resindices: set[int],
) -> list[tuple[int, int]]:
    return [item for item in mapping if int(ref_residues[item[1]].resindex) in ref_resindices]

