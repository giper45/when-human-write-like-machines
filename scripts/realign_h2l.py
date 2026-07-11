"""Recover canonical H2L ordering without modifying the original artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from datasets import load_from_disk
from rapidfuzz import fuzz, process


DATASETS = ("owt", "wp", "xsum")
H2L_SUFFIX = "_h2l.txt"


def normalize_text(text: str) -> str:
    return " ".join(str(text).split())


def text_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def build_alignment(human_texts, source_lines, fuzzy_threshold=0.85):
    """Map each TXT row to one unique Hugging Face row."""
    positions = defaultdict(list)
    normalized_human = [normalize_text(text) for text in human_texts]
    for index, text in enumerate(normalized_human):
        positions[text].append(index)

    used = set()
    rows = []
    for txt_index, source_text in enumerate(source_lines):
        normalized_source = normalize_text(source_text)
        exact = [index for index in positions[normalized_source] if index not in used]
        if len(exact) == 1:
            hf_index = exact[0]
            method = "exact"
            score = 1.0
        else:
            available = {
                index: text
                for index, text in enumerate(normalized_human)
                if index not in used
            }
            matches = process.extract(
                normalized_source,
                available,
                scorer=fuzz.ratio,
                limit=2,
            )
            if not matches:
                raise ValueError(f"No available HF source for TXT row {txt_index}.")
            _, raw_score, hf_index = matches[0]
            score = raw_score / 100.0
            second_score = matches[1][1] / 100.0 if len(matches) > 1 else 0.0
            if score < fuzzy_threshold or score - second_score < 0.10:
                raise ValueError(
                    f"Ambiguous fuzzy match for TXT row {txt_index}: "
                    f"best={score:.3f}, second={second_score:.3f}."
                )
            method = "fuzzy"

        used.add(hf_index)
        rows.append(
            {
                "txt_index": txt_index,
                "hf_index": hf_index,
                "match_method": method,
                "match_score": score,
                "txt_source_sha256": text_hash(source_text),
                "hf_source_sha256": text_hash(human_texts[hf_index]),
            }
        )

    if len(rows) != len(human_texts) or len(used) != len(human_texts):
        raise ValueError(
            "Alignment is not bijective: "
            f"txt={len(rows)}, hf={len(human_texts)}, matched={len(used)}."
        )
    return rows


def write_manifest(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def align_h2l_file(source_path: Path, target_path: Path, alignment) -> None:
    source_lines = source_path.read_text(encoding="utf-8").splitlines()
    if len(source_lines) != len(alignment):
        raise ValueError(
            f"{source_path} has {len(source_lines)} rows; expected {len(alignment)}."
        )
    aligned = [None] * len(alignment)
    for row in alignment:
        aligned[row["hf_index"]] = source_lines[row["txt_index"]]
    if any(text is None for text in aligned):
        raise ValueError(f"Incomplete aligned output for {source_path}.")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text("\n".join(aligned) + "\n", encoding="utf-8")


def migrate_npz(source_path: Path, target_path: Path, inverse, aligned_text_path: Path):
    with np.load(source_path, allow_pickle=False) as source:
        arrays = {key: source[key].copy() for key in source.files}

    metadata = json.loads(arrays["metadata"].item())
    human_count = int(metadata["no_human"])
    machine_count = int(metadata["no_machine"])
    if machine_count != len(inverse):
        raise ValueError(
            f"{source_path}: machine rows={machine_count}, alignment rows={len(inverse)}."
        )
    total_count = human_count + machine_count

    for key, values in list(arrays.items()):
        if key in {"metadata", "ids"} or values.ndim == 0:
            continue
        if values.shape[0] == total_count:
            arrays[key] = np.concatenate(
                [values[:human_count], values[human_count:][inverse]],
                axis=0,
            )

    old_ids = arrays.get("ids")
    human_ids = (
        old_ids[:human_count]
        if old_ids is not None and len(old_ids) == total_count
        else np.asarray([f"human::{index}" for index in range(human_count)])
    )
    machine_ids = np.asarray([f"h2l::{index}" for index in range(machine_count)])
    arrays["ids"] = np.concatenate([human_ids.astype(str), machine_ids])

    metadata["machine_texts_path"] = str(aligned_text_path)
    metadata["alignment_status"] = "canonical_hf_order"
    metadata["alignment_version"] = 1
    arrays["metadata"] = np.asarray(json.dumps(metadata))

    target_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(target_path, **arrays)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated-dir", type=Path, default=Path("generated-texts"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--aligned-generated-dir", type=Path, default=Path("generated-texts-aligned")
    )
    parser.add_argument("--aligned-results-dir", type=Path, default=Path("results-aligned"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for target in (args.aligned_generated_dir, args.aligned_results_dir):
        if target.exists() and not args.overwrite:
            raise FileExistsError(f"{target} already exists; pass --overwrite to replace it.")
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)

    alignments = {}
    for dataset_name in DATASETS:
        sampled_path = args.generated_dir / f"{dataset_name}_sampled"
        source_txt_path = args.generated_dir / f"{dataset_name}_sampled.txt"
        human_texts = list(load_from_disk(str(sampled_path))["text"])
        source_lines = source_txt_path.read_text(encoding="utf-8").splitlines()
        alignment = build_alignment(human_texts, source_lines)
        alignments[dataset_name] = alignment
        write_manifest(
            args.aligned_generated_dir / "manifests" / f"{dataset_name}_h2l_alignment.csv",
            alignment,
        )

        for source_path in sorted(args.generated_dir.glob(f"{dataset_name}_*{H2L_SUFFIX}")):
            target_path = args.aligned_generated_dir / source_path.name
            align_h2l_file(source_path, target_path, alignment)

    for source_path in sorted(args.results_dir.glob("*.npz")):
        target_path = args.aligned_results_dir / source_path.name
        with np.load(source_path, allow_pickle=False) as source:
            metadata = json.loads(source["metadata"].item())
        if metadata.get("target_regime") != "h2l":
            shutil.copy2(source_path, target_path)
            continue

        dataset_name = metadata["dataset_name"]
        alignment = alignments[dataset_name]
        inverse = np.empty(len(alignment), dtype=int)
        for row in alignment:
            inverse[row["hf_index"]] = row["txt_index"]
        aligned_text_path = args.aligned_generated_dir / Path(
            metadata["machine_texts_path"]
        ).name
        migrate_npz(source_path, target_path, inverse, aligned_text_path)

    print(f"Aligned H2L texts: {args.aligned_generated_dir}")
    print(f"Aligned result set: {args.aligned_results_dir}")


if __name__ == "__main__":
    main()
