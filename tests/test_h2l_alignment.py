import json

import numpy as np

from scripts.realign_h2l import build_alignment, migrate_npz


def test_build_alignment_recovers_a_permutation():
    human = ["alpha text", "beta text", "gamma text"]
    source_lines = ["gamma text", "alpha text", "beta text"]

    alignment = build_alignment(human, source_lines)

    assert [row["hf_index"] for row in alignment] == [2, 0, 1]
    assert {row["match_method"] for row in alignment} == {"exact"}


def test_migrate_npz_reorders_machine_arrays_and_ids(tmp_path):
    source_path = tmp_path / "source.npz"
    target_path = tmp_path / "target.npz"
    metadata = {
        "target_regime": "h2l",
        "no_human": 2,
        "no_machine": 3,
        "machine_texts_path": "legacy_h2l.txt",
    }
    np.savez(
        source_path,
        true_labels=np.asarray([0, 0, 1, 1, 1]),
        scores=np.asarray([1.0, 2.0, 10.0, 20.0, 30.0]),
        ids=np.asarray(["human::0", "human::1", "h2l::0", "h2l::1", "h2l::2"]),
        metadata=json.dumps(metadata),
    )

    # New HF positions 0,1,2 receive old TXT positions 1,2,0.
    migrate_npz(
        source_path,
        target_path,
        inverse=np.asarray([1, 2, 0]),
        aligned_text_path=tmp_path / "aligned_h2l.txt",
    )

    with np.load(target_path, allow_pickle=False) as migrated:
        assert migrated["scores"].tolist() == [1.0, 2.0, 20.0, 30.0, 10.0]
        assert migrated["ids"].tolist() == [
            "human::0",
            "human::1",
            "h2l::0",
            "h2l::1",
            "h2l::2",
        ]
        migrated_metadata = json.loads(migrated["metadata"].item())
        assert migrated_metadata["alignment_status"] == "canonical_hf_order"
