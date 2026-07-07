import random
from collections import defaultdict


RANGE_SHORT = "short"
RANGE_MEDIUM = "medium"
RANGE_LONG = "long"


import random
from collections import defaultdict


RANGE_SHORT = "short"
RANGE_MEDIUM = "medium"
RANGE_LONG = "long"


def stratified_random_sampling_by_range(
    dataset,
    ranges,
    range_column="range",
    n_key="n_per_domain",
    seed=42,
    shuffle=True,
    allow_less_if_not_enough=False,
):
    """
    Stratified random sampling using an existing string column, e.g. 'range'.

    Expected dataset columns:
        - text
        - length
        - range

    Expected ranges YAML/Hydra:

        ranges:
          short:
            range: [150, 220]
            n_per_domain: 200
          medium:
            range: [221, 350]
            n_per_domain: 200
          long:
            range: [351, 500]
            n_per_domain: 200

    Sampling logic:
        - use dataset[range_column] directly
        - sample n_per_domain examples for each configured range label
    """
    rng = random.Random(seed)

    valid_labels = set(ranges.keys())

    filtered_dataset = dataset.filter(
        lambda x: x[range_column] in valid_labels
    )

    strata = defaultdict(list)

    for idx, example in enumerate(filtered_dataset):
        strata[example[range_column]].append(idx)

    sampled_indices = []

    for label, range_cfg in ranges.items():
        n_required = range_cfg[n_key]

        candidate_indices = strata.get(label, [])

        if len(candidate_indices) < n_required:
            if not allow_less_if_not_enough:
                raise ValueError(
                    f"Not enough samples for range='{label}'. "
                    f"Required={n_required}, available={len(candidate_indices)}"
                )

            chosen = candidate_indices
        else:
            chosen = rng.sample(candidate_indices, n_required)

        sampled_indices.extend(chosen)

    if shuffle:
        rng.shuffle(sampled_indices)

    return filtered_dataset.select(sampled_indices)