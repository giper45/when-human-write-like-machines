RANGE_SHORT = "short"
RANGE_MEDIUM = "medium"
RANGE_LONG = "long"

def add_range_to_dataset(dataset, short_range, medium_range, high_range):
    def _get_range(batch, short_range, medium_range, high_range):
        lengths = []
        ranges = []

        for text in batch["text"]:
            length = len(text.split())
            lengths.append(length)

            if short_range[0] <= length <= short_range[1]:
                ranges.append(RANGE_SHORT)
            elif medium_range[0] <= length <= medium_range[1]:
                ranges.append(RANGE_MEDIUM)
            elif high_range[0] <= length <= high_range[1]:
                ranges.append(RANGE_LONG)
            else:
                ranges.append(None)

        return {
            "length": lengths,
            "range": ranges
        }

    return dataset.map(
        _get_range,
        fn_kwargs={
            "short_range": short_range,
            "medium_range": medium_range,
            "high_range": high_range
        },
        batched=True,
        batch_size=100
    )