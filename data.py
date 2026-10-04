"""Fixed train and test pools, and the loaders that batch them.

The pools are sampled once. The training loader may reshuffle that same
pool every epoch. It does not draw new examples.

Also loads the Hugging Face dataset, tokenizer, and model. Those imports
are inside the functions so the math code can be imported without them.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import constants


class ExampleDataset(Dataset):
    """The fixed pool of text examples. Order is shuffled only by the loader.

    Each row is a dict with text_a, label, and text_b for sentence pairs.
    """

    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]


def load_hf_splits(spec: dict):
    """Download the dataset and return (train split, eval split) from spec."""
    from datasets import load_dataset

    # GLUE tasks need a config name; SNLI and TREC do not.
    if spec["config"] is None:
        dataset = load_dataset(spec["path"])
    else:
        dataset = load_dataset(spec["path"], spec["config"])
    return dataset[spec["train_split"]], dataset[spec["test_split"]]


def load_hf_tokenizer(hf_model_name: str):
    """Tokenizer that matches the model checkpoint."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(hf_model_name)


def load_hf_model(hf_model_name: str, num_labels: int, tokenizer):
    """Load a sequence-classification model with num_labels outputs, in fp32."""
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(
        hf_model_name,
        num_labels=num_labels,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )

    # OPT-style models ship without a pad token. Padding is needed to batch,
    # and the model must know the pad id to find each sequence's last token.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    return model


def sample_training_rows(dataset, spec: dict, k: int, rng: np.random.Generator) -> list[dict]:
    """k examples per class, or every example in a class that has fewer than k.

    Indices are sorted so the pool itself does not depend on the order
    the random draw happens to emit. The DataLoader shuffles later.
    """
    # Group row indices by class. Label -1 means unlabeled; skip it.
    labels = _labeled_column(dataset, spec["label"])
    by_class: dict[int, list[int]] = {}
    for index, label in enumerate(labels.tolist()):
        label = int(label)
        if label < 0:
            continue
        by_class.setdefault(label, []).append(index)

    if not by_class:
        raise RuntimeError("Training split has no labeled examples.")

    # Draw k from each class in sorted class order, so one seed always
    # gives the same pool.
    chosen = []
    for label in sorted(by_class):
        pool = np.asarray(by_class[label])
        take = min(k, int(pool.shape[0]))
        if take < k:
            print(
                f"class {label} has {pool.shape[0]} training examples, "
                f"fewer than k={k}; using all {take}"
            )
        picked = rng.choice(pool, size=take, replace=False)
        chosen.extend(int(i) for i in picked)
        print(f"train class {label}: {take} examples")

    chosen.sort()
    return _materialize(dataset, chosen, spec)


def sample_test_rows(dataset, spec: dict, test_size: int, rng: np.random.Generator) -> list[dict]:
    """A fixed random subset of the labeled eval split. Not stratified."""
    # Only labeled rows can be scored.
    labels = _labeled_column(dataset, spec["label"])
    pool = np.asarray([index for index, label in enumerate(labels.tolist()) if int(label) >= 0])
    if pool.size == 0:
        raise RuntimeError(f"Split {spec['test_split']} has no labeled examples.")

    # A request larger than the split falls back to the whole split.
    take = min(test_size, int(pool.size))
    if take < test_size:
        print(
            f"requested test size {test_size} but split {spec['test_split']} "
            f"has {pool.size} labeled examples; using all {take}"
        )
    picked = rng.choice(pool, size=take, replace=False)
    picked.sort()
    print(f"test examples: {take} from split {spec['test_split']}")
    return _materialize(dataset, picked, spec)


def make_collate(tokenizer):
    """Return a collate_fn that tokenizes a list of rows into one batch."""

    def collate(rows: list[dict]) -> dict:
        # Sentence-pair tasks also carry text_b. Single-sentence tasks pass None.
        text_b = [row["text_b"] for row in rows] if "text_b" in rows[0] else None
        encoded = tokenizer(
            [row["text_a"] for row in rows],
            text_pair=text_b,
            padding=True,
            truncation=True,
            max_length=constants.MAX_SEQUENCE_LENGTH,
            return_tensors="pt",
        )
        # Passing labels makes the model return the loss as well as the logits.
        encoded["labels"] = torch.tensor([row["label"] for row in rows], dtype=torch.long)
        return encoded

    return collate


def make_loader(rows: list[dict], tokenizer, batch_size: int, shuffle: bool) -> DataLoader:
    """Batch a fixed pool. Shuffling uses DATALOADER_SEED, not the run seed.

    drop_last is False, so a batch size larger than the pool gives one
    batch of the whole pool.
    """
    # A seeded generator makes the per-epoch order reproducible.
    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(constants.DATALOADER_SEED)
    return DataLoader(
        ExampleDataset(rows),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        generator=generator,
        collate_fn=make_collate(tokenizer),
    )


def _labeled_column(dataset, label_column: str) -> np.ndarray:
    """All labels in a split as a NumPy array."""
    return np.asarray(dataset[label_column])


def _materialize(dataset, indices, spec: dict) -> list[dict]:
    """Copy the chosen rows into plain dicts with the field names spec maps."""
    rows = []
    for index in indices:
        example = dataset[int(index)]
        row = {
            "text_a": example[spec["text_a"]],
            "label": int(example[spec["label"]]),
        }
        if spec["text_b"] is not None:
            row["text_b"] = example[spec["text_b"]]
        rows.append(row)
    return rows
