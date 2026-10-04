#!/usr/bin/env python3
"""Entry point. Translates short names to Hugging Face ids and calls run_pipeline.

Usage:
    python main.py --model roberta-large --dataset sst2 --perturbation-rank 8 \
        --batch-size 64 --learning-rate 1e-6 --epsilon 1e-3 --seed 0 \
        --test-size 256 --epochs 10
"""

import argparse
import sys

from pipeline import run_pipeline


# Short name -> hub id. Sequence-classification checkpoints only.
MODELS = {
    "roberta-base": "roberta-base",
    "roberta-large": "roberta-large",
    "bert-base": "bert-base-uncased",
    "bert-large": "bert-large-uncased",
    "opt-125m": "facebook/opt-125m",
    "opt-350m": "facebook/opt-350m",
    "opt-1.3b": "facebook/opt-1.3b",
    "opt-2.7b": "facebook/opt-2.7b",
    "opt-6.7b": "facebook/opt-6.7b",
    "opt-13b": "facebook/opt-13b",
}


# GLUE's public test split has no labels, so the fixed test pool is sampled
# from validation (validation_matched for MNLI). SNLI and TREC have a
# labeled test split, and that is the pool we sample.
DATASETS = {
    "sst2": {
        "path": "nyu-mll/glue",
        "config": "sst2",
        "train_split": "train",
        "test_split": "validation",
        "text_a": "sentence",
        "text_b": None,
        "label": "label",
        "num_labels": 2,
    },
    
    "sst5": {
    "path": "SetFit/sst5",
    "config": None,
    "train_split": "train",
    "test_split": "test",
    "text_a": "text",
    "text_b": None,
    "label": "label",
    "num_labels": 5,
    },
    
    
    "cola": {
        "path": "nyu-mll/glue",
        "config": "cola",
        "train_split": "train",
        "test_split": "validation",
        "text_a": "sentence",
        "text_b": None,
        "label": "label",
        "num_labels": 2,
    },
    "mrpc": {
        "path": "nyu-mll/glue",
        "config": "mrpc",
        "train_split": "train",
        "test_split": "validation",
        "text_a": "sentence1",
        "text_b": "sentence2",
        "label": "label",
        "num_labels": 2,
    },
    "qqp": {
        "path": "nyu-mll/glue",
        "config": "qqp",
        "train_split": "train",
        "test_split": "validation",
        "text_a": "question1",
        "text_b": "question2",
        "label": "label",
        "num_labels": 2,
    },
    "mnli": {
        "path": "nyu-mll/glue",
        "config": "mnli",
        "train_split": "train",
        "test_split": "validation_matched",
        "text_a": "premise",
        "text_b": "hypothesis",
        "label": "label",
        "num_labels": 3,
    },
    "qnli": {
        "path": "nyu-mll/glue",
        "config": "qnli",
        "train_split": "train",
        "test_split": "validation",
        "text_a": "question",
        "text_b": "sentence",
        "label": "label",
        "num_labels": 2,
    },
    "rte": {
        "path": "nyu-mll/glue",
        "config": "rte",
        "train_split": "train",
        "test_split": "validation",
        "text_a": "sentence1",
        "text_b": "sentence2",
        "label": "label",
        "num_labels": 2,
    },
    "wnli": {
        "path": "nyu-mll/glue",
        "config": "wnli",
        "train_split": "train",
        "test_split": "validation",
        "text_a": "sentence1",
        "text_b": "sentence2",
        "label": "label",
        "num_labels": 2,
    },
    "snli": {
        "path": "stanfordnlp/snli",
        "config": None,
        "train_split": "train",
        "test_split": "test",
        "text_a": "premise",
        "text_b": "hypothesis",
        "label": "label",
        "num_labels": 3,
    },
    "trec": {
        "path": "SetFit/trec",
        "config": None,
        "train_split": "train",
        "test_split": "test",
        "text_a": "text",
        "text_b": None,
        "label": "label",
        "num_labels": 6,
    },
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read the run's flags. --model and --dataset take the short names above."""
    parser = argparse.ArgumentParser(
        description="Resolved low-rank zeroth-order fine-tuning."
    )
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--perturbation-rank", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--k", type=int, default=16, help="Training examples per class.")
    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--test-size", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--output-dir", default="outputs")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Parse flags, swap short names for hub ids, and start the run."""
    args = parse_args(argv)

    # The short names are kept too, so the saved config is readable.
    run_pipeline(
        hf_model_name=MODELS[args.model],
        friendly_model=args.model,
        dataset_spec=DATASETS[args.dataset],
        friendly_dataset=args.dataset,
        perturbation_rank=args.perturbation_rank,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        k=args.k,
        epsilon=args.epsilon,
        seed=args.seed,
        test_size=args.test_size,
        epochs=args.epochs,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    # A bad flag value prints one line and exits with code 2, like argparse.
    try:
        main()
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
