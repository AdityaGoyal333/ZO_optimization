"""Run the resolved low-rank zeroth-order loop.

Data pools live in data.py. Directions, including the SVD, live in
directions.py. The model forwards live in forward.py. This file only
wires those pieces together and writes the metrics.

Outputs, rewritten every FLUSH_EVERY updates. The file name contains the
run settings, so two jobs that share --output-dir do not overwrite each other:
    <model>_<dataset>_rank<r>_bs<b>_lr<lr>_k<k>_eps<eps>_seed<s>_test<n>_ep<e>_metrics.pkl
    <model>_<dataset>_rank<r>_bs<b>_lr<lr>_k<k>_eps<eps>_seed<s>_test<n>_ep<e>_checkpoint.pt
"""

import os
import pickle
from pathlib import Path

import numpy as np
import torch

import constants
from data import (
    load_hf_model,
    load_hf_splits,
    load_hf_tokenizer,
    make_loader,
    sample_test_rows,
    sample_training_rows,
)
from directions import collect_matrices
from forward import evaluate, move_batch, zo_step


def run_pipeline(
    hf_model_name: str,
    friendly_model: str,
    dataset_spec: dict,
    friendly_dataset: str,
    perturbation_rank: int,
    batch_size: int,
    learning_rate: float,
    k: int,
    epsilon: float,
    seed: int,
    test_size: int,
    epochs: int,
    output_dir: str,
):
    """Train on one fixed few-shot pool and periodically score a fixed test pool."""
    _validate_run_args(perturbation_rank, batch_size, learning_rate, k, epsilon, test_size, epochs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"model: {friendly_model} -> {hf_model_name}")
    print(
        f"dataset: {friendly_dataset} -> {dataset_spec['path']}"
        + (f"/{dataset_spec['config']}" if dataset_spec["config"] else "")
    )

    # Separate streams so drawing perturbation seeds cannot change the pools.
    data_rng = np.random.default_rng(seed)
    perturb_rng = np.random.default_rng(seed)

    # --- Fixed pools: sampled once, reused for the whole run. ---
    train_split, test_split = load_hf_splits(dataset_spec)
    train_rows = sample_training_rows(train_split, dataset_spec, k, data_rng)
    test_rows = sample_test_rows(test_split, dataset_spec, test_size, data_rng)
    print(f"fixed training pool: {len(train_rows)} examples")
    print(f"fixed test pool: {len(test_rows)} examples")
    if batch_size > len(train_rows):
        print(
            f"requested batch size {batch_size} is larger than the training pool "
            f"{len(train_rows)}; each step uses {len(train_rows)} examples"
        )

    # --- Model ---
    tokenizer = load_hf_tokenizer(hf_model_name)
    model = load_hf_model(hf_model_name, dataset_spec["num_labels"], tokenizer)
    model.to(device)
    # Dropout would make the plus and minus probes two different functions.
    model.eval()

    # --- Which matrices move, and d ---
    layers, parameter_count, excluded = collect_matrices(model, perturbation_rank)
    # No backward pass is ever run, so autograd bookkeeping is turned off.
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    thin = [layer.name for layer in layers if layer.rank_cap < perturbation_rank]
    print(f"d (participating matrix entries): {parameter_count}")
    print(f"participating matrices: {len(layers)}")
    print(f"excluded parameters: {len(excluded)}")
    if thin:
        print(
            "matrices thinner than the requested rank, "
            f"capped at their smaller side: {thin}"
        )

    train_loader = make_loader(train_rows, tokenizer, batch_size, shuffle=True)
    test_loader = make_loader(test_rows, tokenizer, batch_size, shuffle=False)

    # --- Metrics: the run config, then one record per update and per eval ---
    output_path = Path(output_dir)
    metrics = {
        "config": {
            "friendly_model": friendly_model,
            "hf_model_name": hf_model_name,
            "friendly_dataset": friendly_dataset,
            "hf_dataset_path": dataset_spec["path"],
            "hf_dataset_config": dataset_spec["config"],
            "train_split": dataset_spec["train_split"],
            "test_split": dataset_spec["test_split"],
            "perturbation_rank": perturbation_rank,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "k": k,
            "epsilon": epsilon,
            "seed": seed,
            "test_size_requested": test_size,
            "epochs": epochs,
            "d": parameter_count,
            "dataloader_seed": constants.DATALOADER_SEED,
            "train_size": len(train_rows),
            "test_size_actual": len(test_rows),
            "participating": [
                {
                    "name": layer.name,
                    "shape": [layer.rows, layer.cols],
                    "rank_cap": layer.rank_cap,
                    "alpha": layer.alpha,
                }
                for layer in layers
            ],
            "excluded": excluded,
            "flush_every": constants.FLUSH_EVERY,
            "eval_every": constants.EVAL_EVERY,
            "scale_rule": "d / realized_directions",
        },
        "updates": [],
        "evals": [],
    }
    # Write once before training so the config exists even if step 1 fails.
    flush_outputs(metrics, model, output_path)

    # --- Training loop ---
    forward_total = 0
    update = 0
    try:
        for epoch in range(epochs):
            for step, batch in enumerate(train_loader):
                batch = move_batch(batch, device)

                # One new perturbation seed per update, as in FZOO.
                step_seed = int(perturb_rng.integers(0, constants.PERTURBATION_SEED_MODULUS))
                record = zo_step(
                    model,
                    batch,
                    layers,
                    parameter_count,
                    perturbation_rank,
                    epsilon,
                    learning_rate,
                    step_seed,
                )

                # Add the loop position and the running probe count.
                update += 1
                forward_total += record["n_forward_passes"]
                record["update"] = update
                record["epoch"] = epoch
                record["step_in_epoch"] = step
                record["forward_passes_total"] = forward_total
                metrics["updates"].append(record)

                print(
                    f"update={update} epoch={epoch} step={step} "
                    f"effective_batch={record['effective_batch_size']} "
                    f"directions={record['n_directions']} "
                    f"layer_rank_min={record['min_layer_rank']} "
                    f"layer_rank_max={record['max_layer_rank']} "
                    f"forwards={record['n_forward_passes']} "
                    f"forwards_total={forward_total} "
                    f"scale={record['scale']:.6g} "
                    f"train_loss={record['train_loss']:.6f} "
                    f"train_acc={record['train_accuracy']:.4f}"
                )

                # Score the fixed test pool. These forwards are not counted.
                if update % constants.EVAL_EVERY == 0:
                    test_loss, test_accuracy, test_count = evaluate(model, test_loader, device)
                    evaluation = {
                        "update": update,
                        "test_loss": test_loss,
                        "test_accuracy": test_accuracy,
                        "test_examples": test_count,
                    }
                    metrics["evals"].append(evaluation)
                    print(
                        f"eval update={update} test_loss={test_loss:.6f} "
                        f"test_acc={test_accuracy:.4f} test_examples={test_count}"
                    )

                if update % constants.FLUSH_EVERY == 0:
                    flush_outputs(metrics, model, output_path)
    finally:
        # Also runs on a crash or Ctrl-C, so the last updates are not lost.
        flush_outputs(metrics, model, output_path)

    print(f"finished updates={update} probe_forwards={forward_total}")
    return metrics


def _experiment_stem(config: dict) -> str:
    """File stem for this run. Every setting that distinguishes a job is in it."""

    def token(value) -> str:
        # 1e-4 and 0.0001 must become the same text, or the same run gets two names.
        if isinstance(value, float):
            return f"{value:.0e}".replace("+", "")
        return str(value)

    return (
        f"{config['friendly_model']}_{config['friendly_dataset']}"
        f"_rank{token(config['perturbation_rank'])}"
        f"_bs{token(config['batch_size'])}"
        f"_lr{token(config['learning_rate'])}"
        f"_k{token(config['k'])}"
        f"_eps{token(config['epsilon'])}"
        f"_seed{token(config['seed'])}"
        f"_test{token(config['test_size_requested'])}"
        f"_ep{token(config['epochs'])}"
    )


def flush_outputs(metrics: dict, model: torch.nn.Module, output_dir: Path):
    """Replace this run's metrics and checkpoint files via a temp file.

    A crash mid-write leaves the previous flush intact. The names include
    the experiment, so a parallel job with another dataset or seed writes
    a different pair of files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _experiment_stem(metrics["config"])
    metrics_path = output_dir / f"{stem}_metrics.pkl"
    checkpoint_path = output_dir / f"{stem}_checkpoint.pt"
    _atomic_pickle(metrics_path, metrics)
    _atomic_torch_save(
        checkpoint_path,
        {
            "update": metrics["updates"][-1]["update"] if metrics["updates"] else 0,
            "model": model.state_dict(),
            "config": metrics["config"],
        },
    )
    print(f"flushed {metrics_path.name} and {checkpoint_path.name} to {output_dir}")


def _atomic_pickle(path: Path, payload: dict):
    """Pickle to path.tmp, sync to disk, then rename over path."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    # os.replace is atomic, so readers see the old file or the new one.
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, payload: dict):
    """torch.save to path.tmp, then rename over path."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _validate_run_args(perturbation_rank, batch_size, learning_rate, k, epsilon, test_size, epochs):
    """Fail fast, before any download, on a value that cannot work."""
    if perturbation_rank < 1:
        raise ValueError("perturbation rank must be >= 1")
    if batch_size < 1:
        raise ValueError("batch size must be >= 1")
    if learning_rate <= 0:
        raise ValueError("learning rate must be > 0")
    if k < 1:
        raise ValueError("k must be >= 1")
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    if test_size < 1:
        raise ValueError("test size must be >= 1")
    if epochs < 1:
        raise ValueError("epochs must be >= 1")
