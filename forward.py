"""Model forwards for one zeroth-order step and for the test set.

Probe forwards estimate the directional derivatives. The forward after
the update, and the periodic test pass, are logged and are not counted
as probes.

This is the only file that calls model(**batch). The directions it uses
come from directions.py.
"""

import torch

from directions import apply_resolved_update, build_step_directions, probe_coefficients


def forward_loss(model, batch) -> float:
    """Mean loss of this batch. The caller decides whether it is a probe."""
    outputs = model(**batch)
    return float(outputs.loss.item())


def forward_loss_and_accuracy(model, batch) -> tuple[float, float]:
    """Mean loss and fraction correct. Used after the weights have moved."""
    outputs = model(**batch)
    predictions = outputs.logits.argmax(dim=-1)
    accuracy = float((predictions == batch["labels"]).float().mean().item())
    return float(outputs.loss.item()), accuracy


def zo_step(model, batch, layers, parameter_count, requested_rank, epsilon, learning_rate, step_seed):
    """One minibatch update. Counted forwards are the 2 r' probes.

    The loss returned here is a separate forward on the updated weights
    and the same batch.

    Returns one record for metrics["updates"].
    """
    # 1. Build this step's directions from its seed.
    frames, norms = build_step_directions(layers, step_seed)
    layer_ranks = [rank_num for _, _, rank_num in frames]

    # 2. Probe every direction on this same batch B, then step once.
    def loss_on_this_batch() -> float:
        return forward_loss(model, batch)

    coefficients = probe_coefficients(layers, frames, norms, epsilon, loss_on_this_batch)
    scale = apply_resolved_update(
        layers, frames, norms, coefficients, parameter_count, learning_rate
    )

    # Logged after the step, on the same batch. Not a probe forward.
    with torch.no_grad():
        train_loss, accuracy = forward_loss_and_accuracy(model, batch)

    # 3. Debug record. n_forward_passes excludes the logging forward above.
    return {
        "train_loss": train_loss,
        "train_accuracy": accuracy,
        "n_directions": len(norms),
        "n_forward_passes": 2 * len(norms),
        "requested_rank": requested_rank,
        "scale": scale,
        "parameter_count_d": parameter_count,
        "min_layer_rank": min(layer_ranks) if layer_ranks else 0,
        "max_layer_rank": max(layer_ranks) if layer_ranks else 0,
        "zo_random_seed": int(step_seed),
        "effective_batch_size": int(batch["labels"].shape[0]),
    }


def evaluate(model, loader, device) -> tuple[float, float, int]:
    """Mean loss and accuracy on a loader. These forwards are not probes.

    Returns (mean loss, accuracy, number of examples scored).
    """
    total_loss = 0.0
    correct = 0
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            outputs = model(**batch)

            # outputs.loss is a batch mean. Weight it by batch size so a
            # short final batch counts per example, not per batch.
            batch_size = int(batch["labels"].shape[0])
            total_loss += float(outputs.loss.item()) * batch_size
            correct += int((outputs.logits.argmax(dim=-1) == batch["labels"]).sum().item())
            count += batch_size
    if count == 0:
        raise RuntimeError("Evaluation loader produced no examples.")
    return total_loss / count, correct / count, count


def move_batch(batch: dict, device: torch.device) -> dict:
    """Move every tensor in a tokenized batch to the model's device."""
    return {key: value.to(device) for key, value in batch.items()}
