"""Gradient-informed directions built from the low-rank factors.

Each matrix gets Gaussian factors A and B. The SVD of their product
splits it into orthonormal pieces. Component i of the global direction stacks
alpha_ell * u_ell,i v_ell,i^T across the matrices that have that component.
Dividing by the stacked length makes the queried direction unit length,
which is what the two-point formula assumes.

This file never runs the model. When a loss is needed, the caller passes
in a function that computes it (see forward.zo_step).
"""

import math
from dataclasses import dataclass

import torch

import constants


@dataclass
class MatrixLayer:
    """One weight matrix in the zeroth-order update.

    Built once by collect_matrices and reused on every step. Only the
    tensor behind `param` changes during training.
    """

    name: str
    param: torch.nn.Parameter
    # sqrt(d_ell / d): this matrix's share of a unit-length global direction.
    alpha: float
    rows: int
    cols: int
    # min(requested rank, rows, cols). A thin matrix cannot hold more
    # orthonormal rank-1 pieces than its smaller side.
    rank_cap: int


def svd_factors(factor_a: torch.Tensor, factor_b: torch.Tensor, rel_tol: float):
    """Return U (rows x r_num), V (cols x r_num), and r_num.

    A B^T = U diag(s) V^T for the kept components. The pieces used later
    are the singular vectors, not the singular values, so each u v^T has
    Frobenius norm 1. Components whose singular value is numerical junk
    are dropped, so r_num can be smaller than the width of A and B.
    """
    # Thin SVD of each factor. Cheap: both are tall with only r columns,
    # so the full rows x cols product is never formed.
    left_u, left_s, left_vh = torch.linalg.svd(factor_a, full_matrices=False)
    right_u, right_s, right_vh = torch.linalg.svd(factor_b, full_matrices=False)

    # M = Σ_A V_A^T V_B Σ_B, so A B^T = U_A M U_B^T.
    middle = (left_s[:, None] * left_vh) @ (right_vh.transpose(0, 1) * right_s[None, :])
    mid_u, mid_s, mid_vh = torch.linalg.svd(middle, full_matrices=False)

    # Rotate the factor bases by M's singular vectors and keep only the
    # components with a real singular value.
    kept = _leading_singular_count(mid_s, rel_tol)
    direction_u = left_u @ mid_u[:, :kept]
    direction_v = right_u @ mid_vh.transpose(0, 1)[:, :kept]
    return direction_u, direction_v, kept


def _leading_singular_count(singular_values: torch.Tensor, rel_tol: float) -> int:
    """Count leading singular values above rel_tol times the largest one.

    torch.linalg.svd sorts singular values in descending order, so the
    first value at or below the cutoff ends the count.
    """
    if singular_values.numel() == 0:
        return 0
    largest = singular_values[0].abs()
    if largest.item() == 0.0:
        return 0
    limit = rel_tol * largest
    kept = 0
    for value in singular_values:
        if value.abs() <= limit:
            break
        kept += 1
    return kept


def collect_matrices(model: torch.nn.Module, requested_rank: int):
    """Select the matrices the update is allowed to move.

    d is the sum of their entries, not the parameter count of the whole
    model. A matrix thinner than the requested rank still participates:
    it only receives its leading rank_cap components.

    Returns (included layers, d, excluded parameters with a reason).
    """
    included = []
    excluded = []

    # Keep trainable 2-D weights. Vectors (biases, LayerNorm) cannot be
    # written as a u v^T product, so they stay at their initial values.
    for name, param in model.named_parameters():
        if not param.requires_grad:
            excluded.append({"name": name, "reason": "requires_grad is False"})
            continue
        if param.ndim != 2:
            excluded.append({"name": name, "reason": f"{param.ndim}-dimensional"})
            continue
        rows, cols = param.shape
        rank_cap = min(requested_rank, rows, cols)
        if rank_cap < 1:
            excluded.append({"name": name, "reason": "empty matrix"})
            continue
        included.append(
            MatrixLayer(
                name=name,
                param=param,
                alpha=0.0,
                rows=int(rows),
                cols=int(cols),
                rank_cap=int(rank_cap),
            )
        )

    if not included:
        raise RuntimeError("No trainable matrices to perturb.")

    # d needs every included matrix, so alpha is filled in on a second pass.
    total = sum(layer.rows * layer.cols for layer in included)
    for layer in included:
        # alpha_ell = sqrt(d_ell / d) gives each full direction unit length
        # when every matrix contributes a piece.
        layer.alpha = math.sqrt((layer.rows * layer.cols) / total)
    return included, total, excluded


def sample_frames(layers: list[MatrixLayer], step_seed: int, rel_tol: float):
    """Draw A, B ~ N(0, I) for every matrix from one step seed.

    One stored seed regenerates the whole structured perturbation.
    Layers are drawn in parameter order, so the stream stays aligned
    with the model.

    Returns one (U, V, r_num) frame per layer.
    """
    # A private CPU generator: same numbers on any device, and the global
    # torch seed is left alone.
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(step_seed))

    frames = []
    for layer in layers:
        factor_a = torch.randn(layer.rows, layer.rank_cap, generator=generator)
        factor_b = torch.randn(layer.cols, layer.rank_cap, generator=generator)
        frames.append(svd_factors(factor_a, factor_b, rel_tol))
    return frames


def direction_norms(layers: list[MatrixLayer], frames) -> list[float]:
    """Length of component i before it is rescaled to a unit direction.

    r' is the largest rank any matrix returned. A thin matrix simply
    does not contribute to the higher components.

    The length of the returned list is r'.
    """
    # r' = the widest frame on this draw.
    realized = 0
    for _, _, rank_num in frames:
        realized = max(realized, rank_num)

    # Each piece u v^T has norm 1, so component i has squared length
    # equal to the sum of alpha^2 over the matrices that carry it.
    norms = []
    for index in range(realized):
        energy = 0.0
        for layer, (_, _, rank_num) in zip(layers, frames):
            if index < rank_num:
                energy += layer.alpha * layer.alpha
        norms.append(math.sqrt(energy))
    return norms


def accumulate_direction(layers, frames, index: int, coefficient: float, norm: float):
    """Add coefficient * q_index to every matrix that carries component index.

    This is the only function that writes to the weights. Probes, the
    restore after a probe, and the final update all go through it.
    """
    for layer, (direction_u, direction_v, rank_num) in zip(layers, frames):
        # A thin matrix has no piece for this component.
        if index >= rank_num:
            continue

        # Frames are built on CPU; move this one column pair to the weight.
        column_u = direction_u[:, index].to(device=layer.param.device, dtype=layer.param.dtype)
        column_v = direction_v[:, index].to(device=layer.param.device, dtype=layer.param.dtype)

        # alpha / norm turns this matrix's piece into its share of unit q_i.
        layer.param.data.add_(
            torch.outer(column_u, column_v),
            alpha=coefficient * layer.alpha / norm,
        )


def probe_coefficients(layers, frames, norms, epsilon: float, loss_fn) -> list[float]:
    """Two forwards per direction, all on the loss_fn the caller closed over.

    The caller passes a loss on one fixed batch. The +eps / -2eps / +eps
    cycle puts the weights back where they started before the next direction.

    Returns c_i ≈ q_i^T grad L for each realized direction.
    """
    coefficients = []
    with torch.no_grad():
        for index, norm in enumerate(norms):
            # W + eps q_i
            accumulate_direction(layers, frames, index, epsilon, norm)
            loss_plus = loss_fn()

            # W - eps q_i
            accumulate_direction(layers, frames, index, -2.0 * epsilon, norm)
            loss_minus = loss_fn()

            # Back to W before the next direction is probed.
            accumulate_direction(layers, frames, index, epsilon, norm)

            # Central difference along the unit direction q_i.
            coefficients.append((loss_plus - loss_minus) / (2.0 * epsilon))
    return coefficients


def apply_resolved_update(layers, frames, norms, coefficients, parameter_count: int, learning_rate: float):
    """Step by (d / r') * lr along the sum of c_i q_i.

    Returns the scale actually used. A draw with no surviving direction
    does not move the weights and returns 0.
    """
    # r' is the number of directions that were actually probed.
    realized = len(norms)
    if realized == 0:
        return 0.0
    scale = parameter_count / realized

    # Directions are orthogonal, so adding them one at a time is the same
    # as adding their sum.
    with torch.no_grad():
        for index, (coefficient, norm) in enumerate(zip(coefficients, norms)):
            accumulate_direction(
                layers,
                frames,
                index,
                -scale * learning_rate * coefficient,
                norm,
            )
    return scale


def build_step_directions(layers: list[MatrixLayer], step_seed: int):
    """Factors and unit-length norms for one minibatch.

    Returns (frames, norms). len(norms) is r' for this step.
    """
    frames = sample_frames(layers, step_seed, constants.SINGULAR_VALUE_REL_TOL)
    norms = direction_norms(layers, frames)
    return frames, norms
