# chunked_vmap.py

from __future__ import annotations

import functools
from typing import Any, Callable, Sequence, TypeVar

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

F = TypeVar("F", bound=Callable[..., Any])


def _is_ferminet_data_like(x: Any) -> bool:
  # Very lightweight check; matches networks.FermiNetData
  return (
      hasattr(x, "positions")
      and hasattr(x, "spins")
      and hasattr(x, "atoms")
      and hasattr(x, "charges")
  )


def _infer_batch_size_for_arg(arg: Any, ax: Any) -> int | None:
  """Infer batch size contributed by a single argument + in_axes entry.

  Returns None if this arg is not mapped (all axes None).
  """
  if ax is None:
    return None

  # Simple case: whole argument is mapped along integer axis `ax`.
  if isinstance(ax, int):
    axis = int(ax)
    leaves = jtu.tree_leaves(arg)
    if not leaves:
      return None
    return leaves[0].shape[axis]

  # FermiNetData(arg) with FermiNetData axes.
  if _is_ferminet_data_like(arg) and _is_ferminet_data_like(ax):
    sizes = []

    if ax.positions is not None:
      sizes.append(arg.positions.shape[int(ax.positions)])
    if ax.spins is not None:
      sizes.append(arg.spins.shape[int(ax.spins)])
    if ax.atoms is not None:
      sizes.append(arg.atoms.shape[int(ax.atoms)])
    if ax.charges is not None:
      sizes.append(arg.charges.shape[int(ax.charges)])

    if not sizes:
      return None

    size0 = sizes[0]
    for s in sizes[1:]:
      assert s == size0, f"Inconsistent batch size in FermiNetData: {sizes}"
    return size0

  # Fallback: we don't know how to interpret this in_axes; say "no info".
  return None


def _slice_ferminet_arg(arg, ax, start: int, end: int):
  """Slice FermiNetData-like arg along batch dimension, respecting axes."""
  assert _is_ferminet_data_like(arg) and _is_ferminet_data_like(ax)

  def maybe_slice(field_val, field_ax):
    if field_ax is None:
      return field_val
    axis = int(field_ax)
    # We assume axis==0 in your use cases.
    assert axis == 0, "chunked_vmap only supports axis=0 for FermiNetData fields"
    return field_val[start:end]

  positions = maybe_slice(arg.positions, ax.positions)
  spins = maybe_slice(arg.spins, ax.spins)
  atoms = maybe_slice(arg.atoms, ax.atoms)
  charges = maybe_slice(arg.charges, ax.charges)

  # FermiNetData is typically a namedtuple, so _replace should exist.
  if hasattr(arg, "_replace"):
    return arg._replace(
        positions=positions, spins=spins, atoms=atoms, charges=charges
    )

  # Very defensive fallback: reconstruct via type.
  return type(arg)(
      positions=positions, spins=spins, atoms=atoms, charges=charges
  )


def _slice_arg(arg: Any, ax: Any, start: int, end: int) -> Any:
  """Slice a single argument along its mapped batch axis (if any)."""
  if ax is None:
    return arg

  if _is_ferminet_data_like(arg) and _is_ferminet_data_like(ax):
    return _slice_ferminet_arg(arg, ax, start, end)

  if isinstance(ax, int):
    axis = int(ax)

    def _slice_leaf(x):
      idx = jnp.arange(start, end)
      return jnp.take(x, idx, axis=axis)

    return jtu.tree_map(_slice_leaf, arg)

  # Unknown in_axes type; just return arg (best effort).
  return arg


def batched_vmap(
    fn: F,
    *,
    max_batch_size: int,
    in_axes: int | None | Sequence[Any] = 0,
    out_axes: Any = 0,
) -> F:
  """Memory-safe vmap that splits a large batch into smaller chunks.

  Designed as a drop-in for the way you use folx.batched_vmap in make_loss:
    - in_axes is a tuple at top-level (params, keys, data, ...)
    - params has in_axes=None
    - keys / data have in_axes=0 or FermiNetData(positions=0, ...)
    - all outputs are mapped along leading axis (0).

  If max_batch_size <= 0, just returns a plain jax.vmap(fn, ...).
  """

  @functools.wraps(fn)
  def mapped_fn(*args, **kwargs):
    wrapped_fn = functools.partial(fn, **kwargs)

    # No chunking: fall back to vanilla vmap.
    if max_batch_size is None or max_batch_size <= 0:
      return jax.vmap(wrapped_fn, in_axes=in_axes, out_axes=out_axes)(*args)

    if not isinstance(in_axes, (tuple, list)):
      # For general fancy in_axes, just use vmap.
      return jax.vmap(wrapped_fn, in_axes=in_axes, out_axes=out_axes)(*args)

    in_axes_seq = tuple(in_axes)
    if len(in_axes_seq) != len(args):
      raise ValueError(
          f"in_axes length {len(in_axes_seq)} != number of args {len(args)}"
      )

    # ------------------------------------------------------------------
    # Infer global batch_size from mapped args.
    # ------------------------------------------------------------------
    batch_size: int | None = None
    for arg, ax in zip(args, in_axes_seq):
      size = _infer_batch_size_for_arg(arg, ax)
      if size is None:
        continue
      if batch_size is None:
        batch_size = size
      else:
        assert (
            batch_size == size
        ), f"Inconsistent batch sizes across args: {batch_size} vs {size}"

    # Nothing is actually mapped: just call fn once.
    if batch_size is None:
      return wrapped_fn(*args)

    # If batch fits in one go, use ordinary vmap.
    if batch_size <= max_batch_size:
      return jax.vmap(wrapped_fn, in_axes=in_axes_seq, out_axes=out_axes)(
          *args
      )

    # ------------------------------------------------------------------
    # Chunked path: manually slice mapped arguments and run vmap per chunk.
    # ------------------------------------------------------------------
    chunk_outputs = []
    start = 0

    while start < batch_size:
      end = min(start + max_batch_size, batch_size)

      chunk_args = [
          _slice_arg(arg, ax, start, end)
          for arg, ax in zip(args, in_axes_seq)
      ]

      chunk_out = jax.vmap(
          wrapped_fn, in_axes=in_axes_seq, out_axes=out_axes
      )(*chunk_args)
      chunk_outputs.append(chunk_out)

      start = end

    # Concatenate outputs from all chunks along leading dimension (axis 0).
    out0 = chunk_outputs[0]
    out_tree = jtu.tree_structure(out0)
    leaves_per_chunk = [out_tree.flatten_up_to(o) for o in chunk_outputs]

    concatenated_leaves = []
    for leaf_idx in range(len(leaves_per_chunk[0])):
      leaf_chunks = [lc[leaf_idx] for lc in leaves_per_chunk]
      concatenated_leaves.append(jnp.concatenate(leaf_chunks, axis=0))

    return out_tree.unflatten(concatenated_leaves)

  return mapped_fn  # type: ignore[return-value]