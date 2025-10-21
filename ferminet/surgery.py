# Copyright Max Geier (MIT) 2025
#
# Based on Google Deepmind's FermiNet public github.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import functools
import importlib
import os
import time
from typing import Optional, Mapping, Sequence, Tuple, Union

from absl import logging
import chex
from ferminet import checkpoint
from ferminet import constants
from ferminet import curvature_tags_and_blocks
from ferminet import envelopes
from ferminet import hamiltonian
from ferminet import loss as qmc_loss_functions
from ferminet import mcmc
from ferminet import networks
from ferminet import observables
from ferminet import pretrain
from ferminet import psiformer
from ferminet.utils import statistics
from ferminet.utils import system
from ferminet.utils import utils
from ferminet.utils import writers
from ferminet.utils import jax_utils

import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
import ml_collections
import numpy as np
from typing_extensions import Protocol
from jax import tree_util

import copy

def clone_pytree(tree):
    """
    Return a PyTree whose leaves are fresh, independent copies.

    • Handles nested dicts / lists / tuples out-of-the-box.
    • Keeps shapes & dtypes unchanged.
    • Works on both JAX DeviceArrays and NumPy ndarrays.
    • Scalars are passed through unchanged (they are immutable anyway).
    """

    def _clone_leaf(x):
        # ---- JAX DeviceArray (or jax.Array from ≥0.4.25) --------------------
        if isinstance(x, jax.Array):
            # Fast, device-to-device copy.  copy=True needs jax ≥ 0.4.25;
            # if you are on an older version just write `x + 0` instead.
            return jnp.array(x, copy=True)

        # ---- NumPy ndarray (still in host memory) ---------------------------
        if isinstance(x, np.ndarray):
            return jax.device_put(x.copy())

        # ---- Anything else (Python scalar, bool, None, etc.) ----------------
        return x

    return tree_util.tree_map(_clone_leaf, tree)

def resize_blocks(
    donor: jax.Array,
    N_old: int,
    d_old: int,
    N_new: int,
    d_new: int,
    key: chex.PRNGKey,
    rng_fn=jax.random.normal,        # or jax.random.uniform etc.
) -> jax.Array:
    """
    donor  : shape (L, d_old*N_old) –– flattened blocks of length N_old
    returns: shape (L, d_new*N_new) –– same block-layout, possibly padded with random values
    """
    L = donor.shape[0]

    # 1. restore the logical 3-tensor (L, d_old, N_old)
    donor3 = donor.reshape(L, d_old, N_old)

    # 2. how much can we copy?
    d_copy = min(d_old, d_new)
    N_copy = min(N_old, N_new)

    # 3. allocate recipient and fill *all* entries with random numbers first
    recip3 = rng_fn(key, shape=(L, d_new, N_new), dtype=donor.dtype) / jnp.sqrt(float(L))

    # 4. overwrite the region that actually exists in the donor
    recip3 = recip3.at[:, :d_copy, :N_copy].set(donor3[:, :d_copy, :N_copy])

    # 5. flatten back to 2-tensor (L, d_new*N_new) with the same block ordering
    return recip3.reshape(L, d_new * N_new)

def transfer_initialization(
      flatten_num_devices: bool,
      donor_filename: str, 
      recipient_params: dict,
      host_batch_size: int,
      prng_key: chex.PRNGKey,
      ignore_batch: bool = False,
      modifications: list = [],
      modifications_kwargs: dict = {},
):
  """ Loads weights from donor_filename and transfers them to fit desired network 
   architecture by applyting the recipient_modifications.  """
  del modifications_kwargs, prng_key # placeholders, unused
   
   # load donor data
  if flatten_num_devices is True: 
    (donor_t_init,
      donor_data,
      donor_params,
      opt_state_ckpt,
      mcmc_width_ckpt,
      density_state_ckpt) = checkpoint.restore_no_batch_check(
          donor_filename, host_batch_size)
    donor_params = jax.tree.map(lambda x: x[0], donor_params)
    for attr_name in dir(donor_data):
      # Skip special attributes (those starting with '__')
      if attr_name.startswith("__"):
          continue
      
      attr_value = getattr(donor_data, attr_name)
      
      # Check if the attribute is a JAX array and has at least 2 dimensions
      if isinstance(attr_value, jnp.ndarray) and len(attr_value.shape) >= 2:
          # Compute the new shape
          new_shape = (1, attr_value.shape[0] * attr_value.shape[1], *attr_value.shape[2:])
          
          # Reshape the attribute
          reshaped_value = jnp.reshape(attr_value, new_shape)
          
          # Update the attribute in `data`
          setattr(donor_data, attr_name, reshaped_value)
  elif ignore_batch is True:
    (donor_t_init,
      donor_data,
      donor_params,
      opt_state_ckpt,
      mcmc_width_ckpt,
      density_state_ckpt) = checkpoint.restore_no_batch_check(
          donor_filename, host_batch_size)
  else:
    (donor_t_init,
      donor_data,
      donor_params,
      opt_state_ckpt,
      mcmc_width_ckpt,
      density_state_ckpt) = checkpoint.restore(
          donor_filename, host_batch_size)
    
  # transfer and modify
  if 'orbital-rnd' in modifications:
    for key in recipient_params.keys():
      if key in {'orbital','envelope'}:  
        logging.info(f"transfer initialization with 'orbital-rnd' modification: skipped transfer of '{key}' parameters.")
      else:
        try:
          recipient_params[key] = clone_pytree(donor_params[key])
          logging.info(f"Successful transfer of '{key}' params from donor.")
        except:
          logging.info(f"WARNING: transfer of '{key}' parameter NOT SUCCESSFUL!")
  else:
    recipient_params = clone_pytree(donor_params)

     
  return (donor_t_init,
    donor_data,
    recipient_params,
    opt_state_ckpt,
    mcmc_width_ckpt,
    density_state_ckpt)