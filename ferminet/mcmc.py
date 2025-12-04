# # Copyright 2020 DeepMind Technologies Limited.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# # https://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

# """Metropolis-Hastings Monte Carlo.

# NOTE: these functions operate on batches of MCMC configurations and should not
# be vmapped.
# """

# import chex
# from ferminet import constants
# from ferminet import networks
# import jax
# from jax import lax
# from jax import numpy as jnp
# import numpy as np


# def apply_random_lattice_shifts(key: chex.PRNGKey, x1, lattice, move_width = 1):
#     """
#     Shift all electron positions by the same random lattice vector per batch element.
    
#     Args:
#         key: JAX PRNGKey
#         x1: jnp.ndarray of shape (batch_size, N * d), where d = 2
#         lattice: jnp.ndarray of shape (2, 2) representing the lattice vectors as columns
#         move_width: width of lattice shifts

#     Returns:
#         x2: jnp.ndarray of shape (batch_size, N * d)
#     """
#     batch_size = x1.shape[0]
#     dim = lattice.shape[1]  # spatial dimension
#     # print(x1.shape)
#     num_electrons = x1.shape[1] // dim

#     # Step 1: Sample two random integers per batch element
#     key, subkey = jax.random.split(key)
#     normal_samples = move_width*jax.random.normal(key, shape=(batch_size, 2), dtype=jnp.float32)
#     random_ints = jnp.round(normal_samples).astype(jnp.int32)
#     # random_ints = jax.random.randint(subkey, shape=(batch_size, 2), minval=-move_width, maxval=move_width)

#     # Step 2: Compute random lattice shifts: (batch_size, 2)
#     shift_vectors = jnp.einsum('ij,kj->ki', lattice, random_ints)  # (2,2) x (batch_size, 2)ᵀ -> (batch_size, 2)
#     # print(shift_vectors.shape)
#     # Step 3: Tile shift vector across N electrons
#     tiled_shifts = jnp.tile(shift_vectors[:, None, :], (1, num_electrons, 1))  # (batch_size, N, 2)
#     reshaped_shifts = tiled_shifts.reshape((batch_size, num_electrons * dim))    # (batch_size, N*d)

#     # Step 4: Apply shifts
#     x2 = x1 + reshaped_shifts

#     return x2

# def _harmonic_mean(x, atoms):
#   """Calculates the harmonic mean of each electron distance to the nuclei.

#   Args:
#     x: electron positions. Shape (batch, nelectrons, 1, ndim). Note the third
#       dimension is already expanded, which allows for avoiding additional
#       reshapes in the MH algorithm.
#     atoms: atom positions. Shape (natoms, ndim)

#   Returns:
#     Array of shape (batch, nelectrons, 1, 1), where the (i, j, 0, 0) element is
#     the harmonic mean of the distance of the j-th electron of the i-th MCMC
#     configuration to all atoms.
#   """
#   ae = x - atoms[None, ...]
#   r_ae = jnp.linalg.norm(ae, axis=-1, keepdims=True)
#   return 1.0 / jnp.mean(1.0 / r_ae, axis=-2, keepdims=True)


# def _log_prob_gaussian(x, mu, sigma):
#   """Calculates the log probability of Gaussian with diagonal covariance.

#   Args:
#     x: Positions. Shape (batch, nelectron, 1, ndim) - as used in mh_update.
#     mu: means of Gaussian distribution. Same shape as or broadcastable to x.
#     sigma: standard deviation of the distribution. Same shape as or
#       broadcastable to x.

#   Returns:
#     Log probability of Gaussian distribution with shape as required for
#     mh_update - (batch, nelectron, 1, 1).
#   """
#   numer = jnp.sum(-0.5 * ((x - mu)**2) / (sigma**2), axis=[1, 2, 3])
#   denom = x.shape[-1] * jnp.sum(jnp.log(sigma), axis=[1, 2, 3])
#   return numer - denom


# def mh_accept(x1, x2, lp_1, lp_2, ratio, key, num_accepts):
#   """Given state, proposal, and probabilities, execute MH accept/reject step."""
#   key, subkey = jax.random.split(key)
#   rnd = jnp.log(jax.random.uniform(subkey, shape=ratio.shape))
#   cond = ratio > rnd
#   x_new = jnp.where(cond[..., None], x2, x1)
#   lp_new = jnp.where(cond, lp_2, lp_1)
#   num_accepts += jnp.sum(cond)
#   return x_new, key, lp_new, num_accepts


# def mh_update(
#     params: networks.ParamTree,
#     f: networks.LogFermiNetLike,
#     data: networks.FermiNetData,
#     key: chex.PRNGKey,
#     lp_1,
#     num_accepts,
#     stddev=0.02,
#     atoms=None,
#     ndim=3,
#     blocks=1,
#     i=0,
#     enforce_symmetry_by_shift: str = 'none',
#     symmetry_shift_kwargs: dict = {'lattice': None, 'move_width': 1},
# ):
#   """Performs one Metropolis-Hastings step using an all-electron move.

#   Args:
#     params: Wavefuncttion parameters.
#     f: Callable with signature f(params, x) which returns the log of the
#       wavefunction (i.e. the sqaure root of the log probability of x).
#     data: Initial MCMC configurations (batched).
#     key: RNG state.
#     lp_1: log probability of f evaluated at x1 given parameters params.
#     num_accepts: Number of MH move proposals accepted.
#     stddev: width of Gaussian move proposal.
#     atoms: If not None, atom positions. Shape (natoms, 3). If present, then the
#       Metropolis-Hastings move proposals are drawn from a Gaussian distribution,
#       N(0, (h_i stddev)^2), where h_i is the harmonic mean of distances between
#       the i-th electron and the atoms, otherwise the move proposal drawn from
#       N(0, stddev^2).
#     ndim: dimensionality of system.
#     blocks: Ignored.
#     i: Ignored.

#   Returns:
#     (x, key, lp, num_accepts), where:
#       x: Updated MCMC configurations.
#       key: RNG state.
#       lp: log probability of f evaluated at x.
#       num_accepts: update running total of number of accepted MH moves.
#   """
#   del i, blocks  # electron index ignored for all-electron moves
#   key, subkey = jax.random.split(key)
#   x1 = data.positions
#   if atoms is None:  # symmetric proposal, same stddev everywhere
#     x2 = x1 + stddev * jax.random.normal(subkey, shape=x1.shape)  # proposal
#     if enforce_symmetry_by_shift == 'lattice':
#       x2 = apply_random_lattice_shifts(key, x2, symmetry_shift_kwargs['lattice'], symmetry_shift_kwargs['move_width'])
#     lp_2 = 2.0 * f(
#         params, x2, data.spins, data.atoms, data.charges
#     )  # log prob of proposal
#     ratio = lp_2 - lp_1
#   else:  # asymmetric proposal, stddev propto harmonic mean of nuclear distances
#     n = x1.shape[0]
#     x1 = jnp.reshape(x1, [n, -1, 1, ndim])
#     hmean1 = _harmonic_mean(x1, atoms)  # harmonic mean of distances to nuclei

#     x2 = x1 + stddev * hmean1 * jax.random.normal(subkey, shape=x1.shape)
#     lp_2 = 2.0 * f(
#         params, x2, data.spins, data.atoms, data.charges
#     )  # log prob of proposal
#     hmean2 = _harmonic_mean(x2, atoms)  # needed for probability of reverse jump

#     lq_1 = _log_prob_gaussian(x1, x2, stddev * hmean1)  # forward probability
#     lq_2 = _log_prob_gaussian(x2, x1, stddev * hmean2)  # reverse probability
#     ratio = lp_2 + lq_2 - lp_1 - lq_1

#     x1 = jnp.reshape(x1, [n, -1])
#     x2 = jnp.reshape(x2, [n, -1])
#   x_new, key, lp_new, num_accepts = mh_accept(
#       x1, x2, lp_1, lp_2, ratio, key, num_accepts)
#   new_data = networks.FermiNetData(**(dict(data) | {'positions': x_new}))
#   return new_data, key, lp_new, num_accepts


# def mh_block_update(
#     params: networks.ParamTree,
#     f: networks.LogFermiNetLike,
#     data: networks.FermiNetData,
#     key: chex.PRNGKey,
#     lp_1,
#     num_accepts,
#     stddev=0.02,
#     atoms=None,
#     ndim=3,
#     blocks=1,
#     i=0,
#     enforce_symmetry_by_shift: str = 'none',
#     symmetry_shift_kwargs: dict = {'lattice': None, 'move_width': 1}
# ):
#   """Performs one Metropolis-Hastings step for a block of electrons.

#   Args:
#     params: Wavefuncttion parameters.
#     f: Callable with LogFermiNetLike signature which returns the log of the
#       wavefunction (i.e. the sqaure root of the log probability of x).
#     data: Initial MCMC configuration (batched).
#     key: RNG state.
#     lp_1: log probability of f evaluated at x1 given parameters params.
#     num_accepts: Number of MH move proposals accepted.
#     stddev: width of Gaussian move proposal.
#     atoms: Not implemented. Raises an error if not None.
#     ndim: dimensionality of system.
#     blocks: number of blocks to split electron updates into.
#     i: index of block of electrons to move.

#   Returns:
#     (x, key, lp, num_accepts), where:
#       x: MCMC configurations with updated positions.
#       key: RNG state.
#       lp: log probability of f evaluated at x.
#       num_accepts: update running total of number of accepted MH moves.

#   Raises:
#     NotImplementedError: if atoms is supplied.
#   """
#   key, subkey = jax.random.split(key)
#   batch_size = data.positions.shape[0]
#   nelec = data.positions.shape[1] // ndim
#   pad = (blocks - nelec % blocks) % blocks
#   x1 = jnp.reshape(
#       jnp.pad(data.positions, ((0, 0), (0, pad * ndim))),
#       [batch_size, blocks, -1, ndim],
#   )
#   ii = i % blocks
#   if atoms is None:  # symmetric prop, same stddev everywhere
#     x2 = x1.at[:, ii].add(
#         stddev * jax.random.normal(subkey, shape=x1[:, ii].shape))
#     x2 = jnp.reshape(x2, [batch_size, -1])
#     if pad > 0:
#       x2 = x2[..., :-pad*ndim]
#     # log prob of proposal
#     lp_2 = 2.0 * f(params, x2, data.spins, data.atoms, data.charges)
#     ratio = lp_2 - lp_1
#   else:  # asymmetric proposal, stddev propto harmonic mean of nuclear distances
#     raise NotImplementedError('Still need to work out reverse probabilities '
#                               'for asymmetric moves.')

#   x1 = jnp.reshape(x1, [batch_size, -1])
#   if pad > 0:
#     x1 = x1[..., :-pad*ndim]
#   x_new, key, lp_new, num_accepts = mh_accept(
#       x1, x2, lp_1, lp_2, ratio, key, num_accepts)
#   new_data = networks.FermiNetData(**(dict(data) | {'positions': x_new}))
#   return new_data, key, lp_new, num_accepts


# def make_mcmc_step(batch_network,
#                    batch_per_device,
#                    steps=10,
#                    atoms=None,
#                    ndim=3,
#                    blocks=1,
#                    enforce_symmetry_by_shift = 'none',
#                    symmetry_shift_kwargs = {'lattice': None, 'move_width': 1}):
#   """Creates the MCMC step function.

#   Args:
#     batch_network: function, signature (params, x), which evaluates the log of
#       the wavefunction (square root of the log probability distribution) at x
#       given params. Inputs and outputs are batched.
#     batch_per_device: Batch size per device.
#     steps: Number of MCMC moves to attempt in a single call to the MCMC step
#       function.
#     atoms: atom positions. If given, an asymmetric move proposal is used based
#       on the harmonic mean of electron-atom distances for each electron.
#       Otherwise the (conventional) normal distribution is used.
#     ndim: Dimensionality of the system (usually 3).
#     blocks: Number of blocks to split the updates into. If 1, use all-electron
#       moves.

#   Returns:
#     Callable which performs the set of MCMC steps.
#   """
#   inner_fun = mh_block_update if blocks > 1 else mh_update

#   def mcmc_step(params, data, key, width):
#     """Performs a set of MCMC steps.

#     Args:
#       params: parameters to pass to the network.
#       data: (batched) MCMC configurations to pass to the network.
#       key: RNG state.
#       width: standard deviation to use in the move proposal.

#     Returns:
#       (data, pmove), where data is the updated MCMC configurations, key the
#       updated RNG state and pmove the average probability a move was accepted.
#     """
#     pos = data.positions

#     def step_fn(i, x):
#       return inner_fun(
#           params,
#           batch_network,
#           *x,
#           stddev=width,
#           atoms=atoms,
#           ndim=ndim,
#           blocks=blocks,
#           i=i,
#           enforce_symmetry_by_shift = enforce_symmetry_by_shift,
#           symmetry_shift_kwargs = symmetry_shift_kwargs)

#     nsteps = steps * blocks
#     logprob = 2.0 * batch_network(
#         params, pos, data.spins, data.atoms, data.charges
#     )
#     new_data, key, _, num_accepts = lax.fori_loop(
#         0, nsteps, step_fn, (data, key, logprob, 0.0)
#     )
#     pmove = jnp.sum(num_accepts) / (nsteps * batch_per_device)
#     pmove = constants.pmean(pmove)
#     return new_data, pmove

#   return mcmc_step


# def update_mcmc_width(
#     t: int,
#     width: jnp.ndarray,
#     adapt_frequency: int,
#     pmove: jnp.ndarray,
#     pmoves: np.ndarray,
#     pmove_max: float = 0.55,
#     pmove_min: float = 0.5,
# ) -> tuple[jnp.ndarray, np.ndarray]:
#   """Updates the width in MCMC steps.

#   Args:
#     t: Current step.
#     width: Current MCMC width.
#     adapt_frequency: The number of iterations after which the update is applied.
#     pmove: Acceptance ratio in the last step.
#     pmoves: Acceptance ratio over the last N steps, where N is the number of
#       steps between MCMC width updates.
#     pmove_max: The upper threshold for the range of allowed pmove values
#     pmove_min: The lower threshold for the range of allowed pmove values

#   Returns:
#     width: Updated MCMC width.
#     pmoves: Updated `pmoves`.
#   """

#   t_since_mcmc_update = t % adapt_frequency
#   # update `pmoves`; `pmove` should be the same across devices
#   pmoves[t_since_mcmc_update] = pmove.reshape(-1)[0].item()
#   if t > 0 and t_since_mcmc_update == 0:
#     if np.mean(pmoves) > pmove_max:
#       width *= 1.1
#     elif np.mean(pmoves) < pmove_min:
#       width /= 1.1
#   return width, pmoves


import chex
from ferminet import constants
from ferminet import networks
import jax
from jax import lax
from jax import numpy as jnp
import numpy as np


def apply_random_lattice_shifts(key: chex.PRNGKey, x1, lattice, move_width = 1):
    """
    Shift all electron positions by the same random lattice vector per batch element.
    
    Args:
        key: JAX PRNGKey
        x1: jnp.ndarray of shape (batch_size, N * d), where d = 2
        lattice: jnp.ndarray of shape (2, 2) representing the lattice vectors as columns
        move_width: width of lattice shifts

    Returns:
        x2: jnp.ndarray of shape (batch_size, N * d)
    """
    batch_size = x1.shape[0]
    dim = lattice.shape[1]  # spatial dimension
    # print(x1.shape)
    num_electrons = x1.shape[1] // dim

    # Step 1: Sample two random integers per batch element
    key, subkey = jax.random.split(key)
    normal_samples = move_width*jax.random.normal(key, shape=(batch_size, 2), dtype=jnp.float32)
    random_ints = jnp.round(normal_samples).astype(jnp.int32)
    # random_ints = jax.random.randint(subkey, shape=(batch_size, 2), minval=-move_width, maxval=move_width)

    # Step 2: Compute random lattice shifts: (batch_size, 2)
    shift_vectors = jnp.einsum('ij,kj->ki', lattice, random_ints)  # (2,2) x (batch_size, 2)ᵀ -> (batch_size, 2)
    # print(shift_vectors.shape)
    # Step 3: Tile shift vector across N electrons
    tiled_shifts = jnp.tile(shift_vectors[:, None, :], (1, num_electrons, 1))  # (batch_size, N, 2)
    reshaped_shifts = tiled_shifts.reshape((batch_size, num_electrons * dim))    # (batch_size, N*d)

    # Step 4: Apply shifts
    x2 = x1 + reshaped_shifts

    return x2


def _harmonic_mean(x, atoms):
  """Calculates the harmonic mean of each electron distance to the nuclei.

  Args:
    x: electron positions. Shape (batch, nelectrons, 1, ndim). Note the third
      dimension is already expanded, which allows for avoiding additional
      reshapes in the MH algorithm.
    atoms: atom positions. Shape (natoms, ndim)

  Returns:
    Array of shape (batch, nelectrons, 1, 1), where the (i, j, 0, 0) element is
    the harmonic mean of the distance of the j-th electron of the i-th MCMC
    configuration to all atoms.
  """
  ae = x - atoms[None, ...]
  r_ae = jnp.linalg.norm(ae, axis=-1, keepdims=True)
  return 1.0 / jnp.mean(1.0 / r_ae, axis=-2, keepdims=True)


def _log_prob_gaussian(x, mu, sigma):
  """Calculates the log probability of Gaussian with diagonal covariance.

  Args:
    x: Positions. Shape (batch, nelectron, 1, ndim) - as used in mh_update.
    mu: means of Gaussian distribution. Same shape as or broadcastable to x.
    sigma: standard deviation of the distribution. Same shape as or
      broadcastable to x.

  Returns:
    Log probability of Gaussian distribution with shape as required for
    mh_update - (batch, nelectron, 1, 1).
  """
  numer = jnp.sum(-0.5 * ((x - mu)**2) / (sigma**2), axis=[1, 2, 3])
  denom = x.shape[-1] * jnp.sum(jnp.log(sigma), axis=[1, 2, 3])
  return numer - denom


def mh_accept(x1, x2, lp_1, lp_2, ratio, key, num_accepts):
  """Given state, proposal, and probabilities, execute MH accept/reject step."""
  key, subkey = jax.random.split(key)
  rnd = jnp.log(jax.random.uniform(subkey, shape=ratio.shape))
  cond = ratio > rnd
  x_new = jnp.where(cond[..., None], x2, x1)
  lp_new = jnp.where(cond, lp_2, lp_1)
  num_accepts += jnp.sum(cond)
  return x_new, key, lp_new, num_accepts


# def _project_positions_to_supercell(positions, lattice, ndim):
#   """Project positions back into the supercell spanned by columns of `lattice`.

#   positions: (batch, nelec*ndim)
#   lattice: (2, 2) with columns L1, L2
#   Only acts on the first 2 spatial dimensions when ndim == 2.
#   """
#   if lattice is None:
#     return positions
#   if ndim != 2:
#     return positions

#   L = lattice
#   Linv = jnp.linalg.inv(L)  # 2x2
#   batch_size = positions.shape[0]
#   nelec = positions.shape[1] // ndim

#   x_reshaped = positions.reshape(batch_size, nelec, ndim)  # (b, n, 2)
#   r2d = x_reshaped[..., :2]                               # (b, n, 2)

#   # fractional coords t s.t. r = L @ t  =>  t = Linv @ r
#   # r2d: (b, n, 2), Linv: (2,2) => t: (b, n, 2)
#   t = jnp.einsum('bni,ij->bnj', r2d, Linv)

#   # wrap into [0,1) cell
#   t_wrapped = t - jnp.floor(t)

#   # back to Cartesian: r_wrapped = L @ t_wrapped
#   r_wrapped = jnp.einsum('bni,ij->bnj', t_wrapped, L)

#   x_reshaped = x_reshaped.at[..., :2].set(r_wrapped)
#   return x_reshaped.reshape(batch_size, nelec * ndim)

def _project_positions_to_supercell(positions, lattice, ndim):
  """Project positions back into the supercell spanned by rows of `lattice`.

  positions: (batch, nelec*ndim)
  lattice: (2, 2) array representing the supercell lattice vectors
  Only acts on the first 2 spatial dimensions when ndim == 2.
  """
  if lattice is None:
    return positions
  if ndim != 2:
    return positions

  batch_size = positions.shape[0]
  nelec = positions.shape[1] // ndim

  # Reshape to (batch, nelec, 2)
  x_reshaped = positions.reshape(batch_size, nelec, ndim)
  r2d = x_reshaped[..., :2]  # (batch, nelec, 2)

  # Inverse of lattice (2x2) – same role as lattice_inverse in your NumPy code
  lattice_inverse = jnp.linalg.inv(lattice)

  # Convert to fractional coords: fractional = r2d @ lattice_inverse.T
  fractional_coords = jnp.matmul(r2d, lattice_inverse.T)

  eps = 0.0
  fractional_coords = jnp.mod(fractional_coords + eps, 1.0 - eps)

  # Back to Cartesian: supercell_positions = fractional @ lattice.T
  r_wrapped = jnp.matmul(fractional_coords, lattice.T)

  # Put wrapped 2D coords back, keep other dimensions unchanged
  x_reshaped = x_reshaped.at[..., :2].set(r_wrapped)

  return x_reshaped.reshape(batch_size, nelec * ndim)

def mh_update(
    params: networks.ParamTree,
    f: networks.LogFermiNetLike,
    data: networks.FermiNetData,
    key: chex.PRNGKey,
    lp_1,
    num_accepts,
    stddev=0.02,
    atoms=None,
    ndim=3,
    blocks=1,
    i=0,
    enforce_symmetry_by_shift: str = 'none',
    symmetry_shift_kwargs: dict = {'lattice': None, 'move_width': 1},
    project_to_supercell: bool = False,
):
  """Performs one Metropolis-Hastings step using an all-electron move.
  ...
  """
  del i, blocks  # electron index ignored for all-electron moves
  key, subkey = jax.random.split(key)
  x1 = data.positions
  if atoms is None:  # symmetric proposal, same stddev everywhere
    x2 = x1 + stddev * jax.random.normal(subkey, shape=x1.shape)  # proposal
    if enforce_symmetry_by_shift == 'lattice':
      x2 = apply_random_lattice_shifts(
          key, x2, symmetry_shift_kwargs['lattice'],
          symmetry_shift_kwargs['move_width'])
    if project_to_supercell:
      x2 = _project_positions_to_supercell(
          x2, symmetry_shift_kwargs.get('lattice', None), ndim)

    lp_2 = 2.0 * f(
        params, x2, data.spins, data.atoms, data.charges
    )  # log prob of proposal
    ratio = lp_2 - lp_1
  else:  # asymmetric proposal, stddev propto harmonic mean of nuclear distances
    n = x1.shape[0]
    x1 = jnp.reshape(x1, [n, -1, 1, ndim])
    hmean1 = _harmonic_mean(x1, atoms)  # harmonic mean of distances to nuclei

    x2 = x1 + stddev * hmean1 * jax.random.normal(subkey, shape=x1.shape)
    lp_2 = 2.0 * f(
        params, x2, data.spins, data.atoms, data.charges
    )  # log prob of proposal
    hmean2 = _harmonic_mean(x2, atoms)  # needed for probability of reverse jump

    lq_1 = _log_prob_gaussian(x1, x2, stddev * hmean1)  # forward probability
    lq_2 = _log_prob_gaussian(x2, x1, stddev * hmean2)  # reverse probability
    ratio = lp_2 + lq_2 - lp_1 - lq_1

    x1 = jnp.reshape(x1, [n, -1])
    x2 = jnp.reshape(x2, [n, -1])
  x_new, key, lp_new, num_accepts = mh_accept(
      x1, x2, lp_1, lp_2, ratio, key, num_accepts)
  new_data = networks.FermiNetData(**(dict(data) | {'positions': x_new}))
  return new_data, key, lp_new, num_accepts


def mh_block_update(
    params: networks.ParamTree,
    f: networks.LogFermiNetLike,
    data: networks.FermiNetData,
    key: chex.PRNGKey,
    lp_1,
    num_accepts,
    stddev=0.02,
    atoms=None,
    ndim=3,
    blocks=1,
    i=0,
    enforce_symmetry_by_shift: str = 'none',
    symmetry_shift_kwargs: dict = {'lattice': None, 'move_width': 1},
    project_to_supercell: bool = False,
):
  """Performs one Metropolis-Hastings step for a block of electrons.
  ...
  """
  key, subkey = jax.random.split(key)
  batch_size = data.positions.shape[0]
  nelec = data.positions.shape[1] // ndim
  pad = (blocks - nelec % blocks) % blocks
  x1 = jnp.reshape(
      jnp.pad(data.positions, ((0, 0), (0, pad * ndim))),
      [batch_size, blocks, -1, ndim],
  )
  ii = i % blocks
  if atoms is None:  # symmetric prop, same stddev everywhere
    x2 = x1.at[:, ii].add(
        stddev * jax.random.normal(subkey, shape=x1[:, ii].shape))
    x2 = jnp.reshape(x2, [batch_size, -1])
    if pad > 0:
      x2 = x2[..., :-pad*ndim]
    if project_to_supercell:
      x2 = _project_positions_to_supercell(
          x2, symmetry_shift_kwargs.get('lattice', None), ndim)
    # log prob of proposal
    lp_2 = 2.0 * f(params, x2, data.spins, data.atoms, data.charges)
    ratio = lp_2 - lp_1
  else:  # asymmetric proposal, stddev propto harmonic mean of nuclear distances
    raise NotImplementedError('Still need to work out reverse probabilities '
                              'for asymmetric moves.')

  x1 = jnp.reshape(x1, [batch_size, -1])
  if pad > 0:
    x1 = x1[..., :-pad*ndim]
  x_new, key, lp_new, num_accepts = mh_accept(
      x1, x2, lp_1, lp_2, ratio, key, num_accepts)
  new_data = networks.FermiNetData(**(dict(data) | {'positions': x_new}))
  return new_data, key, lp_new, num_accepts


def make_mcmc_step(batch_network,
                   batch_per_device,
                   steps=10,
                   atoms=None,
                   ndim=3,
                   blocks=1,
                   enforce_symmetry_by_shift = 'none',
                   symmetry_shift_kwargs = {'lattice': None, 'move_width': 1},
                   project_to_supercell: bool = False):
  """Creates the MCMC step function.
  ...
  """
  inner_fun = mh_block_update if blocks > 1 else mh_update

  def mcmc_step(params, data, key, width):
    """Performs a set of MCMC steps.
    ...
    """
    pos = data.positions
    if project_to_supercell:
      pos = _project_positions_to_supercell(
          pos, symmetry_shift_kwargs.get('lattice', None), ndim)
      data_proj = networks.FermiNetData(**(dict(data) | {'positions': pos}))
    else:
      data_proj = data

    def step_fn(i, x):
      return inner_fun(
          params,
          batch_network,
          *x,
          stddev=width,
          atoms=atoms,
          ndim=ndim,
          blocks=blocks,
          i=i,
          enforce_symmetry_by_shift = enforce_symmetry_by_shift,
          symmetry_shift_kwargs = symmetry_shift_kwargs,
          project_to_supercell = project_to_supercell)

    nsteps = steps * blocks
    logprob = 2.0 * batch_network(
        params, pos, data_proj.spins, data_proj.atoms, data_proj.charges
    )
    new_data, key, _, num_accepts = lax.fori_loop(
        0, nsteps, step_fn, (data_proj, key, logprob, 0.0)
    )
    pmove = jnp.sum(num_accepts) / (nsteps * batch_per_device)
    pmove = constants.pmean(pmove)
    return new_data, pmove

  return mcmc_step


def update_mcmc_width(
    t: int,
    width: jnp.ndarray,
    adapt_frequency: int,
    pmove: jnp.ndarray,
    pmoves: np.ndarray,
    pmove_max: float = 0.55,
    pmove_min: float = 0.5,
) -> tuple[jnp.ndarray, np.ndarray]:
  """Updates the width in MCMC steps.
  ...
  """

  t_since_mcmc_update = t % adapt_frequency
  # update `pmoves`; `pmove` should be the same across devices
  pmoves[t_since_mcmc_update] = pmove.reshape(-1)[0].item()
  if t > 0 and t_since_mcmc_update == 0:
    if np.mean(pmoves) > pmove_max:
      width *= 1.1
    elif np.mean(pmoves) < pmove_min:
      width /= 1.1
  return width, pmoves

######## MALA #########