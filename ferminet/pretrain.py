# Copyright 2020 DeepMind Technologies Limited.
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

"""Utilities for pretraining and importing PySCF models."""

from typing import Callable, Mapping, Sequence, Tuple, Union

from absl import logging
import chex
from ferminet import constants
from ferminet import mcmc
from ferminet import networks
from ferminet.utils import scf
from ferminet.utils import system
import jax
from jax import numpy as jnp
import kfac_jax
import numpy as np
import optax
import pyscf
from ferminet import ellipticfunctions  

def get_hf(molecule: Sequence[system.Atom] | None = None,
           nspins: Tuple[int, int] | None = None,
           basis: str | None = 'sto-3g',
           ecp: Mapping[str, str] | None = None,
           core_electrons: Mapping[str, int] | None = None,
           pyscf_mol: pyscf.gto.Mole | None = None,
           restricted: bool | None = False,
           states: int = 0,
           excitation_type: str = 'ordered') -> scf.Scf:
  """Returns an Scf object with the Hartree-Fock solution to the system.

  Args:
    molecule: the molecule in internal format.
    nspins: tuple with number of spin up and spin down electrons.
    basis: basis set to use in Hartree-Fock calculation.
    ecp: dictionary of the ECP to use for different atoms.
    core_electrons: dictionary of the number of core electrons excluded by the
      pseudopotential/effective core potential.
    pyscf_mol: pyscf Mole object defining the molecule. If supplied,
      molecule, nspins and basis are ignored.
    restricted: If true, perform a restricted Hartree-Fock calculation,
      otherwise perform an unrestricted Hartree-Fock calculation.
    states: Number of excited states.  If nonzero, compute all single and double
      excitations of the Hartree-Fock solution and return coefficients for the
      lowest ones.
    excitation_type: The way to construct different states for excited state
      pretraining. One of 'ordered' or 'random'. 'Ordered' tends to work better,
      but 'random' is necessary for some systems, especially double excitaitons.
  """
  if pyscf_mol:
    scf_approx = scf.Scf(pyscf_mol=pyscf_mol,
                         restricted=restricted)
  else:
    scf_approx = scf.Scf(molecule,
                         nelectrons=nspins,
                         basis=basis,
                         ecp=ecp,
                         core_electrons=core_electrons,
                         restricted=restricted)
  scf_approx.run(excitations=max(states - 1, 0),
                 excitation_type=excitation_type)
  return scf_approx


def eval_orbitals(scf_approx: scf.Scf, pos: Union[np.ndarray, jnp.ndarray],
                  nspins: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
  """Evaluates SCF orbitals from PySCF at a set of positions.

  Args:
    scf_approx: an scf.Scf object that contains the result of a PySCF
      calculation.
    pos: an array of electron positions to evaluate the orbitals at, of shape
      (..., nelec*3), where the leading dimensions are arbitrary, nelec is the
      number of electrons and the spin up electrons are ordered before the spin
      down electrons.
    nspins: tuple with number of spin up and spin down electrons.

  Returns:
    tuple with matrices of orbitals for spin up and spin down electrons, with
    the same leading dimensions as in pos.
  """
  if not isinstance(pos, np.ndarray):  # works even with JAX array
    try:
      pos = pos.copy()
    except AttributeError as exc:
      raise ValueError('Input must be either NumPy or JAX array.') from exc
  leading_dims = pos.shape[:-1]
  # split into separate electrons
  pos = np.reshape(pos, [-1, 3])  # (batch*nelec, 3)
  mos = scf_approx.eval_mos(pos)  # (batch*nelec, nbasis), (batch*nelec, nbasis)
  # Reshape into (batch, nelec, nbasis) for each spin channel.
  mos = [np.reshape(mo, leading_dims + (sum(nspins), -1)) for mo in mos]
  # Return (using Aufbau principle) the matrices for the occupied alpha and
  # beta orbitals. Number of alpha electrons given by nspins[0].
  alpha_spin = mos[0][..., :nspins[0], :nspins[0]]
  beta_spin = mos[1][..., nspins[0]:, :nspins[1]]
  return alpha_spin, beta_spin


def make_pretrain_step(
    batch_orbitals: networks.OrbitalFnLike,
    batch_network: networks.LogFermiNetLike,
    optimizer_update: optax.TransformUpdateFn,
    electrons: Tuple[int, int],
    batch_size: int = 0,
    full_det: bool = False,
    scf_fraction: float = 0.0,
    states: int = 0,
):
  """Creates function for performing one step of Hartre-Fock pretraining.

  Args:
    batch_orbitals: callable with signature f(params, data), which given network
      parameters and a batch of electron positions, returns the orbitals in the
      network evaluated at those positions.
    batch_network: callable with signature f(params, data), which given network
      parameters and a batch of electron positions, returns the log of the
      magnitude of the (wavefunction) network  evaluated at those positions.
    optimizer_update: callable for transforming the gradients into an update (ie
      conforms to the optax API).
    electrons: number of spin-up and spin-down electrons.
    batch_size: number of walkers per device, used to make MCMC step.
    full_det: If true, evaluate all electrons in a single determinant.
      Otherwise, evaluate products of alpha- and beta-spin determinants.
    scf_fraction: What fraction of the wavefunction sampled from is the SCF
      wavefunction and what fraction is the neural network wavefunction?
    states: Number of excited states, if not 0.

  Returns:
    Callable for performing a single pretraining optimisation step.
  """

  # Create a function which gives either the SCF ansatz, the neural network
  # ansatz, or a weighted mixture of the two.
  if scf_fraction > 1 or scf_fraction < 0:
    raise ValueError('scf_fraction must be in between 0 and 1, inclusive.')

  if states:
    def scf_network(fn, x):
      x = x.reshape(x.shape[:-1] + (states, -1))
      slater_fn = jax.vmap(fn, in_axes=(-2, None), out_axes=-2)
      slogdets = slater_fn(x, electrons)
      # logsumexp trick
      maxlogdet = jnp.max(slogdets[1])
      dets = slogdets[0] * jnp.exp(slogdets[1] - maxlogdet)
      result = jnp.linalg.slogdet(dets)
      return result[1] + maxlogdet * slogdets[1].shape[-1]
  else:
    scf_network = lambda fn, x: fn(x, electrons)[1]

  if scf_fraction < 1e-6:
    def mcmc_network(full_params, pos, spins, atoms, charges):
      return batch_network(full_params['ferminet'], pos, spins, atoms, charges)
  elif scf_fraction > 0.999999:
    def mcmc_network(full_params, pos, spins, atoms, charges):
      del spins, atoms, charges
      return scf_network(full_params['scf'].eval_slater, pos)
  else:
    def mcmc_network(full_params, pos, spins, atoms, charges):
      log_ferminet = batch_network(full_params['ferminet'], pos, spins, atoms,
                                   charges)
      log_scf = scf_network(full_params['scf'].eval_slater, pos)
      return (1 - scf_fraction) * log_ferminet + scf_fraction * log_scf

  mcmc_step = mcmc.make_mcmc_step(
      mcmc_network, batch_per_device=batch_size, steps=20)

  def loss_fn(
      params: networks.ParamTree,
      data: networks.FermiNetData,
      scf_approx: scf.Scf,
  ):
    pos = data.positions
    spins = data.spins
    if states:
      # Make vmap-ed versions of eval_orbitals and batch_orbitals over the
      # states dimension.
      # (batch, states, nelec*ndim)
      pos = jnp.reshape(pos, pos.shape[:-1] + (states, -1))
      # (batch, states, nelec)
      spins = jnp.reshape(spins, spins.shape[:-1] + (states, -1))

      scf_orbitals = jax.vmap(
          scf_approx.eval_orbitals, in_axes=(-2, None), out_axes=-4
      )

      def net_orbitals(params, pos, spins, atoms, charges):
        vmapped_orbitals = jax.vmap(
            batch_orbitals, in_axes=(None, -2, -2, None, None), out_axes=-4
        )
        # Dimensions of result are
        # [(batch, states, ndet*states, nelec, nelec)]
        result = vmapped_orbitals(params, pos, spins, atoms, charges)
        result = [
            jnp.reshape(r, r.shape[:-3] + (states, -1) + r.shape[-2:])
            for r in result
        ]
        result = [jnp.transpose(r, (0, 3, 1, 2, 4, 5)) for r in result]
        # We draw distinct samples for each excited state (electron
        # configuration), and then evaluate each state within each sample.
        # Output dimensions are:
        # (batch, det, electron configuration,
        # excited state, electron, orbital)
        return result

    else:
      scf_orbitals = scf_approx.eval_orbitals
      net_orbitals = batch_orbitals

    target = scf_orbitals(pos, electrons)
    orbitals = net_orbitals(params, pos, spins, data.atoms, data.charges)
    cnorm = lambda x, y: (x - y) * jnp.conj(x - y)  # complex norm
    if full_det:
      dims = target[0].shape[:-2]  # (batch) or (batch, states).
      na = target[0].shape[-2]
      nb = target[1].shape[-2]
      target = jnp.concatenate(
          (
              jnp.concatenate(
                  (target[0], jnp.zeros(dims + (na, nb))), axis=-1),
              jnp.concatenate(
                  (jnp.zeros(dims + (nb, na)), target[1]), axis=-1),
          ),
          axis=-2,
      )
      result = jnp.mean(cnorm(target[:, None, ...], orbitals[0])).real
    else:
      result = jnp.array([
          jnp.mean(cnorm(t[:, None, ...], o)).real
          for t, o in zip(target, orbitals)
      ]).sum()
    return constants.pmean(result)

  def pretrain_step(data, params, state, key, scf_approx):
    """One iteration of pretraining to match HF."""
    val_and_grad = jax.value_and_grad(loss_fn, argnums=0)
    loss_val, search_direction = val_and_grad(params, data, scf_approx)
    search_direction = constants.pmean(search_direction)
    updates, state = optimizer_update(search_direction, state, params)
    params = optax.apply_updates(params, updates)
    full_params = {'ferminet': params, 'scf': scf_approx}
    data, pmove = mcmc_step(full_params, data, key, width=0.02)
    return data, params, state, loss_val, pmove

  return pretrain_step


def pretrain_hartree_fock(
    *,
    params: networks.ParamTree,
    positions: jnp.ndarray,
    spins: jnp.ndarray,
    atoms: jnp.ndarray,
    charges: jnp.ndarray,
    batch_network: networks.FermiNetLike,
    batch_orbitals: networks.OrbitalFnLike,
    network_options: networks.BaseNetworkOptions,
    sharded_key: chex.PRNGKey,
    electrons: Tuple[int, int],
    scf_approx: scf.Scf,
    iterations: int = 1000,
    batch_size: int = 0,
    logger: Callable[[int, float], None] | None = None,
    scf_fraction: float = 0.0,
    states: int = 0,
):
  """Performs training to match initialization as closely as possible to HF.

  Args:
    params: Network parameters.
    positions: Electron position configurations.
    spins: Electron spin configuration (1 for alpha electrons, -1 for beta), as
      a 1D array. Note we always use the same spin configuration for the entire
      batch in pretraining.
    atoms: atom positions (batched).
    charges: atomic charges (batched).
    batch_network: callable with signature f(params, data), which given network
      parameters and a batch of electron positions, returns the log of the
      magnitude of the (wavefunction) network  evaluated at those positions.
    batch_orbitals: callable with signature f(params, data), which given network
      parameters and a batch of electron positions, returns the orbitals in the
      network evaluated at those positions.
    network_options: FermiNet network options.
    sharded_key: JAX RNG state (sharded) per device.
    electrons: tuple of number of electrons of each spin.
    scf_approx: an scf.Scf object that contains the result of a PySCF
      calculation.
    iterations: number of pretraining iterations to perform.
    batch_size: number of walkers per device, used to make MCMC step.
    logger: Callable with signature (step, value) which externally logs the
      pretraining loss.
    scf_fraction: What fraction of the wavefunction sampled from is the SCF
      wavefunction and what fraction is the neural network wavefunction?
    states: Number of excited states, if not 0.

  Returns:
    params, positions: Updated network parameters and MCMC configurations such
    that the orbitals in the network closely match Hartree-Fock and the MCMC
    configurations are drawn from the log probability of the network.
  """
  # Pretraining is slow on larger systems (very low GPU utilization) because the
  # Hartree-Fock orbitals are evaluated on CPU and only on a single host.
  # Implementing the basis set in JAX would enable using GPUs and allow
  # eval_orbitals to be pmapped.

  optimizer = optax.adam(3.e-4)
  opt_state_pt = constants.pmap(optimizer.init)(params)

  pretrain_step = make_pretrain_step(
      batch_orbitals,
      batch_network,
      optimizer.update,
      electrons=electrons,
      batch_size=batch_size,
      full_det=network_options.full_det,
      scf_fraction=scf_fraction,
      states=states,
  )
  pretrain_step = constants.pmap(pretrain_step)

  batch_spins = jnp.tile(spins[None], [positions.shape[1], 1])
  pmap_spins = kfac_jax.utils.replicate_all_local_devices(batch_spins)
  data = networks.FermiNetData(
      positions=positions, spins=pmap_spins, atoms=atoms, charges=charges
  )

  for t in range(iterations):
    sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
    data, params, opt_state_pt, loss, pmove = pretrain_step(
        data, params, opt_state_pt, subkeys, scf_approx)
    logging.info('Pretrain iter %05d: %g %g', t, loss[0], pmove[0])
    if logger:
      logger(t, loss[0])
  return params, data.positions


### Laughlin pretraining


def _laughlin_log_psi_single(
    z_flat: jnp.ndarray,
    laughlin_params: Mapping[str, jnp.ndarray],
) -> jnp.ndarray:
  """log Ψ_Laughlin(z_flat) for a single configuration (2*Ne,)."""
  alpha  = laughlin_params['alpha']
  q      = laughlin_params['q']
  consts = laughlin_params['consts']
  return ellipticfunctions.laughlin_log_psi_torus(z_flat, alpha, q, consts)


_vmap_laughlin_log_psi = jax.vmap(
    _laughlin_log_psi_single, in_axes=(0, None)
)

# ----- Phase gradients for Laughlin target -----

def _laughlin_phase_grad_single(z_flat: jnp.ndarray,
                                laughlin_params) -> jnp.ndarray:
    """
    ∇_{r_i} φ_L for a single configuration.
    z_flat: shape (2*Ne,), real.
    Returns: shape (Ne, 2), real.
    """
    def phi_fn(z_flat_in):
        # log ψ_L is complex; its imaginary part is the phase φ_L
        logpsi_L = _laughlin_log_psi_single(z_flat_in, laughlin_params)
        return jnp.imag(logpsi_L)

    grad_phi_flat = jax.grad(phi_fn)(z_flat)            # (2*Ne,)
    dim = 2
    nelec = z_flat.shape[0] // dim
    return grad_phi_flat.reshape((nelec, dim))          # (Ne, 2)


_vmap_laughlin_phase_grad = jax.vmap(
    _laughlin_phase_grad_single, in_axes=(0, None)
)


# ----- Phase gradients for neural-network wavefunction -----

def _network_phase_grad_single(z_flat: jnp.ndarray,
                               spin_single: jnp.ndarray,
                               params,
                               batch_network,
                               atoms0: jnp.ndarray,
                               charges0: jnp.ndarray) -> jnp.ndarray:
    """
    ∇_{r_i} φ_NN for a single configuration.
    z_flat: shape (2*Ne,), real.
    spin_single: shape (Ne,)
    atoms0, charges0: unbatched (single copy) used for this config.
    Returns: shape (Ne, 2), real.
    """
    def phi_fn(z_flat_in):
        pos_in   = z_flat_in[None, :]        # (1, 2*Ne)
        spins_in = spin_single[None, :]      # (1, Ne)
        # logψ_net is complex: log|ψ| + i φ_NN
        logpsi_net = batch_network(params, pos_in, spins_in, atoms0, charges0)[0]
        return jnp.imag(logpsi_net)

    grad_phi_flat = jax.grad(phi_fn)(z_flat)           # (2*Ne,)
    dim = 2
    nelec = z_flat.shape[0] // dim
    return grad_phi_flat.reshape((nelec, dim))         # (Ne, 2)


def network_phase_grads_batch(params,
                              batch_network,
                              pos: jnp.ndarray,
                              spins: jnp.ndarray,
                              atoms: jnp.ndarray,
                              charges: jnp.ndarray) -> jnp.ndarray:
    """
    Batched phase gradients for NN:
      pos.shape    = (batch, 2*Ne)
      spins.shape  = (batch, Ne)
      atoms.shape  = (1, natoms, 3) or (batch, ...)
      charges.shape= (1, natoms) or (batch, ...)
    Returns: shape (batch, Ne, 2)
    """
    # Use a single copy of atoms/charges if they are batched,
    # just like in HF pretrain.
    atoms0   = atoms[0:1]
    charges0 = charges[0:1]

    def _single(z_flat, s_single):
        return _network_phase_grad_single(
            z_flat, s_single, params, batch_network, atoms0, charges0
        )

    return jax.vmap(_single, in_axes=(0, 0))(pos, spins)   # (batch, Ne, 2)

def laughlin_density_current_loss(
    params,
    batch_network,
    pos: jnp.ndarray,        # (batch, 2*Ne)
    spins: jnp.ndarray,      # (batch, Ne)
    atoms: jnp.ndarray,
    charges: jnp.ndarray,
    laughlin_params,
    lambda_phase: float = 1.0,
) -> jnp.ndarray:
    """
    Implements Eqs. (2)-(4) of Nazaryan et al.:
        L = ⟨ [log ρ_N - log ρ_L]^2 ⟩_{MC} + λ_J ⟨ Σ_i |∇_{r_i} φ_N - ∇_{r_i} φ_L|^2 ⟩_{MC}

    Here the Monte Carlo average is over the current sample set.
    """

    # ---------- 1. log-densities (for Eq. (2)) ----------
    # log ψ_NN and log ψ_L (complex)
    logpsi_net = batch_network(params, pos, spins, atoms, charges)      # (batch,)
    logpsi_L   = _vmap_laughlin_log_psi(pos, laughlin_params)          # (batch,)

    # log ρ = 2 Re log ψ
    logrho_net = 2.0 * jnp.real(logpsi_net)
    logrho_L   = 2.0 * jnp.real(logpsi_L)

    # small epsilon to avoid log(0) issues if you ever re-log these;
    # for Eq. (2) we just use the difference directly:
    diff_logrho = logrho_net - logrho_L
    density_loss = jnp.mean(diff_logrho**2)

    # ---------- 2. current / phase-gradient term (Eq. (3)) ----------
    if lambda_phase != 0.0:
        # ∇ φ_L and ∇ φ_NN, shapes (batch, Ne, 2)
        gradphi_L   = _vmap_laughlin_phase_grad(pos, laughlin_params)
        gradphi_net = network_phase_grads_batch(
            params, batch_network, pos, spins, atoms, charges
        )

        # difference of gradients, squared norm summed over electrons and dims
        grad_diff = gradphi_net - gradphi_L                  # (batch, Ne, 2)
        grad_diff_sq = jnp.sum(grad_diff**2, axis=(1, 2))    # (batch,)

        # optional safety against numerical explosions:
        grad_diff_sq = jnp.nan_to_num(grad_diff_sq, posinf=1e6, neginf=1e6)

        current_loss = jnp.mean(grad_diff_sq)
    else:
        current_loss = 0.0

    # ---------- 3. Total loss (Eq. (4)) ----------
    total_loss = density_loss + lambda_phase * current_loss
    return total_loss


def laughlin_density_only_loss(
    params,
    data,
    batch_network,
    laughlin_params,
):
  """NaN-safe density-matching loss between network and Laughlin.

  L = ⟨ [ (log ρ_net - log ρ_L) - mean ]^2 ⟩ over valid configs.
  """
  pos     = data.positions   # (batch, 2*Ne)
  spins   = data.spins
  atoms   = data.atoms
  charges = data.charges

  # Complex log ψ for network and Laughlin
  log_psi_net = batch_network(params, pos, spins, atoms, charges)    # (batch,)
  log_psi_L   = _vmap_laughlin_log_psi(pos, laughlin_params)         # (batch,)

  # 1) Sanitize NaN / ±∞ first (both real and imag parts)
  log_psi_net = jnp.nan_to_num(log_psi_net,
                               nan=0.0,
                               posinf=1e4,
                               neginf=-1e4)
  log_psi_L   = jnp.nan_to_num(log_psi_L,
                               nan=0.0,
                               posinf=1e4,
                               neginf=-1e4)

  # 2) log ρ = 2 Re(log ψ); we only care about Re part for density
  log_rho_net = 2.0 * jnp.real(log_psi_net)
  log_rho_L   = 2.0 * jnp.real(log_psi_L)

  # 3) Build a validity mask: both sides must be finite
  finite_net = jnp.isfinite(log_rho_net)
  finite_L   = jnp.isfinite(log_rho_L)
  mask_bool  = jnp.logical_and(finite_net, finite_L)     # (batch,)
  mask       = mask_bool.astype(jnp.float32)             # (batch,)

  # Make sure we don't divide by zero later
  valid_count = jnp.sum(mask) + 1e-8

  # 4) Difference in log-density
  delta = log_rho_net - log_rho_L          # (batch,)

  # Optionally clip extreme values (prevents overflow when squaring)
  delta = jnp.clip(delta, -50.0, 50.0)

  # 5) Apply mask WITHOUT boolean indexing (JAX-safe)
  delta_masked = delta * mask              # invalid entries set to 0

  # 6) Subtract mean difference over valid configs (remove global offset)
  mean_delta = jnp.sum(delta_masked) / valid_count
  delta_centered = (delta - mean_delta) * mask  # still zeros where mask=0

  # 7) Mean squared error over valid configs
  sq = delta_centered * delta_centered
  loss = jnp.sum(sq) / valid_count

  # Final NaN/inf guard (just in case)
  loss = jnp.nan_to_num(loss, nan=1e6, posinf=1e6, neginf=1e6)

  return loss

def make_laughlin_pretrain_step(
    batch_network: networks.FermiNetLike,   # returns complex log ψ_net
    optimizer_update: optax.TransformUpdateFn,
    electrons: Tuple[int, int],
    batch_size: int = 0,
    laughlin_fraction: float = 1.0,
    lambda_current: float = 0.1,
    laughlin_params: Mapping[str, jnp.ndarray] | None = None,
):
  """Creates function for performing one step of Laughlin pretraining.

  Args:
    batch_network: f(params, pos, spins, atoms, charges) -> complex log ψ_net.
    optimizer_update: optax-style update function.
    electrons: (n_up, n_down), kept for API symmetry (unused here).
    batch_size: walkers per device for MCMC.
    laughlin_fraction: fraction of sampling log-prob from Laughlin vs NN.
    laughlin_params: dict with keys 'alpha', 'q', 'consts' for Laughlin state.
  """
  if laughlin_params is None:
    raise ValueError('laughlin_params must be provided.')

  if laughlin_fraction < 0.0 or laughlin_fraction > 1.0:
    raise ValueError('laughlin_fraction must be in [0, 1].')

  # --- MCMC log-probability (uses only Re[log ψ]) ---

  if laughlin_fraction < 1e-6:
    # Pure NN sampling.
    def mcmc_network(full_params, pos, spins, atoms, charges):
      logpsi_net = batch_network(full_params['ferminet'], pos, spins,
                                 atoms, charges)
      return jnp.real(logpsi_net)

  elif laughlin_fraction > 0.999999:
    # Pure Laughlin sampling.
    def mcmc_network(full_params, pos, spins, atoms, charges):
      del spins, atoms, charges, full_params
      logpsi_L = _vmap_laughlin_log_psi(pos, laughlin_params)
      return jnp.real(logpsi_L)

  else:
    # Mixture in log-space for sampling.
    def mcmc_network(full_params, pos, spins, atoms, charges):
      logpsi_net = batch_network(full_params['ferminet'], pos, spins,
                                 atoms, charges)
      logpsi_L   = _vmap_laughlin_log_psi(pos, laughlin_params)
      logp_net = jnp.real(logpsi_net)
      logp_L   = jnp.real(logpsi_L)
      return (1.0 - laughlin_fraction) * logp_net + laughlin_fraction * logp_L

  # Same as before, but we will now pass a *variable* width.
  mcmc_step = mcmc.make_mcmc_step(
      mcmc_network, batch_per_device=batch_size, steps=20
  )

  def loss_fn(params, data, laughlin_params):
    pos   = data.positions      # (batch, 2*Ne)
    spins = data.spins          # (batch, Ne)
    atoms = data.atoms
    charges = data.charges

    loss = laughlin_density_current_loss(
        params,
        batch_network,
        pos,
        spins,
        atoms,
        charges,
        laughlin_params,
        lambda_phase=1.0,
    )
    return constants.pmean(loss)

  def pretrain_step(
      data: networks.FermiNetData,
      params: networks.ParamTree,
      state,
      key: chex.PRNGKey,
      mcmc_width: jnp.ndarray,   # <--- NEW ARG: per-device width
  ):
    """One iteration of Laughlin pretraining (amplitude + phase)."""
    val_and_grad = jax.value_and_grad(loss_fn, argnums=0)
    loss_val, grad = val_and_grad(params, data, laughlin_params)
    grad = constants.pmean(grad)

    updates, state = optimizer_update(grad, state, params)
    params = optax.apply_updates(params, updates)

    # Parameters seen by MCMC network.
    full_params = {'ferminet': params}

    # Refresh configs with one MCMC sweep using current width.
    data, pmove = mcmc_step(full_params, data, key, width=mcmc_width)

    return data, params, state, loss_val, pmove

  return pretrain_step

def pretrain_laughlin(
    *,
    params: networks.ParamTree,
    positions: jnp.ndarray,
    spins: jnp.ndarray,
    atoms: jnp.ndarray,
    charges: jnp.ndarray,
    batch_network: networks.FermiNetLike,      # returns complex log ψ_net
    network_options: networks.BaseNetworkOptions,
    sharded_key: chex.PRNGKey,
    electrons: Tuple[int, int],
    alpha: jnp.ndarray,
    q: int,
    consts: Mapping[str, jnp.ndarray],
    iterations: int = 1000,
    batch_size: int = 0,
    logger: Callable[[int, float], None] | None = None,
    laughlin_fraction: float = 1.0,
    lambda_current: float = 1.0,
    # NEW: adaptive MCMC controls
    mcmc_move_width: float = 0.02,
    mcmc_adapt_frequency: int = 50,
):
  """Pretrain network to match Laughlin (amplitude + phase).

  Returns:
    (params, positions): updated network params and MCMC configurations.
  """
  # Pack Laughlin state info into one object.
  laughlin_params = {
      'alpha': alpha,
      'q': jnp.asarray(q),
      'consts': consts,
  }

  optimizer = optax.adam(1.0e-5)
  opt_state_pt = constants.pmap(optimizer.init)(params)

  pretrain_step = make_laughlin_pretrain_step(
      batch_network=batch_network,
      optimizer_update=optimizer.update,
      electrons=electrons,
      batch_size=batch_size,
      laughlin_fraction=laughlin_fraction,
      laughlin_params=laughlin_params,
  )
  pretrain_step = constants.pmap(pretrain_step)

  # Spin broadcasting as in HF pretrain.
  pretrain_spins = spins  # you pass spins[0,0] from train()
  batch_spins = jnp.tile(pretrain_spins[None], [positions.shape[1], 1])
  pmap_spins = kfac_jax.utils.replicate_all_local_devices(batch_spins)

  data = networks.FermiNetData(
      positions=positions,
      spins=pmap_spins,
      atoms=atoms,
      charges=charges,
  )

  # ---------- Adaptive MCMC width state (host) ----------
  # scalar width → replicate to devices
  mcmc_width_scalar = float(mcmc_move_width)
  mcmc_width = kfac_jax.utils.replicate_all_local_devices(
      jnp.asarray(mcmc_width_scalar, dtype=jnp.float32)
  )
  pmoves = np.zeros(mcmc_adapt_frequency, dtype=np.float32)

  for t in range(iterations):
    sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
    # Note: pass mcmc_width into the pmapped step
    data, params, opt_state_pt, loss, pmove = pretrain_step(
        data, params, opt_state_pt, subkeys, mcmc_width
    )

    # pmove and loss are replicated across devices: take first one on host.
    loss_host = float(loss[0])
    pmove_host = float(pmove[0])

    # ----- Adapt MCMC move width on host, then re-broadcast -----
    mcmc_width_scalar, pmoves = mcmc.update_mcmc_width(
        t,
        mcmc_width_scalar,
        mcmc_adapt_frequency,
        pmove_host,
        pmoves,
    )
    mcmc_width = kfac_jax.utils.replicate_all_local_devices(
        jnp.asarray(mcmc_width_scalar, dtype=jnp.float32)
    )

    logging.info(
        'Laughlin pretrain iter %05d: loss=%g pmove=%g width=%g',
        t, loss_host, pmove_host, mcmc_width_scalar
    )
    if logger:
      logger(t, loss_host)

  return params, data.positions