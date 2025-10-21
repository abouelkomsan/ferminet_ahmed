from absl import logging
import chex
import importlib
import time
import os
from datetime import datetime
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
from ferminet import base_config
from ferminet import train
from ferminet.utils import statistics
from ferminet.utils import system
from ferminet.utils import utils
from ferminet.utils import writers
from ferminet import surgery
import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
import kfac_jax
import ml_collections
import numpy as np
import optax
from typing_extensions import Protocol
import pandas as pd
from math import comb
from functools import partial




def wavefunction(cfg: ml_collections.ConfigDict, writer_manager=None):
  """returns log of NN wavefunction.

  Args:
    cfg: ConfigDict containing the system and training parameters to run on. See
      base_config.default for more details.
    writer_manager: context manager with a write method for logging output. If
      None, a default writer (ferminet.utils.writers.Writer) is used.

  Raises:
    ValueError: if an illegal or unsupported value in cfg is detected.
  """
  # Device logging
  num_devices = jax.local_device_count()
  num_hosts = jax.device_count() // num_devices
  num_states = cfg.system.get('states', 0) or 1  # avoid 0/1 confusion
  host_batch_size = cfg.batch_size // num_hosts  # batch size per host
  total_host_batch_size = host_batch_size * num_states
  device_batch_size = host_batch_size // num_devices  # batch size per device
  data_shape = (num_devices, device_batch_size)

  # Convert mol config into array of atomic positions and charges
  atoms = jnp.stack([jnp.array(atom.coords) for atom in cfg.system.molecule])
  charges = jnp.array([atom.charge for atom in cfg.system.molecule])
  nspins = cfg.system.electrons

  if cfg.debug.deterministic:
    seed = 23
  else:
    seed = jnp.asarray([1e6 * time.time()])
    seed = int(multihost_utils.broadcast_one_to_all(seed)[0])
  key = jax.random.PRNGKey(seed)

  # Create parameters, network, and vmaped/pmaped derivations

  if cfg.network.make_feature_layer_fn:
    feature_layer_module, feature_layer_fn = (
        cfg.network.make_feature_layer_fn.rsplit('.', maxsplit=1))
    feature_layer_module = importlib.import_module(feature_layer_module)
    make_feature_layer: networks.MakeFeatureLayer = getattr(
        feature_layer_module, feature_layer_fn
    )
    feature_layer = make_feature_layer(
        natoms=charges.shape[0],
        nspins=cfg.system.electrons,
        ndim=cfg.system.ndim,
        **cfg.network.make_feature_layer_kwargs)
  else:
    feature_layer = networks.make_ferminet_features(
        natoms=charges.shape[0],
        nspins=cfg.system.electrons,
        ndim=cfg.system.ndim,
        rescale_inputs=cfg.network.get('rescale_inputs', False),
    )

  if cfg.network.make_envelope_fn:
    envelope_module, envelope_fn = (
        cfg.network.make_envelope_fn.rsplit('.', maxsplit=1))
    envelope_module = importlib.import_module(envelope_module)
    make_envelope = getattr(envelope_module, envelope_fn)
    envelope = make_envelope(**cfg.network.make_envelope_kwargs)  # type: envelopes.Envelope
  else:
    envelope = envelopes.make_isotropic_envelope()

  use_complex = cfg.network.get('complex', False)
  if cfg.network.network_type == 'ferminet':
    network = networks.make_fermi_net(
        nspins,
        charges,
        ndim=cfg.system.ndim,
        determinants=cfg.network.determinants,
        states=cfg.system.states,
        envelope=envelope,
        feature_layer=feature_layer,
        jastrow=cfg.network.get('jastrow', 'default'),
        bias_orbitals=cfg.network.bias_orbitals,
        full_det=cfg.network.full_det,
        rescale_inputs=cfg.network.get('rescale_inputs', False),
        complex_output=use_complex,
        **cfg.network.ferminet,
    )
  elif cfg.network.network_type == 'psiformer':
    network = psiformer.make_fermi_net(
        nspins,
        charges,
        ndim=cfg.system.ndim,
        determinants=cfg.network.determinants,
        states=cfg.system.states,
        envelope=envelope,
        feature_layer=feature_layer,
        jastrow=cfg.network.get('jastrow', 'default'),
        bias_orbitals=cfg.network.bias_orbitals,
        rescale_inputs=cfg.network.get('rescale_inputs', False),
        complex_output=use_complex,
        **cfg.network.psiformer,
    )
  key, subkey = jax.random.split(key)
  params = network.init(subkey)
  params = kfac_jax.utils.replicate_all_local_devices(params)
  signed_network = network.apply
  # Often just need log|psi(x)|.
  if cfg.system.get('states', 0):
    if cfg.optim.objective == 'vmc_overlap':
      logabs_network = networks.make_state_trace(signed_network,
                                                 cfg.system.states)
    else:
      logabs_network = utils.select_output(
          networks.make_total_ansatz(signed_network,
                                     cfg.system.get('states', 0),
                                     complex_output=use_complex), 1)
  else:
    logabs_network = lambda *args, **kwargs: signed_network(*args, **kwargs)[1]
  batch_network = jax.vmap(
      logabs_network, in_axes=(None, 0, 0, 0, 0), out_axes=0
  )  # batched network

  # Exclusively when computing the gradient wrt the energy for complex
  # wavefunctions, it is necessary to have log(psi) rather than log(|psi|).
  # This is unused if the wavefunction is real-valued.
  if cfg.system.get('states', 0):
    if cfg.optim.objective == 'vmc_overlap':
      # In the case of a penalty method, we actually need all outputs
      # to compute the gradient
      log_network_for_loss = networks.make_state_matrix(signed_network,
                                                        cfg.system.states)
      def log_network(*args, **kwargs):
        phase, mag = log_network_for_loss(*args, **kwargs)
        return mag + 1.j * phase
      return log_network
    else:
      def log_network(*args, **kwargs):
        if not use_complex:
          raise ValueError('This function should never be used if the '
                           'wavefunction is real-valued.')
        meta_net = networks.make_total_ansatz(signed_network,
                                              cfg.system.get('states', 0),
                                              complex_output=True)
        phase, mag = meta_net(*args, **kwargs)
        return mag + 1.j * phase
      return log_network
  else:
    def log_network(*args, **kwargs):
      if not use_complex:
        raise ValueError('This function should never be used if the '
                         'wavefunction is real-valued.')
      phase, mag = signed_network(*args, **kwargs)
      return mag + 1.j * phase
    return log_network
      


def load_recent_npz_files(directory, n=5):
    """
    Get the names of the most recent `.npz` files in a directory.

    Args:
        directory (str): Path to the directory containing `.npz` files.
        n (int): Number of recent files to return.

    Returns:
        list of str: List of file names (strings) of the most recent `.npz` files.
    """
    # List all `.npz` files in the directory
    npz_files = [f for f in os.listdir(directory) if f.endswith('.npz')]

    # Get full paths and sort by modification time
    npz_files_with_time = [
        (f, os.path.getmtime(os.path.join(directory, f))) for f in npz_files
    ]
    sorted_files = sorted(npz_files_with_time, key=lambda x: x[1], reverse=True)

    # Get the `n` most recent file names
    recent_files = [os.path.join(directory, f[0]) for f in sorted_files[:n]]

    return recent_files


def read_and_combine_checkpoints(file_list, host_batch_size):
    """
    Read data from multiple checkpoint files, collect attributes of `FermiNetData`,
    and combine them into a single `FermiNetData` object.

    Args:
        checkpoint: Checkpoint object with a `restore` method.
        file_list (list of str): List of checkpoint filenames to read.
        host_batch_size (int): Batch size for restoring checkpoints.

    Returns:
        tuple: Combined `FermiNetData` and the last checkpoint components.
    """
    combined_positions = []
    combined_spins = []
    combined_atoms = []
    combined_charges = []

    for file in file_list:
        # Restore data from the current checkpoint file
        t_init, data, params, opt_state_ckpt, mcmc_width_ckpt, density_state_ckpt = checkpoint.restore_no_batch_check(file, host_batch_size)
        
        # Collect attributes from `FermiNetData`
        combined_positions.append(data.positions)
        combined_spins.append(data.spins)
        combined_atoms.append(data.atoms)
        combined_charges.append(data.charges)

    # Concatenate all attributes
    positions = jnp.concatenate(combined_positions, axis=0)
    spins = jnp.concatenate(combined_spins, axis=0)
    atoms = jnp.concatenate(combined_atoms, axis=0)
    charges = jnp.concatenate(combined_charges, axis=0)

    # Reconstruct the combined `FermiNetData` object
    combined_data = networks.FermiNetData(
        positions=positions, spins=spins, atoms=atoms, charges=charges
    )

    # Return the combined data and the last checkpoint components
    return t_init, combined_data, params, opt_state_ckpt, mcmc_width_ckpt, density_state_ckpt

def lattice_vecs(a1:np.ndarray, a2:np.ndarray,Tmatrix:np.ndarray) -> np.ndarray:
  "Return the basis T1,T2 of the super-cell built from the unit cell lattice vectors a1 and a2"
  T1 = Tmatrix[0,0]*a1 + Tmatrix[0,1]*a2
  T2 = Tmatrix[1,0]*a1 + Tmatrix[1,1]*a2
  return np.column_stack([T1, T2])

def map_to_supercell_jax(positions, lattice_vectors):
    """
    Map positions back to the supercell using lattice vectors (batched version).

    Args:
        positions: (M, N*d) array of positions (flattened)
        lattice_vectors: (d,d) lattice matrix
    Returns:
        positions mapped back to supercell, same shape as input
    """
    M, Nd = positions.shape
    d = lattice_vectors.shape[0]
    N = Nd // d

    positions_reshaped = positions.reshape(M, N, d)             # (M,N,d)
    lattice_inv = jnp.linalg.inv(lattice_vectors)               # (d,d)
    frac_coords = jnp.einsum('mnd,dc->mnc', positions_reshaped, lattice_inv.T)  # (M,N,d)
    frac_coords = jnp.mod(frac_coords, 1.0)
    supercell_pos = jnp.einsum('mnd,dc->mnc', frac_coords, lattice_vectors.T)
    return supercell_pos.reshape(M, N*d)

def hbar2_over_m_eff(meff: float) -> float:
    """
    Returns hbar^2 / (m_eff) in units of meV·nm².
    `meff` should be given in units of the electron mass m_e.
    """
    # Physical constants
    hbar = 1.054571817e-34  # J·s
    m_e = 9.10938356e-31    # kg
    eV = 1.602176634e-19    # J
    meV = 1e-3 * eV
    nm2 = 1e-18             # m²

    # Convert hbar^2 / (2 m_eff) to meV·nm²
    value = hbar**2 / (meff * m_e)  # in J·m²
    value_meV_nm2 = value / meV / nm2  # convert to meV·nm²

    return value_meV_nm2

def coulomb_prefactor(epsilon_r: float) -> float:
    """
    Compute e^2 / (4 * pi * epsilon_0 * epsilon_r * r) at r = 1 nm,
    return result in meV units.
    """

    # Constants
    e = 1.602176634e-19        # C
    epsilon_0 = 8.854187817e-12  # F/m
    r_nm = 1e-9                # m
    J_to_meV = 1 / 1.602176634e-22  # 1 J = this many meV

    # Energy in J at r = 1 nm
    energy_J = e**2 / (4 * np.pi * epsilon_0 * epsilon_r * r_nm)

    # Convert to meV
    energy_meV = energy_J * J_to_meV

    return energy_meV

def one_rdm_binned(log_psi, params, samples, spins, atoms, charges,
                   r1, r2, h):
    """
    Estimate <c^\dagger(r1) c(r2)> using binning and wavefunction ratios.

    Args:
        log_psi: function(params, positions, spins, atoms, charges) -> scalar log(psi)
        params: NN parameters
        samples: (M, N*2) sampled particle coordinates
        spins, atoms, charges: sample-specific features
        r1, r2: (2,) arrays for positions
        h: bin width 
    """
    M, Nd = samples.shape
    N = Nd // 2
    coords = samples.reshape(M, N, 2)  # (M, N, 2)

    # Precompute distance weights for r2
    dist_r2 = jnp.linalg.norm(coords - r2[None, None, :], axis=-1)  # (M, N)
    #weights_r2 = jnp.exp(-0.5 * (dist_r2 / h) ** 2)

    def sample_contrib(sample_coords, sample_spin, sample_atoms, sample_charges, w_r2):
        contrib = 0.0 + 0.0j

        def particle_update(i, acc):
            w = w_r2[i]
            def move_particle():
                new_coords = sample_coords.at[i].set(r1)
                psi_ref = log_psi(params, sample_coords.flatten(), sample_spin, sample_atoms, sample_charges)
                psi_mod = log_psi(params, new_coords.flatten(), sample_spin, sample_atoms, sample_charges)
                return acc + jnp.exp(psi_mod - psi_ref) 
            return jax.lax.cond(w < h, move_particle, lambda: acc)

        contrib = jax.lax.fori_loop(0, N, particle_update, contrib)
        return contrib

    contribs = jax.vmap(sample_contrib)(coords, spins, atoms, charges, dist_r2)
    return jnp.mean(contribs)

def twoRDM_bandbasis(log_psi, samples, kinds,
                         k_list, band_wavefunctions, wavefuncs, Mcutoff, G1, G2, grid, area_lattice, eps=1e-12):
    """
    2RDM in band basis with symmetry: sum over i<j and include swapped contribution
    """
    M, Nd = samples.shape
    N = Nd // 2
    coords = samples.reshape(M, N, 2)

    def sample_contrib(sample_coords, sample_spin, sample_atoms, sample_charges):
        psi_ref = log_psi(pos =  sample_coords.flatten())
        contrib = 0.0 + 0.0j

        # sum over particle pairs i<j
        def pair_update(i, acc_i):
            def inner_loop(j, acc_j):
                r1s = jnp.repeat(grid, grid.shape[0], axis=0)
                r2s = jnp.tile(grid, (grid.shape[0],1))
                dist = jnp.linalg.norm(r1s - r2s, axis=1)
                weight = (dist > eps).astype(r1s.dtype)
                r_pairs = jnp.stack([r1s, r2s], axis=1)

                def integrand(r_pair, w):
                    return jax.lax.cond(
                        w>0,
                        lambda _: compute_pair_sym(r_pair, psi_ref, sample_coords, sample_spin, sample_atoms, sample_charges, i, j),
                        lambda _: 0.0 + 0.0j,
                        operand=None
                    )

                vals = jax.vmap(integrand)(r_pairs, weight)
                integral = jnp.sum(vals) * (area_lattice / grid.shape[0])**2
                return acc_j + integral

            return jax.lax.fori_loop(i+1, N, inner_loop, acc_i)

        contrib = jax.lax.fori_loop(0, N-1, pair_update, contrib)
        return contrib

    def compute_pair_sym(r_pair, psi_ref, sample_coords, sample_spin, sample_atoms, sample_charges, i, j):
        r1, r2 = r_pair
        new_coords = sample_coords.at[i].set(r1)
        new_coords = new_coords.at[j].set(r2)
        psi_mod = log_psi(pos =  new_coords.flatten())

        # original ordering
        b1 = band_wavefunction(r1, wavefuncs[0], k_list[0], Mcutoff, G1, G2)
        b2 = band_wavefunction(r2, wavefuncs[1], k_list[1], Mcutoff, G1, G2)
        b3 = band_wavefunction(sample_coords[i], wavefuncs[2], k_list[2], Mcutoff, G1, G2)
        b4 = band_wavefunction(sample_coords[j], wavefuncs[3], k_list[3], Mcutoff, G1, G2)
        b3s = band_wavefunction(sample_coords[j], wavefuncs[2], k_list[2], Mcutoff, G1, G2)
        b4s = band_wavefunction(sample_coords[i], wavefuncs[3], k_list[3], Mcutoff, G1, G2)
        val = jnp.exp(psi_mod - psi_ref) * (jnp.conj(b1) * jnp.conj(b2) * b3 * b4 - jnp.conj(b1) * jnp.conj(b2) * b3s * b4s)

        return val

    contribs = jax.vmap(sample_contrib)(coords, spins, atoms, charges)
    return jnp.mean(contribs)

def twoRDM_bandbasis_jit(log_psi, samples, kinds,
                         k_list,  band_wavefunction, wavefuncs, Mcutoff, G1, G2, grid, area_lattice, eps=1e-12):
    """
    Fully JIT-compatible, memory-efficient 2RDM in the band basis.
    Precompute band wavefunctions on the grid for efficiency.
    """
    M, Nd = samples.shape
    N = Nd // 2
    coords = samples.reshape(M, N, 2)

    # Precompute band wavefunctions on the grid
    grid_size = grid.shape[0]
    precomputed_bands = []
    for idx in kinds[:2]:  # for r1 and r2
        bvals = jax.vmap(lambda r: band_wavefunction(r, wavefuncs[idx], k_list[idx], Mcutoff, G1, G2))(grid)
        precomputed_bands.append(bvals)  # shape: (grid_size, )

    def sample_contrib(sample_coords):
        psi_ref = log_psi(pos = sample_coords.flatten())
        contrib = 0.0 + 0.0j

        # sum over particle pairs i<j
        def pair_update(i, acc_i):
            def inner_loop(j, acc_j):
                integral = 0.0 + 0.0j

                # nested loops over r1 and r2 indices
                def r1_loop(r1_idx, acc_r1):
                    r1 = grid[r1_idx]
                    b1_val = precomputed_bands[0][r1_idx]

                    def r2_loop(r2_idx, acc_r2):
                        r2 = grid[r2_idx]
                        b2_val = precomputed_bands[1][r2_idx]

                        dist = jnp.linalg.norm(r1 - r2)
                        val = jax.lax.cond(
                            dist > eps,
                            lambda _: compute_pair_sym_jit(r1, r2, b1_val, b2_val, psi_ref, sample_coords,i, j),
                            lambda _: 0.0 + 0.0j,
                            operand=None
                        )
                        return acc_r2 + val

                    acc_r1 += jax.lax.fori_loop(0, grid_size, r2_loop, 0.0 + 0.0j)
                    return acc_r1

                integral += jax.lax.fori_loop(0, grid_size, r1_loop, 0.0 + 0.0j)
                integral *= (area_lattice / grid_size)**2
                return acc_j + integral

            return jax.lax.fori_loop(i+1, N, inner_loop, acc_i)

        contrib = jax.lax.fori_loop(0, N-1, pair_update, contrib)
        return contrib

    def compute_pair_sym_jit(r1, r2, b1_val, b2_val, psi_ref, sample_coords, i, j):
        # replace coordinates
        new_coords = sample_coords.at[i].set(r1).at[j].set(r2)
        psi_mod = log_psi(pos = new_coords.flatten())

        # b3, b4 at original sample positions
        b3 = band_wavefunction(sample_coords[i], wavefuncs[kinds[2]], k_list[kinds[2]], Mcutoff, G1, G2)
        b4 = band_wavefunction(sample_coords[j], wavefuncs[kinds[3]], k_list[kinds[3]], Mcutoff, G1, G2)
        # swapped contribution
        b3s = band_wavefunction(sample_coords[j], wavefuncs[kinds[2]], k_list[kinds[2]], Mcutoff, G1, G2)
        b4s = band_wavefunction(sample_coords[i], wavefuncs[kinds[3]], k_list[kinds[3]], Mcutoff, G1, G2)

        return jnp.exp(psi_mod - psi_ref) * (jnp.conj(b1_val) * jnp.conj(b2_val))*(b3* b4 - b3s * b4s)

    # vmap over samples
    contribs = jax.vmap(sample_contrib)(coords)
    return jnp.mean(contribs)

def twoRDM_bandbasis_jit_vmapped(log_psi, samples, kinds,
                         k_list, band_wavefunction, wavefuncs, Mcutoff, G1, G2, grid, area_lattice, eps=1e-12):
    """
    Fully JIT-compatible, memory-efficient 2RDM in the band basis.
    Precompute band wavefunctions on the grid for efficiency.
    """
    M, Nd = samples.shape
    N = Nd // 2
    coords = samples.reshape(M, N, 2)

    # Precompute band wavefunctions on the grid
    grid_size = grid.shape[0]
    precomputed_bands = []
    for idx in kinds[:2]:  # for r1 and r2
        bvals = jax.vmap(lambda r: band_wavefunction(r, wavefuncs[idx], k_list[idx], Mcutoff, G1, G2))(grid)
        precomputed_bands.append(bvals)  # shape: (grid_size, )

    def sample_contrib(sample_coords):
        psi_ref = log_psi(pos=sample_coords.flatten())
        contrib = 0.0 + 0.0j

        # Sum over particle pairs i < j
        def pair_update(i, acc_i):
            def inner_loop(j, acc_j):
                # Vectorized r2 loop
                def r2_contrib(r2_idx, r1_idx):
                    r2 = grid[r2_idx]
                    b2_val = precomputed_bands[1][r2_idx]
                    r1 = grid[r1_idx]
                    b1_val = precomputed_bands[0][r1_idx]

                    dist = jnp.linalg.norm(r1 - r2)
                    return jax.lax.cond(
                        dist > eps,
                        lambda _: compute_pair_sym_jit(r1, r2, b1_val, b2_val, psi_ref, sample_coords, i, j),
                        lambda _: 0.0 + 0.0j,
                        operand=None
                    )

                # Vectorized r1 loop
                def r1_contrib(r1_idx):
                    r2_indices = jnp.arange(grid_size)
                    r2_contributions = jax.vmap(lambda r2_idx: r2_contrib(r2_idx, r1_idx))(r2_indices)
                    return jnp.sum(r2_contributions)

                r1_indices = jnp.arange(grid_size)
                integral = jnp.sum(jax.vmap(r1_contrib)(r1_indices))
                integral *= (area_lattice / grid_size) ** 2
                return acc_j + integral

            return jax.lax.fori_loop(i + 1, N, inner_loop, acc_i)

        contrib = jax.lax.fori_loop(0, N - 1, pair_update, contrib)
        return contrib

    def compute_pair_sym_jit(r1, r2, b1_val, b2_val, psi_ref, sample_coords, i, j):
        # Replace coordinates
        new_coords = sample_coords.at[i].set(r1).at[j].set(r2)
        psi_mod = log_psi(pos=new_coords.flatten())

        # b3, b4 at original sample positions
        b3 = band_wavefunction(sample_coords[i], wavefuncs[kinds[2]], k_list[kinds[2]], Mcutoff, G1, G2)
        b4 = band_wavefunction(sample_coords[j], wavefuncs[kinds[3]], k_list[kinds[3]], Mcutoff, G1, G2)
        # Swapped contribution
        b3s = band_wavefunction(sample_coords[j], wavefuncs[kinds[2]], k_list[kinds[2]], Mcutoff, G1, G2)
        b4s = band_wavefunction(sample_coords[i], wavefuncs[kinds[3]], k_list[kinds[3]], Mcutoff, G1, G2)

        return jnp.exp(psi_mod - psi_ref) * (jnp.conj(b1_val) * jnp.conj(b2_val)) * (b3 * b4 - b3s * b4s)

    # vmap over samples
    contribs = jax.vmap(sample_contrib)(coords)
    return jnp.mean(contribs)

def twoRDM_bandbasis_mc_importance(
    log_psi, samples, kinds,
    k_list, band_wavefunction, wavefuncs,
    Mcutoff, G1, G2 ,r1_points,r2_points
    , eps=1e-6
):
    """
    Importance sampling MC integration for 2RDM band basis elements.
    Samples r1, r2 according to |band_wavefunction(k1)|^2, |band_wavefunction(k2)|^2,
    excluding contributions where r1 ~ r2 (distance < eps).
    """

    
    M, Nd = samples.shape
    N = Nd // 2
    coords = samples.reshape(M, N, 2)
    n_mc_points = r1_points.shape[0]
    def sample_contrib(sample_coords):
        psi_ref = log_psi(pos = sample_coords.flatten())

        def pair_contrib(i, j):
            def mc_term(r1_idx, acc_r1):
                r1 = r1_points[r1_idx]

                def r2_loop(r2_idx, acc_r2):
                    r2 = r2_points[r2_idx]

                    # Skip contribution if r1 ≈ r2
                    dist = jnp.linalg.norm(r1 - r2)
                    def compute_term():
                        new_coords = sample_coords.at[i].set(r1).at[j].set(r2)
                        psi_mod = log_psi(pos = new_coords.flatten())

                        b3 = band_wavefunction(sample_coords[i], wavefuncs[kinds[2]], k_list[kinds[2]], Mcutoff, G1, G2)
                        b4 = band_wavefunction(sample_coords[j], wavefuncs[kinds[3]], k_list[kinds[3]], Mcutoff, G1, G2)
                        b3s = band_wavefunction(sample_coords[j], wavefuncs[kinds[2]], k_list[kinds[2]], Mcutoff, G1, G2)
                        b4s = band_wavefunction(sample_coords[i], wavefuncs[kinds[3]], k_list[kinds[3]], Mcutoff, G1, G2)

                        weight = jnp.exp(psi_mod - psi_ref)
                        integrand = (1/(band_wavefunction(r1, wavefuncs[kinds[0]], k_list[kinds[0]], Mcutoff, G1, G2))
                                     * (1/band_wavefunction(r2, wavefuncs[kinds[1]], k_list[kinds[1]], Mcutoff, G1, G2))
                                     * (b3 * b4 - b3s * b4s))

                        return weight * integrand / (n_mc_points ** 2)

                    term = jax.lax.cond(dist > eps, compute_term, lambda: 0.0+0.0j)
                    return acc_r2 + term

                return acc_r1 + jax.lax.fori_loop(0, n_mc_points, r2_loop, 0.0+0.0j)

            total = jax.lax.fori_loop(0, n_mc_points, mc_term, 0.0+0.0j)

            return total 

        def j_loop(i, acc_i):
            return acc_i + jax.lax.fori_loop(i+1, N, lambda j, acc_j: acc_j + pair_contrib(i, j), 0.0+0.0j)

        return jax.lax.fori_loop(0, N-1, j_loop, 0.0+0.0j)

    contribs = jax.vmap(sample_contrib)(coords)
    return jnp.mean(contribs)

def twoRDM_bandbasis_mc_importance_vmapped(
    log_psi, samples, kinds,
    k_list, band_wavefunction, wavefuncs,
    Mcutoff, G1, G2 ,r1_points,r2_points
    , eps=1e-6
):
    """
    Importance sampling MC integration for 2RDM band basis elements.
    Samples r1, r2 according to |band_wavefunction(k1)|^2, |band_wavefunction(k2)|^2,
    excluding contributions where r1 ~ r2 (distance < eps).
    """

    M, Nd = samples.shape
    N = Nd // 2
    coords = samples.reshape(M, N, 2)
    n_mc_points = r1_points.shape[0]

    def sample_contrib(sample_coords):
        psi_ref = log_psi(pos=sample_coords.flatten())

        def pair_contrib(i, j):
            def compute_r1_contrib(r1):
                def compute_r2_contrib(r2):
                    # Skip contribution if r1 ≈ r2
                    dist = jnp.linalg.norm(r1 - r2)

                    def compute_term():
                        new_coords = sample_coords.at[i].set(r1).at[j].set(r2)
                        psi_mod = log_psi(pos=new_coords.flatten())

                        b3 = band_wavefunction(sample_coords[i], wavefuncs[kinds[2]], k_list[kinds[2]], Mcutoff, G1, G2)
                        b4 = band_wavefunction(sample_coords[j], wavefuncs[kinds[3]], k_list[kinds[3]], Mcutoff, G1, G2)
                        b3s = band_wavefunction(sample_coords[j], wavefuncs[kinds[2]], k_list[kinds[2]], Mcutoff, G1, G2)
                        b4s = band_wavefunction(sample_coords[i], wavefuncs[kinds[3]], k_list[kinds[3]], Mcutoff, G1, G2)

                        weight = jnp.exp(psi_mod - psi_ref)
                        integrand = (1 / band_wavefunction(r1, wavefuncs[kinds[0]], k_list[kinds[0]], Mcutoff, G1, G2)
                                     * (1 / band_wavefunction(r2, wavefuncs[kinds[1]], k_list[kinds[1]], Mcutoff, G1, G2))
                                     * (b3 * b4 - b3s * b4s))

                        return weight * integrand / (n_mc_points ** 2)

                    return jax.lax.cond(dist > eps, compute_term, lambda: 0.0 + 0.0j)

                return jnp.sum(jax.vmap(compute_r2_contrib)(r2_points))

            return jnp.sum(jax.vmap(compute_r1_contrib)(r1_points))

        def j_loop(i, acc_i):
            return acc_i + jax.lax.fori_loop(i + 1, N, lambda j, acc_j: acc_j + pair_contrib(i, j), 0.0 + 0.0j)

        return jax.lax.fori_loop(0, N - 1, j_loop, 0.0 + 0.0j)

    contribs = jax.vmap(sample_contrib)(coords)
    return jnp.mean(contribs)

def build_band_pdf_and_sampler(band_wavefunction, wavefunc, kvec, Mcutoff, G1, G2, lattice, grid_size=64):
    """
    Build normalized PDF of |band_wavefunction|^2 over a grid defined 
    by a supercell with arbitrary lattice vectors (columns of `lattice`).
    
    Args:
        band_wavefunction: function r -> complex wavefunction value
        wavefunc, kvec, Mcutoff, G1, G2: wavefunction parameters
        lattice: (2, 2) array, columns are lattice vectors
        grid_size: number of points along each lattice direction
    
    Returns:
        sampler(key, n_samples) -> (samples, pdf_vals)
    """
    # Grid in fractional coordinates
    us = jnp.linspace(0, 1, grid_size)
    vs = jnp.linspace(0, 1, grid_size)
    frac_grid = jnp.stack(jnp.meshgrid(us, vs, indexing='ij'), axis=-1).reshape(-1, 2)  # (N,2)

    # Map to real space using lattice matrix
    grid = frac_grid @ lattice.T  # shape (N,2)

    # Evaluate |psi|^2
    vals = jax.vmap(lambda r: jnp.abs(band_wavefunction(r, wavefunc, kvec, Mcutoff, G1, G2))**2)(grid)
    pdf = vals / jnp.sum(vals)  # Normalize

    # Flatten PDF and compute CDF
    cdf = jnp.cumsum(pdf)
    cdf /= cdf[-1]

    def sampler(key, n_samples):
        # Sample uniformly in [0,1], invert CDF
        u = jax.random.uniform(key, (n_samples,))
        idx = jnp.searchsorted(cdf, u)
        chosen = grid[idx]
        return chosen, pdf[idx]

    return sampler

def enumerate_pairs(N_k):
    """
    Enumerate ordered distinct momentum pairs (a,b) with a < b.
    Returns:
      pairs: (D,2) int array of pairs
      pair_to_index: dict mapping (a,b)-> idx
      D: number of pairs = C(N_k,2)
    """
    pairs = [(a, b) for a in range(N_k) for b in range(a+1, N_k)]
    D = len(pairs)
    pairs_arr = jnp.array(pairs, dtype=jnp.int32)  # (D,2)
    pair_to_index = {pair: idx for idx, pair in enumerate(pairs)}
    return pairs_arr, pair_to_index, D

def build_half_index_table(N_k):
    pairs_arr, pair_to_index, D = enumerate_pairs(N_k)

    # build lower-triangle index lists
    prow_list = []
    pcol_list = []
    for prow in range(D):
        for pcol in range(prow + 1):
            prow_list.append(prow)
            pcol_list.append(pcol)
    prow_arr = jnp.array(prow_list, dtype=jnp.int32)
    pcol_arr = jnp.array(pcol_list, dtype=jnp.int32)

    # map prow/pcol to (k1,k2,k3,k4)
    k1s = pairs_arr[prow_arr, 0]
    k2s = pairs_arr[prow_arr, 1]
    k3s = pairs_arr[pcol_arr, 0]
    k4s = pairs_arr[pcol_arr, 1]

    # NEW: dense JAX array for pair -> index
    pair_index_table = -jnp.ones((N_k, N_k), dtype=jnp.int32)
    pair_index_table = pair_index_table.at[
        pairs_arr[:, 0], pairs_arr[:, 1]
    ].set(jnp.arange(D, dtype=jnp.int32))

    return {
        "k1s": k1s, "k2s": k2s, "k3s": k3s, "k4s": k4s,
        "row_idx": prow_arr, "col_idx": pcol_arr,
        "pairs_arr": pairs_arr, "pair_to_index": pair_to_index, "pair_index_table": pair_index_table,
        "D": D
    }

def build_full_index_table(N_k):
    pairs_arr, pair_to_index, D = enumerate_pairs(N_k)

    # Build full index lists (all combinations of rows and columns)
    prow_list = []
    pcol_list = []
    for prow in range(D):
        for pcol in range(D):  # Include all columns, not just lower triangle
            prow_list.append(prow)
            pcol_list.append(pcol)
    prow_arr = jnp.array(prow_list, dtype=jnp.int32)
    pcol_arr = jnp.array(pcol_list, dtype=jnp.int32)

    # Map prow/pcol to (k1, k2, k3, k4)
    k1s = pairs_arr[prow_arr, 0]
    k2s = pairs_arr[prow_arr, 1]
    k3s = pairs_arr[pcol_arr, 0]
    k4s = pairs_arr[pcol_arr, 1]

    # Dense JAX array for pair -> index
    pair_index_table = -jnp.ones((N_k, N_k), dtype=jnp.int32)
    pair_index_table = pair_index_table.at[
        pairs_arr[:, 0], pairs_arr[:, 1]
    ].set(jnp.arange(D, dtype=jnp.int32))

    return {
        "k1s": k1s, "k2s": k2s, "k3s": k3s, "k4s": k4s,
        "row_idx": prow_arr, "col_idx": pcol_arr,
        "pairs_arr": pairs_arr, "pair_to_index": pair_to_index, "pair_index_table": pair_index_table,
        "D": D
    }

def map_full_to_reduced(k1, k2, k3, k4, pair_index_table):
    # canonicalize each pair
    a = jnp.minimum(k1, k2)
    b = jnp.maximum(k1, k2)
    s1 = jnp.where(k1 < k2, 1, -1).astype(jnp.int32)

    c = jnp.minimum(k3, k4)
    d = jnp.maximum(k3, k4)
    s2 = jnp.where(k3 < k4, 1, -1).astype(jnp.int32)

    row = pair_index_table[a, b]
    col = pair_index_table[c, d]
    sign = (s1 * s2).astype(jnp.int32)
    return row, col, sign

    
def build_reduced_2RDM_from_enumeration(compute_element_fn, log_psi, samples, 
                         k_list, band_wavefunction, wavefuncs, Mcutoff, G1, G2, grid, area_lattice, enum, D, jit_compile=True):
    """
    Build a Hermitian 2RDM matrix from pre-enumerated antisymmetrized indices.
    
    Args:
        compute_element_fn: function (k1, k2, k3, k4) -> complex scalar
        enum: dict containing arrays:
            - k1, k2, k3, k4: arrays of indices
            - row_idx, col_idx: arrays of matrix indices
        D: dimension of reduced 2RDM
        jit_compile: if True, JIT compile the builder function
    
    Returns:
        builder(): function that returns a (D,D) Hermitian 2RDM matrix
    """
    k1_arr = enum["k1s"]
    k2_arr = enum["k2s"]
    k3_arr = enum["k3s"]
    k4_arr = enum["k4s"]
    row_idx = enum["row_idx"]
    col_idx = enum["col_idx"]
    L = k1_arr.shape[0]  # number of enumerated elements

    # Wrapper: index -> matrix element
    def compute_element_by_index(idx):
        k1 = k1_arr[idx]
        k2 = k2_arr[idx]
        k3 = k3_arr[idx]
        k4 = k4_arr[idx]
        kinds1 = jnp.array([k1, k2, k3, k4])
        #kinds2 = jnp.array([k2, k1, k3, k4])
        #kinds3 = jnp.array([k1, k2, k4, k3])
        #kinds4 = jnp.array([k2, k1, k4, k3])
        out = compute_element_fn(log_psi, samples, kinds1, k_list, band_wavefunction, wavefuncs, Mcutoff, G1, G2, grid, area_lattice, eps=1e-12) #+ compute_element_fn(log_psi, samples, kinds4,k_list, band_wavefunction, wavefuncs, Mcutoff, G1, G2, grid, area_lattice, eps=1e-12) - compute_element_fn(log_psi, samples, kinds2, k_list, band_wavefunction, wavefuncs, Mcutoff, G1, G2, grid, area_lattice, eps=1e-12) - compute_element_fn(log_psi, samples, kinds3,k_list, band_wavefunction, wavefuncs, Mcutoff, G1, G2, grid, area_lattice, eps=1e-12)
        return out

    def builder():
        # Compute all values vectorized
        #values = jax.vmap(compute_element_by_index)(jnp.arange(L, dtype=jnp.int32))

        values = jnp.zeros(L, dtype=jnp.complex64)  # Adjust dtype as needed

        # Define the loop body
        def loop_body(i, acc):
            return acc.at[i].set(compute_element_by_index(i))

        # Use jax.lax.fori_loop for the loop
        values = jax.lax.fori_loop(0, L, loop_body, values)
        # Initialize lower-triangular matrix
        mat0 = jnp.zeros((D, D), dtype=jnp.complex64)

        def body_fun(i, mat):
            r = row_idx[i]
            c = col_idx[i]
            val = values[i]
            mat = mat.at[r, c].set(val)
            return mat

        mat_lower = jax.lax.fori_loop(0, L, body_fun, mat0)
        # Fill upper-triangular part by Hermitian conjugation
        mat_full = mat_lower + jnp.tril(mat_lower, k=-1).conj().T
        return mat_full

    if jit_compile:
        return jax.jit(builder)
    else:
        return builder

def build_reduced_2RDM_from_enumeration_mc(compute_element_fn, log_psi, samples, 
                         k_list, band_wavefunction, wavefuncs,samplerlist, Mcutoff, G1, G2, enum, D, n_mcpoints, jit_compile=True):
    """
    Build a Hermitian 2RDM matrix from pre-enumerated antisymmetrized indices.
    
    Args:
        compute_element_fn: function (k1, k2, k3, k4) -> complex scalar
        enum: dict containing arrays:
            - k1, k2, k3, k4: arrays of indices
            - row_idx, col_idx: arrays of matrix indices
        D: dimension of reduced 2RDM
        jit_compile: if True, JIT compile the builder function
    
    Returns:
        builder(): function that returns a (D,D) Hermitian 2RDM matrix
    """
    k1_arr = enum["k1s"]
    k2_arr = enum["k2s"]
    k3_arr = enum["k3s"]
    k4_arr = enum["k4s"]
    row_idx = enum["row_idx"]
    col_idx = enum["col_idx"]
    L = k1_arr.shape[0]  # number of enumerated elements


    # Wrapper: index -> matrix element
    def compute_element_by_index(idx):
        k1 = k1_arr[idx]
        k2 = k2_arr[idx]
        k3 = k3_arr[idx]
        k4 = k4_arr[idx]
        kinds1 = jnp.array([k1, k2, k3, k4])
        kinds2 = jnp.array([k2, k1, k3, k4])
        kinds3 = jnp.array([k1, k2, k4, k3])
        kinds4 = jnp.array([k2, k1, k4, k3])


        key = jax.random.PRNGKey(0) #using same seed, change later to a random seed
        key_r1, key_r2 = jax.random.split(key)
        r1_points, p_r1 = jax.lax.switch(k1, samplerlist,key_r1)
        r2_points, p_r2 = jax.lax.switch(k2, samplerlist,key_r2)

        #twoRDM_bandbasis_mc_importance(log_psi, params, samples, spins, atoms, charges, kinds,k_list, band_wavefunction, wavefuncs,Mcutoff, G1, G2,r1_points,r2_points, eps=1e-6)
        #Below can be further optimized
        out1 = compute_element_fn(log_psi, samples, kinds1, k_list, band_wavefunction, wavefuncs, Mcutoff, G1, G2, r1_points, r2_points, eps=1e-12)
        out2 = compute_element_fn(log_psi, samples, kinds2, k_list, band_wavefunction, wavefuncs, Mcutoff, G1, G2, r2_points, r1_points, eps=1e-12)
        out3 = compute_element_fn(log_psi, samples, kinds3, k_list, band_wavefunction, wavefuncs, Mcutoff, G1, G2, r1_points, r2_points, eps=1e-12)
        out4 = compute_element_fn(log_psi, samples, kinds4, k_list, band_wavefunction, wavefuncs, Mcutoff, G1, G2, r2_points, r1_points, eps=1e-12)
        return out1 - out2 - out3 + out4

    def builder():
        # Compute all values vectorized
        #values = jax.vmap(compute_element_by_index)(jnp.arange(L, dtype=jnp.int32))
        values = jnp.zeros(L, dtype=jnp.complex64)  # Adjust dtype as needed

        # Define the loop body
        def loop_body(i, acc):
            return acc.at[i].set(compute_element_by_index(i))

        # Use jax.lax.fori_loop for the loop
        values = jax.lax.fori_loop(0, L, loop_body, values)

        # Initialize lower-triangular matrix
        mat0 = jnp.zeros((D, D), dtype=jnp.complex64)

        def body_fun(i, mat):
            r = row_idx[i]
            c = col_idx[i]
            val = values[i]
            mat = mat.at[r, c].set(val)
            return mat

        mat_lower = jax.lax.fori_loop(0, L, body_fun, mat0)
        # Fill upper-triangular part by Hermitian conjugation
        mat_full = mat_lower + jnp.tril(mat_lower, k=-1).conj().T
        return mat_full

    if jit_compile:
        return jax.jit(builder)
    else:
        return builder

def build_reduced_2RDM_from_enumeration_mc_full(compute_element_fn, log_psi, samples, 
                         k_list, band_wavefunction, wavefuncs,samplerlist, Mcutoff, G1, G2, enum, D, n_mcpoints, jit_compile=True):
    """
    Build a Hermitian 2RDM matrix from pre-enumerated antisymmetrized indices.
    
    Args:
        compute_element_fn: function (k1, k2, k3, k4) -> complex scalar
        enum: dict containing arrays:
            - k1, k2, k3, k4: arrays of indices
            - row_idx, col_idx: arrays of matrix indices
        D: dimension of reduced 2RDM
        jit_compile: if True, JIT compile the builder function
    
    Returns:
        builder(): function that returns a (D,D) Hermitian 2RDM matrix
    """
    k1_arr = enum["k1s"]
    k2_arr = enum["k2s"]
    k3_arr = enum["k3s"]
    k4_arr = enum["k4s"]
    row_idx = enum["row_idx"]
    col_idx = enum["col_idx"]
    L = k1_arr.shape[0]  # number of enumerated elements


    # Wrapper: index -> matrix element
    def compute_element_by_index(idx):
        k1 = k1_arr[idx]
        k2 = k2_arr[idx]
        k3 = k3_arr[idx]
        k4 = k4_arr[idx]
        kinds1 = jnp.array([k1, k2, k3, k4])
        kinds2 = jnp.array([k2, k1, k3, k4])
        kinds3 = jnp.array([k1, k2, k4, k3])
        kinds4 = jnp.array([k2, k1, k4, k3])


        key = jax.random.PRNGKey(0) #using same seed, change later to a random seed
        key_r1, key_r2 = jax.random.split(key)
        r1_points, p_r1 = jax.lax.switch(k1, samplerlist,key_r1)
        r2_points, p_r2 = jax.lax.switch(k2, samplerlist,key_r2)

        #twoRDM_bandbasis_mc_importance(log_psi, params, samples, spins, atoms, charges, kinds,k_list, band_wavefunction, wavefuncs,Mcutoff, G1, G2,r1_points,r2_points, eps=1e-6)
        #Below can be further optimized
        out1 = compute_element_fn(log_psi, samples, kinds1, k_list, band_wavefunction, wavefuncs, Mcutoff, G1, G2, r1_points, r2_points, eps=1e-12)
        out2 = compute_element_fn(log_psi, samples, kinds2, k_list, band_wavefunction, wavefuncs, Mcutoff, G1, G2, r2_points, r1_points, eps=1e-12)
        out3 = compute_element_fn(log_psi, samples, kinds3, k_list, band_wavefunction, wavefuncs, Mcutoff, G1, G2, r1_points, r2_points, eps=1e-12)
        out4 = compute_element_fn(log_psi, samples, kinds4, k_list, band_wavefunction, wavefuncs, Mcutoff, G1, G2, r2_points, r1_points, eps=1e-12)
        return out1 - out2 - out3 + out4

    def builder():
        # Compute all values vectorized
        #values = jax.vmap(compute_element_by_index)(jnp.arange(L, dtype=jnp.int32))
        values = jnp.zeros(L, dtype=jnp.complex64)  # Adjust dtype as needed

        # Define the loop body
        def loop_body(i, acc):
            return acc.at[i].set(compute_element_by_index(i))

        # Use jax.lax.fori_loop for the loop
        values = jax.lax.fori_loop(0, L, loop_body, values)

        # Initialize lower-triangular matrix
        mat0 = jnp.zeros((D, D), dtype=jnp.complex64)

        def body_fun(i, mat):
            r = row_idx[i]
            c = col_idx[i]
            val = values[i]
            mat = mat.at[r, c].set(val)
            return mat

        mat = jax.lax.fori_loop(0, L, body_fun, mat0)

        return mat

    if jit_compile:
        return jax.jit(builder)
    else:
        return builder



def Ham(k, M, b1, b2, V0, phi, kin, λ_0, Alist, phi2list, momentumfouriergrid):
    size = (2 * M + 1) ** 2
    H = np.zeros((size, size), dtype=np.complex128)

    for m in range(-M, M + 1):
        for n in range(-M, M + 1):
            p = (n + M) + (m + M) * (2 * M + 1)
            kt = k + m * b1 + n * b2
            H[p, p] += kin * np.linalg.norm(kt) ** 2

            if m != M:
                x = (n + M) + (m + 1 + M) * (2 * M + 1)
                H[x, p] += V0 * np.exp(1j * phi) + (2 * kin * λ_0) * np.dot(kt + b1, Alist[0]) * np.exp(1j * phi2list[0])

            if n != -M:
                x = (n - 1 + M) + (m + M) * (2 * M + 1)
                H[x, p] += V0 * np.exp(1j * phi) + (2 * kin * λ_0) * np.dot(kt - b2, Alist[4]) * np.exp(1j * phi2list[4])

            if n != M and m != -M:
                x = (n + 1 + M) + (m - 1 + M) * (2 * M + 1)
                H[x, p] += V0 * np.exp(1j * phi) + (2 * kin * λ_0) * np.dot(kt - b1 + b2, Alist[2]) * np.exp(1j * phi2list[2])

            if m != -M:
                x = (n + M) + (m - 1 + M) * (2 * M + 1)
                H[x, p] += V0 * np.exp(-1j * phi) + (2 * kin * λ_0) * np.dot(kt - b1, Alist[3]) * np.exp(1j * phi2list[3])

            if n != M:
                x = (n + 1 + M) + (m + M) * (2 * M + 1)
                H[x, p] += V0 * np.exp(-1j * phi) + (2 * kin * λ_0) * np.dot(kt + b2, Alist[1]) * np.exp(1j * phi2list[1])

            if n != -M and m != M:
                x = (n - 1 + M) + (m + 1 + M) * (2 * M + 1)
                H[x, p] += V0 * np.exp(-1j * phi) + (2 * kin * λ_0) * np.dot(kt + b1 - b2, Alist[5]) * np.exp(1j * phi2list[5])

            # Additional Fourier grid contributions
            for m1 in range(-M, M + 1):
                for n1 in range(-M, M + 1):
                    x = (n - n1 + M) + (m - m1 + M) * (2 * M + 1)
                    y = (n1 + M) + (m1 + M) * (2 * M + 1)
                    if 0 <= x < size:
                        H[x, p] += kin * (λ_0 ** 2) * momentumfouriergrid[y]

    return (H + H.T.conj()) / 2  # Ensure Hermitian matrix

def band_wavefunction(r, wavefunc, k, M, G1, G2):
    # Create the grid of m and n values
    m = jnp.arange(-M, M + 1)
    n = jnp.arange(-M, M + 1)
    m_grid, n_grid = jnp.meshgrid(m, n, indexing='ij')

    # Compute the linear index p
    p = (n_grid + M) + (m_grid + M) * (2 * M + 1)

    # Compute the reciprocal lattice vectors G
    G = m_grid[..., None] * G1 + n_grid[..., None] * G2

    # Compute the wavefunction contributions
    real_wavefunc = jnp.sum(
        wavefunc[p] * jnp.exp(1j * jnp.dot(G + k, r)),
        axis=(0, 1)
    )

    return real_wavefunc

########################## Main code ##########################

def get_config():
  # Get default options.
  cfg = base_config.default()
  cfg.system.electrons = (8, 0)
  cfg.system.ndim = 2
  # A ghost atom at the origin defines one-electron coordinate system.
  # Element 'X' is a dummy nucleus with zero charge
  cfg.system.molecule = [system.Atom("X", (0., 0.,))]
  # Pretraining is not currently implemented for systems in PBC
  cfg.pretrain.method = None

  """ Defining the potential unit cell and the supercell """
  a0 = 1.0
  a1 = a0 * np.array([np.sqrt(3)/2,-0.5])
  #a1 = a0* np.array([1,0])
  a2 = a0 * np.array([0,1])
  Tmatrix = np.array([[4,0], [0, 6]]) 
  lattice = lattice_vecs(a1, a2, Tmatrix)
  potential_lattice = lattice_vecs(a1, a2, np.array([[1,0], [0, 1]]))
  #kpoints = envelopes.make_kpoints(lattice, cfg.system.electrons)

  """Defining KE, potential and interaction parameters"""
  meff = 1.0
  KE_prefactor = hbar2_over_m_eff(meff)
  print(KE_prefactor)
  pp_coffs = np.array([0.0, 0.0, 0.0])  
  pp_phases = np.array([0.0, 0.0, 0.0])
  intcoff = (coulomb_prefactor(5.0))/KE_prefactor
  print(intcoff)
  cfg.system.make_local_energy_fn = "ferminet.pbc.Hamiltonian_minimalChern.local_energy"
  cfg.system.make_local_energy_kwargs = {"lattice": lattice, "heg": True,"potential_kwargs": {"laplacian_method": "folx","interaction_energy_scale": intcoff},"kinetic_energy_kwargs": {"prefactor": KE_prefactor}, "periodic_lattice": potential_lattice,"periodic_potential_kwargs": {"coefficients": pp_coffs, "phases": pp_phases}}
  cfg.network.network_type = "psiformer"
  cfg.network.complex = True
  cfg.network.psiformer.num_layers = 4
  cfg.network.psiformer.num_heads = 6
  cfg.network.psiformer.heads_dim = 64
  cfg.network.psiformer.mlp_hidden_dims  = (256,)
  cfg.network.determinants = 4
  cfg.batch_size = 1024
  #cfg.mcmc.move_width = 2.0
  #cfg.mcmc.init_width = 3.0
  #cfg.mcmc.steps = 20
  #cfg.initialization.modifications = ['orbital-rnd']
  cfg.network.make_feature_layer_fn = (
      "ferminet.pbc.feature_layer.make_pbc_feature_layer")
  cfg.network.make_feature_layer_kwargs = {
      "lattice": lattice,
      "include_r_ae": False,
  }
  cfg.network.jastrow = 'none'
  cfg.network.make_envelope_fn = ( "ferminet.envelopes.make_null_envelope" )
  #cfg.network.make_envelope_kwargs = {"kpoints": kpoints}
  cfg.network.full_det = True
  return cfg


cfg = get_config()
log_network = wavefunction(cfg)


#files = load_recent_npz_files("/ceph/submit/data/user/a/ahmed95/ferminet_2025_08_10_14:19:09/", 1)
files = load_recent_npz_files("/work/submit/ahmed95/ferminet_ahmed/ferminet_2025_08_24_11:33:55 copy/", 1)

t_init, data, params, opt_state_ckpt, mcmc_width_ckpt, density_state_ckpt = read_and_combine_checkpoints(files,1024)
gathered_params = jax.tree.map(lambda x: jax.device_get(x[0]), params)

"Repeated in the function get_config (to fix later)"
a0 = 1.0
a1 = a0 * np.array([np.sqrt(3)/2,-0.5])
a2 = a0 * np.array([0,1])
Tmatrix = np.array([[4,0], [0, 6]]) 
lattice = lattice_vecs(a1, a2, Tmatrix)
area_lattice = jnp.abs(jnp.linalg.det(lattice))
potential_lattice = lattice_vecs(a1, a2, np.array([[1,0], [0, 1]]))
rec = 2*jnp.pi*jnp.linalg.inv(lattice)
potential_rec = 2*jnp.pi*jnp.linalg.inv(potential_lattice)


#Some necessary reshaping
reshaped_positions = data.positions.reshape(-1, data.positions.shape[-1])
reshaped_spins = data.spins.reshape(-1, data.spins.shape[-1])
new_shape = (data.atoms.shape[0] * data.atoms.shape[1],) + data.atoms.shape[2:]
reshaped_atoms = data.atoms.reshape(new_shape)
reshaped_charges = data.charges.reshape(-1, data.charges.shape[-1])
reshaped_positions_mapped = map_to_supercell_jax(reshaped_positions, lattice) 

log_network_fixed = partial(log_network, params=gathered_params, spins=reshaped_spins[0], atoms=reshaped_atoms[0], charges=reshaped_charges[0])
##spins, atoms, charges are redundant here

### Non-interacting bands (repeated parameters, to be fixed later) ###

def Rot(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], 
                     [np.sin(theta), np.cos(theta)]])


# Parameters
M = 6  # cutoff
Mlist = [np.array([i, j]) - np.array([M + 1, M + 1]) for i in range(1, 2 * M + 2) for j in range(1, 2 * M + 2)]

aM = 1.0  # nm

prefactor = 495  # meV·Å²
meff = 1.0  # effective mass
dMBZ = (4 * np.pi / (3 * aM))

# Constants
hbar = 4.135667 * 10**3 * 10**(-15) / (2 * np.pi)  # meV·s
me = 0.51099895 * 10**3 * 10**6  # meV/c²
m_eff = meff * me
kin = hbar**2 / (2 * m_eff) * 9 * 10**16
kin = kin / (1 * 10**-18)  # meV·nm²

esquare = (1.602176634 * 10**(-19))**2
epsilon_0 = 8.8541878128 * 10**(-12)

# Reciprocal lattice vectors
b1 = (4 * np.pi / (np.sqrt(3) * aM)) * np.array([1, 0])
b2 = Rot(np.pi / 3) @ b1
b3 = Rot(2 * np.pi / 3) @ b1
b4 = Rot(3 * np.pi / 3) @ b1
b5 = Rot(4 * np.pi / 3) @ b1
b6 = Rot(5 * np.pi / 3) @ b1

G1 = b1
G2 = b2
blist = [b1, b2, b3, b4, b5, b6]


# Real-space lattice vectors
a1 = aM * np.array([np.sqrt(3) / 2, -1 / 2])
a2 = aM * np.array([0, 1])

# Other parameters
V0 = 0.0
phi = 0.0
phi2 = 0.0
lambda_0 = -0.23  # dimensionless parameter
# Alist and phi2list
Alist = [[1j * b[1], -1j * b[0]] for b in blist]
phi2list = [phi2, -phi2, phi2, -phi2, phi2, -phi2]

momentumfouriergrid = np.load("/work/submit/ahmed95/ferminet_ahmed/momentumfouriergrid.npy")

### Initialize k-points and band wavefunctions #################

g1 = rec[0,:]  # Example vector
g2 = rec[1,:]  # Example vector
# Define the range for m and n
m_range = np.arange(0, 4)  # Range for m (e.g., 0 to 4)
n_range = np.arange(0, 6)  # Range for n (e.g., 0 to 9)

# Create a grid of m and n values
m_grid, n_grid = np.meshgrid(m_range, n_range)

# Compute k = m * g1 + n * g2 for all combinations
kgrid = m_grid[..., None] * g1 + n_grid[..., None] * g2  # Shape: (len(n_range), len(m_range), 2)

# Flatten the grid to get a list of k points
kgrid = kgrid.reshape(-1, 2)  # Shape: (num_k_points, 2)


# Initialize an empty list to store kpt_wavefunc for each k
kpt_wavefuncs = []

for k in kgrid:
    H = Ham(k, M, b1, b2, V0, phi, kin, lambda_0, Alist, phi2list, momentumfouriergrid) 
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    # Get the eigenfunctions corresponding to the smallest eigenvalue
    sorted_indices = np.argsort(eigenvalues)  # Indices to sort eigenvalues
    smallest_n_indices = sorted_indices[:1]  # Indices of the smallest n eigenvalues
    kpt_wavefunc = eigenvectors[:, smallest_n_indices]  # Corresponding eigenvectors
    kpt_wavefunc = kpt_wavefunc[:, 0]  # Extract the first eigenvector
    kpt_wavefunc = jnp.array(kpt_wavefunc)  # Convert to JAX array
    kpt_wavefuncs.append(kpt_wavefunc)  # Append to the list

# Convert the list to a JAX array of shape (len(kgrid), ...)
kpt_wavefuncs = jnp.array(kpt_wavefuncs)

################### Prepare band wavefunction samplers #################

"""prop_grid_size = 500
n_mcpoints = 40

samplerlist = tuple(build_band_pdf_and_sampler(band_wavefunction, kpt_wavefuncs[k], kgrid[k], Mcutoff=M, G1=G1, G2=G2, lattice=lattice, grid_size=prop_grid_size) for k in range(len(kgrid)))
samplerlist = tuple(partial(sampler, n_samples=n_mcpoints) for sampler in samplerlist)


print("Initialization done")"""


##### Real space grid for integration #####

numx = 10
numy = 10
dAgrid = area_lattice / (numx**2)
rlist = [i * (lattice[:,0] / numx) + j * (lattice[:,1] / numy) for i in range(0, numx) for j in range(0, numy)]
rlist = jnp.array(rlist)

print("Initialization done")

####################### Construct 2RDM #####################

"""N_k = len(kgrid)
enum = build_half_index_table(N_k)
D = enum["D"]

builder = build_reduced_2RDM_from_enumeration(twoRDM_bandbasis_jit, log_network, gathered_params, reshaped_positions_mapped, reshaped_spins, reshaped_atoms, reshaped_charges, 
                         kgrid, band_wavefunction, kpt_wavefuncs, M, G1, G2, rlist, area_lattice, enum, D, jit_compile=True)
Gamma = builder()   # this returns a (D,D) Hermitian matrix

np.save("Gamma2.npy", np.array(Gamma)) """

N_k = len(kgrid)
enum = build_half_index_table(N_k)
#enum = build_full_index_table(N_k)
D = enum["D"]

#import jax.profiler

#profiler.start_trace("/work/submit/ahmed95/jax_trace")
builder = build_reduced_2RDM_from_enumeration(twoRDM_bandbasis_jit, log_network_fixed, reshaped_positions_mapped, 
                         kgrid, band_wavefunction, kpt_wavefuncs,M, G1, G2, rlist, area_lattice,enum, D,jit_compile=True)
Gamma = builder()   # this returns a (D,D) Hermitian matrix

#profiler.stop_trace()

np.save("Gamma_8particles_nosampling.npy", np.array(Gamma)) 