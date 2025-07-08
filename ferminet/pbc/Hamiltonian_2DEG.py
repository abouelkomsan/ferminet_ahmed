# Copyright 2025 Max Geier (MIT)
# arXiv:2502.05383

"""Ewald summation of Coulomb Hamiltonian in a two-dimensional periodic system.

See M. Geier, K. Nazaryan, T. Zaklama, and L. Fu, arXiv:2502.05383 for a derivation of the Ewald summation.
"""

import itertools
from typing import Callable, Optional, Sequence, Tuple

import chex
from ferminet import hamiltonian
from ferminet import networks
import jax
import jax.numpy as jnp
import numpy as np
from jax.debug import print as jprint

#from ferminet import fwdlap
from jax import lax
from ferminet.utils import utils


def make_2DCoulomb_potential(
    lattice: jnp.ndarray,
    atoms: jnp.ndarray,
    charges: jnp.ndarray,
    truncation_limit: int = 5,
    interaction_energy_scale: float = 1.0,
) -> Callable[[jnp.ndarray, jnp.ndarray], float]:
  """Creates a function to evaluate infinite Coulomb sum for periodic lattice.

    Args:
        lattice: Shape (2, 2). Matrix whose columns are the primitive lattice vectors.
        atoms: Shape (natoms, 2). Positions of the atoms.
        charges: Shape (natoms). Nuclear charges of the atoms.
        nspins: Tuple of the number of spin-up and spin-down electrons.
        truncation_limit: Integer. Half side length of square of nearest neighbours
        to primitive cell which are summed over in evaluation of Ewald sum.

    Returns:
        Callable with signature f(ae, ee, spins), where (ae, ee) are atom-electron and
        electron-electron displacement vectors respectively, and spins are electron spins,
        which evaluates the Coulomb sum for the periodic lattice via the Ewald method.
  """
  del atoms, charges # unused for 2d system without atoms
  print("making 2DCoulomb potential with energy scale: " + str(interaction_energy_scale))
  rec = 2 * jnp.pi * jnp.linalg.inv(lattice)
  #volume = jnp.abs(jnp.linalg.det(lattice))
  area = jnp.abs(jnp.linalg.det(lattice)) #area for 2D
  # the factor gamma tunes the width of the summands in real / reciprocal space
  # and this value is chosen to optimize the convergence trade-off between the
  # two sums. See CASINO QMC manual.
  gamma_factor = 2.8
  gamma = (gamma_factor / area**(1 / 2))**2  # Adjusted for 2D systems

  ordinals = sorted(range(-truncation_limit, truncation_limit + 1), key=abs)
  ordinals = jnp.array(list(itertools.product(ordinals, repeat=2)))  # Adjusted for 2D
  lat_vectors = jnp.einsum('kj,ij->ik', lattice, ordinals)
  rec_vectors = jnp.einsum('jk,ij->ik', rec, ordinals[1:])
  rec_vec_square = jnp.einsum('ij,ij->i', rec_vectors, rec_vectors)
  rec_vec_magnitude = jnp.sqrt(rec_vec_square) # |rec_vectors|, same as kappa
  lat_vec_norm = jnp.linalg.norm(lat_vectors[1:], axis=-1)

  def real_space_ewald(separation: jnp.ndarray):
      """Real-space Ewald potential between charges in 2D.
      """
      displacements = jnp.linalg.norm(separation - lat_vectors, axis=-1)  # |r - R|

      return jnp.sum(
          jax.scipy.special.erfc(gamma**0.5 * displacements) / displacements)

  def recp_space_ewald(separation: jnp.ndarray):
      """Reciprocal-space Ewald potential between charges in 2D.
      """
      phase = jnp.cos(jnp.dot(rec_vectors, separation))

      factor = jax.scipy.special.erfc(rec_vec_magnitude / (2 * gamma**0.5) )
      return (2 * jnp.pi / area) * jnp.sum( phase * factor / rec_vec_magnitude)

  def ewald_sum(separation: jnp.ndarray):
      """Combined real and reciprocal space Ewald potential in 2D.
      """
      return real_space_ewald(separation) + recp_space_ewald(separation)
      
  # Compute Madelung constant components
  # Real-space part
  # xi_S_0 = 0 * gamma**0.5 / jnp.pi**0.5
  madelung_real = jnp.sum(
      jax.scipy.special.erfc(gamma**0.5 * lat_vec_norm) / lat_vec_norm
  )

  # q = 0 contribution of the real-space part
  phi_S_q0 = (2 * jnp.pi) / area / gamma**0.5 / jnp.pi**0.5

  # Reciprocal-space part
  xi_L_0 = 2 * gamma**0.5 / jnp.pi**0.5
  madelung_recip = - 0*(2 * jnp.pi / area) * (1 / (gamma**0.5 * jnp.pi**0.5)) + \
      (2 * jnp.pi / area) * jnp.sum(
          jax.scipy.special.erfc(rec_vec_magnitude / (2 * gamma**0.5)) / rec_vec_magnitude
      ) - xi_L_0

  # Total Madelung constant
  madelung_const = madelung_real + madelung_recip
  batch_ewald_sum = jax.vmap(ewald_sum, in_axes=(0,))

  def electron_electron_potential(ee: jnp.ndarray):
      """Evaluates periodic electron-electron potential with charges.

      We always include neutralizing background term for homogeneous electron gas.
      """
      nelec = ee.shape[0]
      ee = jnp.reshape(ee, [-1, 2])
      ewald = batch_ewald_sum(ee)
      ewald = jnp.reshape(ewald, [nelec, nelec])
      # Set diagonal elements to zero (self-interaction)
      ewald = ewald.at[jnp.diag_indices(nelec)].set(0.0)

      # Add Madelung constant term: (1/2) * N * q_i^2 * Madelung_const
      # Since q_i^2 = 1, this simplifies to (1/2) * N * Madelung_const
      potential = 0.5 * jnp.sum(ewald) + 0.5 * nelec * madelung_const - 0.5 * nelec**2 * phi_S_q0
      return potential

  def potential(ae: jnp.ndarray, ee: jnp.ndarray):
    """Accumulates atom-electron, atom-atom, and electron-electron potential."""
    # Reduce vectors into first unit cell
    del ae # for HEG calculations, there are no atoms
    phase_ee = jnp.einsum('il,jkl->jki', rec / (2 * jnp.pi), ee)
    phase_prim_ee = (phase_ee + 0.5)  % 1 - 0.5
    prim_ee = jnp.einsum('il,jkl->jki', lattice, phase_prim_ee)
    return interaction_energy_scale * jnp.real(
        electron_electron_potential(prim_ee)
    )

  return potential

def make_Gaussian_potential(
    lattice: jnp.ndarray,
    atoms: jnp.ndarray,
    charges: jnp.ndarray,
    truncation_limit: int = 1,
    U: float = 1.0,
    U_width: float = 1.0,
) -> Callable[[jnp.ndarray, jnp.ndarray], float]:
  """Creates a function to evaluate a short range e-e-interaction potential with two terms:
    U: A Hubbard onsite repulsion of strengh: integal dr^2 U(r) = U
    V: A ring potential with range rV of strength: integral dr^2 V(r) = V

  Args:
    lattice: Shape (2, 2). Matrix whose columns are the primitive lattice
      vectors.
    atoms: Shape (natoms, ndim). Positions of the atoms.
    charges: Shape (natoms). Nuclear charges of the atoms.
    truncation_limit: Integer. Half side length of cube of nearest neighbours
      to primitive cell which are summed over in evaluation of Ewald sum.
      Must be large enough to achieve convergence for the real and reciprocal
      space sums.
    include_heg_background: bool. When True, includes cell-neutralizing
      background term for homogeneous electron gas.

  Returns:
    Callable with signature f(ee), where (ee) are 
    electron-electron displacement vectors, which evaluates the
    Coulomb sum for the periodic lattice via the Ewald method.
  """
  del atoms, charges # unused
  # ndim = atoms.shape[1]
  print("make Gaussian interaction potential with pars = " + 
    str([U, U_width]))
  rec = 2 * jnp.pi * jnp.linalg.inv(lattice)
  ordinals = sorted(range(-truncation_limit, truncation_limit + 1), key=abs)
  ordinals = jnp.array(list(itertools.product(ordinals, repeat=2))) # Adjust for 2D
  lat_vectors = jnp.einsum('kj,ij->ik', lattice, ordinals)

  def real_space_potential(separation: jnp.ndarray):
    """Real-space Ewald potential between charges seperated by separation."""
    displacements = jnp.linalg.norm(
        separation - lat_vectors, axis=-1)  # |r - R|
    return U * jnp.sum(
      jnp.exp(- displacements**2 / 2 / U_width**2) / (2*jnp.pi*U_width**2))

  batch_potential_sum = jax.vmap(real_space_potential, in_axes=(0,))

  def electron_electron_potential(ee: jnp.ndarray):
    """Evaluates periodic electron-electron potential."""
    nelec = ee.shape[0]
    ee = jnp.reshape(ee, [-1, 2]) # Adjust for 2d
    ee_potential = batch_potential_sum(ee)
    ee_potential = jnp.reshape(ee_potential, [nelec, nelec])
    ee_potential = ee_potential.at[jnp.diag_indices(nelec)].set(0.0)
    return 0.5 * jnp.sum(ee_potential)

  def potential(ae: jnp.ndarray, ee: jnp.ndarray):
    """Accumulates atom-electron, atom-atom, and electron-electron potential."""
    del ae
    # Map relative distances back to first unit cell
    phase_ee = jnp.einsum('il,jkl->jki', rec / (2 * jnp.pi), ee)
    phase_prim_ee = (phase_ee + 0.5)  % 1 - 0.5
    prim_ee = jnp.einsum('il,jkl->jki', lattice, phase_prim_ee)
    return jnp.real(electron_electron_potential(prim_ee))

  return potential

def local_energy(
    f: networks.FermiNetLike,
    charges: jnp.ndarray,
    nspins: Sequence[int],
    use_scan: bool = False,
    complex_output: bool = False,
    states: int = 0,
    lattice: Optional[jnp.ndarray] = None,
    heg: bool = True,
    convergence_radius: int = 5,
    potential_type = 'Coulomb',
    potential_kwargs = {}
) -> hamiltonian.LocalEnergy:
  """Creates the local energy function in periodic boundary conditions.

  Args:
    f: Callable which returns the sign and log of the magnitude of the
      wavefunction given the network parameters and configurations data.
    charges: Shape (natoms). Nuclear charges of the atoms.
    nspins: Number of particles of each spin.
    use_scan: Whether to use a `lax.scan` for computing the laplacian.
    complex_output: If true, the output of f is complex-valued.
    states: Number of excited states to compute. Not implemented, only present
      for consistency of calling convention.
    lattice: Shape (ndim, ndim). Matrix of lattice vectors. Default: identity
      matrix.
    heg: bool. Flag to enable features specific to the electron gas.
    convergence_radius: int. Radius of cluster summed over by Ewald sums.

  Returns:
    Callable with signature e_l(params, key, data) which evaluates the local
    energy of the wavefunction given the parameters params, RNG state key,
    and a single MCMC configuration in data.
  """
  print("Using customized local_energy from pbc.hamiltonian ")
  if states:
    raise NotImplementedError('Excited states not implemented with PBC.')
  del nspins
  assert lattice is not None, "pbc.hamiltonian.local_energy requires lattice to be passed"

  ke = hamiltonian.local_kinetic_energy(f, use_scan=use_scan,
                                        complex_output=complex_output,
                                        laplacian_method=potential_kwargs['laplacian_method'])

  if potential_type == "Coulomb":
    # Coulomb e-e interaction potential. 
    # Optionally, specify interaction_energy_scale as an overall factor.
    # WARNING: Jastrow factors assume interaction_energy_scale = 1
    if 'interaction_energy_scale' in potential_kwargs:
      interaction_energy_scale = potential_kwargs['interaction_energy_scale']
    else:
      interaction_energy_scale = 1.0

    potential_energy = make_2DCoulomb_potential(
        lattice, jnp.array([0.0]), charges, convergence_radius, interaction_energy_scale
    )

  elif potential_type == "Gaussian":
     # Gaussian potential requires U, U_width to be specified in potential_kwargs
     potential_energy = make_2DCoulomb_potential(
        lattice, jnp.array([0.0]), charges, convergence_radius, potential_kwargs['U'], potential_kwargs['U_width']
    )

  def _e_l(
        params: networks.ParamTree, key: chex.PRNGKey, data: networks.FermiNetData
  ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    """Returns the total energy.

    Args:
        params: network parameters.
        key: RNG state.
        data: MCMC configuration.
    """
    del key  # unused
    ae, ee, _, _ = networks.construct_input_features(
        data.positions, data.atoms, ndim=2)
    potential = potential_energy(ae, ee)
    kinetic = ke(params, data)
    return potential + jnp.real(kinetic), None

  return _e_l
