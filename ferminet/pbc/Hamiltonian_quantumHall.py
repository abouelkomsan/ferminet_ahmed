# Copyright 2025 Ahmed Abouelkomsan, Max Geier (MIT)

"""Implementation of the 2DEG with periodic magnetic and periodic potential
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

import logging

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
  gamma_factor = 2.8 #from Casino manual for 2D systems
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
   #   print(0.5 * nelec * madelung_const)
      #potential = 0.5 * jnp.sum(ewald)   - 0.5 * nelec**2 * phi_S_q0
      potential = 0.5*jnp.sum(ewald) - 0.5*nelec*(nelec-1)*phi_S_q0 #modified the second term
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

def make_MR_effective_potential(
    lattice: jnp.ndarray,
    atoms: jnp.ndarray,
    charges: jnp.ndarray,
    truncation_limit_coulomb: int = 5,
    truncation_limit_short: int = 1,
    interaction_energy_scale: float = 1.0,
    a1: float = 117.429,
    a2: float = -755.468,
    alpha1: float = 1.3177,
    alpha2: float = 2.9026,
):
  """Creates the effective real-space interaction V_eff(r) that reproduces
  n=1 Coulomb pseudopotentials when projected to the LLL:

      V_eff(r) = (e^2/eps) [ 1/r + a1 exp(-alpha1 r^2)
                             + a2 r^2 exp(-alpha2 r^2) ].

  Here the 1/r part is handled by the existing 2D Ewald Coulomb, and the
  exponential corrections are added as a short-range periodic potential.

  Args:
    lattice: (2, 2) primitive lattice vectors as columns.
    atoms, charges: unused (kept for API compatibility).
    truncation_limit_coulomb: Ewald cutoff for 1/r Coulomb part.
    truncation_limit_short: number of image shells for short-range terms.
    interaction_energy_scale: overall prefactor (e^2/eps in your units).
    a1, a2, alpha1, alpha2: parameters from the paper, assumed dimensionless
        in units of the magnetic length.

  Returns:
    Callable potential(ae, ee) giving the total e-e interaction energy.
  """
  # Base Coulomb part (1/r + neutralizing background) via Ewald.
  #from your_module import make_2DCoulomb_potential  # adjust import as needed

  print(
      "making MR effective potential with energy scale:",
      interaction_energy_scale,
      "and params:",
      [a1, a2, alpha1, alpha2],
  )

  coulomb_potential = make_2DCoulomb_potential(
      lattice=lattice,
      atoms=atoms,
      charges=charges,
      truncation_limit=truncation_limit_coulomb,
      interaction_energy_scale=interaction_energy_scale,
  )

  # --- Short-range correction part: a1 e^{-α1 r^2} + a2 r^2 e^{-α2 r^2} ---

  # reciprocal lattice for mapping ee to first cell
  rec = 2.0 * jnp.pi * jnp.linalg.inv(lattice)

  # lattice translations for short-range real-space sum
  ordinals = sorted(range(-truncation_limit_short,
                          truncation_limit_short + 1), key=abs)
  ordinals = jnp.array(list(itertools.product(ordinals, repeat=2)))
  lat_vectors = jnp.einsum("kj,ij->ik", lattice, ordinals)

  def short_real_space_potential(separation: jnp.ndarray) -> jnp.ndarray:
    """Short-range part between two electrons with displacement `separation`
    (in real space), including periodic images."""
    # displacements to all lattice images
    displacements = jnp.linalg.norm(separation - lat_vectors, axis=-1)  # |r - R|
    r2 = displacements**2

    # Avoid numerical issues at r = 0; the diagonal (i=j) is removed later anyway,
    # but we keep a tiny epsilon here to be safe.
    eps = 1e-12
    r2_safe = jnp.where(r2 > eps**2, r2, eps**2)

    term1 = a1 * jnp.exp(-alpha1 * r2_safe)
    term2 = a2 * r2_safe * jnp.exp(-alpha2 * r2_safe)
    return interaction_energy_scale * jnp.sum(term1 + term2)

  batch_short_sum = jax.vmap(short_real_space_potential, in_axes=(0,))

  def short_ee_potential(ee: jnp.ndarray) -> jnp.ndarray:
    """Short-range correction to periodic e-e potential."""
    nelec = ee.shape[0]
    ee = jnp.reshape(ee, [-1, 2])  # (nelec^2, 2)
    val = batch_short_sum(ee)
    val = jnp.reshape(val, [nelec, nelec])
    val = val.at[jnp.diag_indices(nelec)].set(0.0)
    return 0.5 * jnp.sum(val)

  def potential(ae: jnp.ndarray, ee: jnp.ndarray) -> jnp.ndarray:
    """Total V_eff = V_Coulomb + short-range corrections."""
    # 1/r part from Ewald Coulomb (already periodic + background corrected).
    v_coul = coulomb_potential(ae, ee)

    # Map relative distances back to first unit cell for short-range part
    phase_ee = jnp.einsum("il,jkl->jki", rec / (2.0 * jnp.pi), ee)
    phase_prim_ee = (phase_ee + 0.5) % 1.0 - 0.5
    prim_ee = jnp.einsum("il,jkl->jki", lattice, phase_prim_ee)

    v_short = jnp.real(short_ee_potential(prim_ee))

    return v_coul + v_short

  return potential

def make_softCoulomb_potential(
    lattice: jnp.ndarray,
    atoms: jnp.ndarray,
    charges: jnp.ndarray,
    truncation_limit_coulomb: int = 5,
    truncation_limit_short: int = 1,
    interaction_energy_scale: float = 1.0,
    lambda_soft: float = 1.0,
):
  """Creates periodic 'soft' Coulomb interaction

      V(r) = (e^2/eps) / sqrt(r^2 + lambda_soft^2)

  using 2D Ewald for the 1/r part plus a short-range real-space correction.

  Args:
    lattice: (2, 2) primitive lattice vectors as columns.
    atoms, charges: unused (kept for API compatibility).
    truncation_limit_coulomb: Ewald cutoff for pure 1/r Coulomb part.
    truncation_limit_short: number of real-space image shells for the
        short-range correction Σ_R [1/sqrt(|r-R|^2 + λ^2) - 1/|r-R|].
    interaction_energy_scale: overall prefactor, e^2/eps in your units.
    lambda_soft: softening length λ (in same units as lattice).

  Returns:
    Callable potential(ae, ee) giving the total e-e interaction energy.
  """
  del atoms, charges  # unused for HEG

  print(
      "making soft Coulomb (Ewald) potential with lambda =",
      lambda_soft,
      "energy scale =",
      interaction_energy_scale,
  )

  # Base 1/r part via existing Ewald implementation (includes background).
  coulomb_potential = make_2DCoulomb_potential(
      lattice=lattice,
      atoms=jnp.zeros((0, 2)),  # dummy
      charges=jnp.zeros((0,)),
      truncation_limit=truncation_limit_coulomb,
      interaction_energy_scale=interaction_energy_scale,
  )

  # Reciprocal lattice for mapping ee into the primitive cell
  rec = 2.0 * jnp.pi * jnp.linalg.inv(lattice)

  # Real-space lattice vectors for the short-range correction
  ordinals = sorted(range(-truncation_limit_short,
                          truncation_limit_short + 1), key=abs)
  ordinals = jnp.array(list(itertools.product(ordinals, repeat=2)))  # (N, 2)
  lat_vectors = jnp.einsum("kj,ij->ik", lattice, ordinals)  # (N, 2)

  def short_real_space_correction(separation: jnp.ndarray) -> jnp.ndarray:
    """Short-range correction ΔV(r) summed over images:
       ΔV(r) = Σ_R [1/sqrt(|r-R|^2 + λ^2) - 1/|r-R|].
    """
    displacements = jnp.linalg.norm(separation - lat_vectors, axis=-1)
    # Regularize extremely small distances (we never evaluate at r=0 self,
    # but r ~ R can still be tiny numerically)
    eps = 1e-12
    d_safe = jnp.where(displacements > eps, displacements, eps)

    v_soft = 1.0 / jnp.sqrt(d_safe**2 + lambda_soft**2)
    v_coul = 1.0 / d_safe
    # This is already short-ranged, so a small truncation_limit_short is OK.
    return interaction_energy_scale * jnp.sum(v_soft - v_coul)

  batch_short_corr = jax.vmap(short_real_space_correction, in_axes=(0,))

  def short_ee_correction(ee: jnp.ndarray) -> jnp.ndarray:
    """Total short-range correction energy for all electrons."""
    nelec = ee.shape[0]
    ee = jnp.reshape(ee, [-1, 2])  # (nelec^2, 2)
    corr = batch_short_corr(ee)
    corr = jnp.reshape(corr, [nelec, nelec])
    corr = corr.at[jnp.diag_indices(nelec)].set(0.0)
    return 0.5 * jnp.sum(corr)

  def potential(ae: jnp.ndarray, ee: jnp.ndarray) -> jnp.ndarray:
    """Total soft Coulomb interaction energy."""
    # 1/r part from Ewald
    v_coul = coulomb_potential(ae, ee)

    # Map separations to primitive cell for short-range correction
    phase_ee = jnp.einsum("il,jkl->jki", rec / (2.0 * jnp.pi), ee)
    phase_prim_ee = (phase_ee + 0.5) % 1.0 - 0.5
    prim_ee = jnp.einsum("il,jkl->jki", lattice, phase_prim_ee)

    v_corr = jnp.real(short_ee_correction(prim_ee))

    return v_coul + v_corr

  return potential

def make_cosine_potential(
    potential_lattice: jnp.ndarray,
    coefficients: jnp.ndarray,
    phases: jnp.ndarray,
) -> Callable[[jnp.ndarray], float]:
    """
    Creates a function to evaluate a periodic potential as a sum of three cosines with phases,
    adapted to take `ae` in the form outputted by `construct_input_features`.

    Args:
        potential_lattice: Shape (2, 2). Matrix whose columns are the primitive lattice vectors.
        coefficients: Shape (3,). Coefficients for the three cosine terms.
        phases: Shape (3,). Phases for the three cosine terms.

    Returns:
        Callable with signature f(ae), where ae is an array of atom-electron
        displacement vectors of (shape (nelectron, natom, ndim)), which evaluates the periodic potential.
    """
    # Compute reciprocal lattice vectors
    rec = 2 * jnp.pi * jnp.linalg.inv(potential_lattice)

    # Define the cosine potential function
    def potential(ae: jnp.ndarray) -> float:
        """
        Evaluates the periodic potential using atom-electron displacement vectors.

        Args:
            ae: Shape (nelec , natom, 2). Array of atom-electron displacement vectors.

        Returns:
            The value of the potential summed over all displacement vectors.
        """
        # Reshape `ae` to (nelec, natom, 2) if necessary
        ae = jnp.reshape(ae, (-1, 2))

        # Compute the cosine terms with phases for each displacement vector
        cos_term1 = coefficients[0] * jnp.cos(jnp.dot(ae, rec[0,:]) + phases[0])
        cos_term2 = coefficients[1] * jnp.cos(jnp.dot(ae, -rec[1,:]) + phases[1])
        cos_term3 = coefficients[2] * jnp.cos(jnp.dot(ae, -rec[0,:] + rec[1,:]) + phases[2])

        # Sum the cosine terms over all displacement vectors
        return jnp.sum(cos_term1 + cos_term2 + cos_term3)

    return potential

# def make_vectorpotential(
#     Bfield_lattice: jnp.ndarray,
#     flux: jnp.ndarray,
#     phase: jnp.ndarray,
#     threadedflux: jnp.ndarray = jnp.array([0, 0])
# ) -> Callable[[jnp.ndarray], jnp.ndarray]:
#     """
#     Creates a function to evaluate the vector potential

#     Args:
#         Bfield_lattice: Shape (2, 2). Matrix whose columns are the primitive lattice vectors.
#         coefficients: Shape (3,). Coefficients for the three cosine terms.
#         phases: Shape (3,). Phases for the three cosine terms.
#         threadedflux: Shape (2,). Optional threading flux vector, defaults to zero.

#     Returns:
#         Callable with signature f(ae), where ae is a flattened array of atom-electron
#         displacement vectors (shape (nelec * natom, 2)), which evaluates the vector potential and returns a vector.
#     """
#     # Compute reciprocal lattice vectors
#     rec = 2 * jnp.pi * jnp.linalg.inv(Bfield_lattice)
#     Glist = jnp.array([rec[0,:], rec[1,:], rec[0,:] - rec[1,:], -rec[0,:], -rec[1,:], -rec[0,:] + rec[1,:]])
    
#     # Precompute norms and coefficients for speed
#     Gcoeff_x = 1.0j * Glist[:, 1] * flux
#     Gcoeff_y = -1.0j * Glist[:, 0] * flux

#     # Define the vector potential component function
#     def vector_potential_comp(r: jnp.ndarray) -> jnp.ndarray:
#         # Compute dot products for all G vectors at once
#         dot_products = jnp.dot(Glist, r)
#         exp_terms = jnp.exp(1.0j * dot_products + phase) 

#         # Compute outx and outy using vectorized operations
#         outx = jnp.sum(exp_terms * Gcoeff_x)
#         outy = jnp.sum(exp_terms * Gcoeff_y)
#         return jnp.array([jnp.real(outx), jnp.real(outy)])

#     # Vectorize the component function using jax.vmap
#     vectorized_potential_comp = jax.vmap(vector_potential_comp)
#     def threadedflux_comp(r: jnp.ndarray) -> jnp.ndarray:
#         """
#         Computes the threaded flux contribution to the vector potential.

#         Args:
#             r: Shape (nelec, 2). 

#         Returns:
#             The threaded flux contribution as a vector of shape (nelec, 2).
#         """
#         return jnp.array([threadedflux[0] * r[:, 1], threadedflux[1] * r[:, 0]]).T
#     # Define the vector potential function
#     def potential(ae: jnp.ndarray) -> jnp.ndarray:
#         """
#         Evaluates the vector potential using atom-electron displacement vectors.

#         Args:
#             ae: Shape (nelec , natom, 2). Array of atom-electron displacement vectors.

#         Returns:
#             The vector potential as a vector of shape (nelec, 2)
#         """
#         # Reshape `ae` to (nelec * natom, 2) if necessary
#         ae = jnp.reshape(ae, (-1, 2))
#         return vectorized_potential_comp(ae),jnp.reshape(vectorized_potential_comp(ae), (-1,))

#     return potential

def make_vectorpotential(
    Bfield_lattice: jnp.ndarray,
    flux: jnp.ndarray,
    phase: jnp.ndarray,
    threadedflux: jnp.ndarray = jnp.array([0, 0])
) -> Callable[[jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Creates a function to evaluate the vector potential.

    Args:
        Bfield_lattice: Shape (2, 2). Matrix whose columns are the primitive lattice vectors.
        flux: Shape (3,). Coefficients for the three cosine terms.
        phase: Shape (3,). Phases for the three cosine terms.
        threadedflux: Shape (2,). Constant threaded flux vector, defaults to zero.

    Returns:
        Callable with signature f(ae), where ae is a flattened array of atom-electron
        displacement vectors (shape (nelec * natom, 2)), which evaluates the vector potential and returns:
            - The vector potential with shape (nelec * natom, 2).
            - A flattened version of the vector potential with shape (nelec * natom * 2,).
    """
    # Compute reciprocal lattice vectors
    rec = 2 * jnp.pi * jnp.linalg.inv(Bfield_lattice)
    Glist = jnp.array([rec[0, :], rec[1, :], rec[0, :] - rec[1, :], -rec[0, :], -rec[1, :], -rec[0, :] + rec[1, :]])
    
    # Precompute norms and coefficients for speed
    Gcoeff_x = 1.0j * Glist[:, 1] * flux
    Gcoeff_y = -1.0j * Glist[:, 0] * flux

    # Define the vector potential component function
    def vector_potential_comp(r: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the vector potential components for a given position vector.

        Args:
            r: Shape (2,). A single position vector.

        Returns:
            The vector potential components as a 2D vector.
        """
        # Compute dot products for all G vectors at once
        dot_products = jnp.dot(Glist, r)
        exp_terms = jnp.exp(1.0j * dot_products + phase)

        # Compute outx and outy using vectorized operations
        outx = jnp.sum(exp_terms * Gcoeff_x)
        outy = jnp.sum(exp_terms * Gcoeff_y)
        return jnp.array([jnp.real(outx), jnp.real(outy)])

    # Vectorize the component function using jax.vmap
    vectorized_potential_comp = jax.vmap(vector_potential_comp)

    # Define the vector potential function
    def potential(ae: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Evaluates the vector potential using atom-electron displacement vectors.

        Args:
            ae: Shape (nelec, natom, 2). Array of atom-electron displacement vectors.

        Returns:
            - The vector potential as a vector of shape (nelec * natom, 2).
            - A flattened version of the vector potential with shape (nelec * natom * 2,).
        """
        # Reshape `ae` to (nelec * natom, 2) if necessary
        ae = jnp.reshape(ae, (-1, 2))
        # Compute the vector potential and add the constant threaded flux
        vector_potential = vectorized_potential_comp(ae) + threadedflux
        # Return both the vector potential and its flattened version
        return vector_potential, jnp.reshape(vector_potential, (-1,))
  
    return potential

def symmetric_gauge_vector_potential() -> Callable[[jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Creates a function to evaluate the vector potential in the symmetric gauge
    for the quantum Hall problem.

    The vector potential is defined as:
        A(r) = (1/2) * (-y * xhat + x * yhat)
    where r = (x, y) is the position vector.

    Returns:
        Callable with signature f(ae), where ae is a flattened array of atom-electron
        displacement vectors (shape (nelec * natom, 2)), which evaluates the vector potential and returns:
            - The vector potential with shape (nelec * natom, 2).
            - A flattened version of the vector potential with shape (nelec * natom * 2,).
    """
    def vector_potential_comp(r: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the vector potential components for a given position vector in the symmetric gauge.

        Args:
            r: Shape (2,). A single position vector.

        Returns:
            The vector potential components as a 2D vector.
        """
        x, y = r
        Ax = -0.5 * y  # x-component of the vector potential
        Ay = 0.5 * x   # y-component of the vector potential
        return jnp.array([Ax, Ay])

    # Vectorize the component function using jax.vmap
    vectorized_potential_comp = jax.vmap(vector_potential_comp)

    # Define the vector potential function
    def potential(ae: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Evaluates the vector potential using atom-electron displacement vectors.

        Args:
            ae: Shape (nelec, natom, 2). Array of atom-electron displacement vectors.

        Returns:
            - The vector potential as a vector of shape (nelec * natom, 2).
            - A flattened version of the vector potential with shape (nelec * natom * 2,).
        """
        # Reshape `ae` to (nelec * natom, 2) if necessary
        ae = jnp.reshape(ae, (-1, 2))
        # Compute the vector potential
        vector_potential = vectorized_potential_comp(ae)
        # Return both the vector potential and its flattened version
        return vector_potential, jnp.reshape(vector_potential, (-1,))
  
    return potential

gradient_Avec_energy = Callable[
    [networks.ParamTree, networks.FermiNetData], jnp.ndarray
]
def local_gradient_vectorpotential_energy(
    f: networks.FermiNetLike,
    Avec: Callable[[jnp.ndarray], jnp.ndarray]
) -> gradient_Avec_energy:
    r"""Creates a function for the local dot product of vector potential and gradient of wavefunction, A(r).grad(\psi)

    Args:
        f: Callable which evaluates the wavefunction as a
            (sign or phase, log magnitude) tuple.
        Avec: Callable which evaluates the vector potential A(r) at positions r and outputs a vector
            of shape (ndim*N, 2) with ndim = 2.

    Returns:
        Callable which evaluates A(r).grad(\psi)
    """
    phase_f = utils.select_output(f, 0)
    logabs_f = utils.select_output(f, 1)

    def Avec_dot_grad_over_f(params, data):
        grad_logabs_f = jax.grad(logabs_f, argnums=1)
        grad_phase_f = jax.grad(phase_f, argnums=1)
        _, Avec_val = Avec(data.positions)
        return jnp.dot(
            Avec_val,
            grad_logabs_f(params, data.positions, data.spins, data.atoms, data.charges)
            + 1.j * grad_phase_f(params, data.positions, data.spins, data.atoms, data.charges)
        )

    return Avec_dot_grad_over_f

def local_gradient(
    f: networks.FermiNetLike,
) -> gradient_Avec_energy:
    r"""Creates a function for the local dot product of vector potential and gradient of wavefunction, A(r).grad(\psi)

    Args:
        f: Callable which evaluates the wavefunction as a
            (sign or phase, log magnitude) tuple.
        Avec: Callable which evaluates the vector potential A(r) at positions r and outputs a vector
            of shape (ndim*N, 2) with ndim = 2.

    Returns:
        Callable which evaluates grad(\psi)
    """
    phase_f = utils.select_output(f, 0)
    logabs_f = utils.select_output(f, 1)

    def grad_over_f(params, data):
        grad_logabs_f = jax.grad(logabs_f, argnums=1)
        grad_phase_f = jax.grad(phase_f, argnums=1)
        #return (
        #grad_logabs_f(params, data.positions, data.spins, data.atoms, data.charges)
        #+ 1.j * grad_phase_f(params, data.positions, data.spins, data.atoms, data.charges)
        
        return grad_logabs_f(params, data.positions, data.spins, data.atoms, data.charges) +  1.j * grad_phase_f(params, data.positions, data.spins, data.atoms, data.charges)
        
    return grad_over_f  

def local_energy(
    f: networks.FermiNetLike,
    charges: jnp.ndarray,
    nspins: Sequence[int],
    use_scan: bool = False,
    complex_output: bool = False,
    states: int = 0,
    lattice: Optional[jnp.ndarray] = None,
    kinetic_energy_kwargs = {},
    Bfield_lattice: Optional[jnp.ndarray] = None,
    Bfield_kwargs = {},
    periodic_lattice: Optional[jnp.ndarray] = None,
    periodic_potential_kwargs = {},
    heg: bool = True,
    convergence_radius: int = 20,
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
    kinetic_energy_kwargs: kwargs for the kinetic energy function.
    periodic_lattice: Shape (ndim, ndim). Matrix of lattice vectors of the unit cell of the periodic potential.
    heg: bool. Flag to enable features specific to the electron gas.
    convergence_radius: int. Radius of cluster summed over by Ewald sums.
    potential_type: specifies the type of ee potential to use.
    potential_kwargs: kwargs for the ee potential

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
    potential_energy = make_Gaussian_potential(
        lattice, jnp.array([0.0]), charges, convergence_radius, potential_kwargs['U'], potential_kwargs['U_width']
    )
  elif potential_type == "LLL_MR_effective":
    if 'interaction_energy_scale' in potential_kwargs:
      interaction_energy_scale = potential_kwargs['interaction_energy_scale']
    else:
      interaction_energy_scale = 1.0
    potential_energy = make_MR_effective_potential(
        lattice,
        jnp.array([0.0]),
        charges,
        convergence_radius,
        convergence_radius,
        interaction_energy_scale)
  elif potential_type == "softCoulomb":
    if 'interaction_energy_scale' in potential_kwargs:
      interaction_energy_scale = potential_kwargs['interaction_energy_scale']
    else:
      interaction_energy_scale = 1.0
    potential_energy = make_softCoulomb_potential(
        lattice,
        jnp.array([0.0]),
        charges,
        convergence_radius,
        convergence_radius,
        interaction_energy_scale,
        potential_kwargs['lambda_soft'])
 # periodic_potential_energy = make_cosine_potential(periodic_lattice,periodic_potential_kwargs['coefficients'], periodic_potential_kwargs['phases'])
  vector_potential = symmetric_gauge_vector_potential()
  grad_func = local_gradient(f)
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
   # periodic_potential = periodic_potential_energy(ae)
    kinetic = ke(params, data)
    gradf = grad_func(params,data)
    _,Avec_val = vector_potential(ae)
    #return potential + kinetic_energy_kwargs['prefactor']*jnp.real(kinetic) + periodic_potential, None
    #+ + jnp.real(-1.0j* jnp.dot(Avec_val,gradf))  + 0.5*jnp.dot(Avec_val,Avec_val)
    #out = kinetic  -1.0j* jnp.dot(Avec_val,gradf)  + 0.5*jnp.dot(Avec_val,Avec_val)
    
    return  potential + kinetic  + 1.0j* jnp.dot(Avec_val,gradf)  + 0.5*jnp.dot(Avec_val,Avec_val)  , None
  return _e_l

def local_energy_enforce_real(
    f: networks.FermiNetLike,
    charges: jnp.ndarray,
    nspins: Sequence[int],
    use_scan: bool = False,
    complex_output: bool = False,
    states: int = 0,
    lattice: Optional[jnp.ndarray] = None,
    kinetic_energy_kwargs = {},
    Bfield_lattice: Optional[jnp.ndarray] = None,
    Bfield_kwargs = {},
    periodic_lattice: Optional[jnp.ndarray] = None,
    periodic_potential_kwargs = {},
    heg: bool = True,
    convergence_radius: int = 10,
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
    kinetic_energy_kwargs: kwargs for the kinetic energy function.
    periodic_lattice: Shape (ndim, ndim). Matrix of lattice vectors of the unit cell of the periodic potential.
    heg: bool. Flag to enable features specific to the electron gas.
    convergence_radius: int. Radius of cluster summed over by Ewald sums.
    potential_type: specifies the type of ee potential to use.
    potential_kwargs: kwargs for the ee potential

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
     potential_energy = make_Gaussian_potential(
        lattice, jnp.array([0.0]), charges, convergence_radius, potential_kwargs['U'], potential_kwargs['U_width']
    )

 # periodic_potential_energy = make_cosine_potential(periodic_lattice,periodic_potential_kwargs['coefficients'], periodic_potential_kwargs['phases'])
  vector_potential = symmetric_gauge_vector_potential()
  grad_func = local_gradient(f)
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
   # periodic_potential = periodic_potential_energy(ae)
    kinetic = ke(params, data)
    gradf = grad_func(params,data)
    _,Avec_val = vector_potential(ae)
    #return potential + kinetic_energy_kwargs['prefactor']*jnp.real(kinetic) + periodic_potential, None
    #jnp.real(kinetic  + jnp.real(-1.0j* jnp.dot(Avec_val,gradf))  + 0.5*jnp.dot(Avec_val,Avec_val))
    return jnp.real(potential + kinetic  -1.0j* jnp.dot(Avec_val,gradf)  + 0.5*jnp.dot(Avec_val,Avec_val))  , None
  return _e_l
""" Note to my self

- The positions of the NN data.positions are flatten (Shape (nelectrons*ndim,))
- The positions "ae" are Shape (nelectron, natom, ndim).

"""
