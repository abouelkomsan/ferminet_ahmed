# Copyright Ahmed Abouelkomsan (MIT) 2025
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
from jax.scipy.special import logsumexp


def g1g2(abs_lattice, G1, G2):
    """
    Compute g1, g2, and related values based on the lattice and G1, G2.
    """
    a, b = abs_lattice[0, 0], abs_lattice[0, 1]
    c, d = abs_lattice[1, 0], abs_lattice[1, 1]
    size = jnp.abs(jnp.linalg.det(abs_lattice))
    sgn = jnp.sign(jnp.linalg.det(abs_lattice))
    g1 = (1 / size) * (d * sgn * G1 - c * sgn * G2)
    g2 = (1 / size) * (-b * sgn * G1 + a * sgn * G2)
    return g1, g2, sgn * G1, sgn * G2, sgn

def kpoints(abs_lattice):
    """
    Generate k-space momentum points based on the lattice.
    """
    a, b = abs_lattice[0, 0], abs_lattice[0, 1]
    c, d = abs_lattice[1, 0], abs_lattice[1, 1]
    limit = int(jnp.abs(jnp.linalg.det(abs_lattice)))  # Determinant of the lattice
    xlist, ylist = [], []
    for x in range(-limit, limit + 1):
        for y in range(-limit, limit + 1):
            if (d * x - b * y >= -limit // 2 and 
                d * x - b * y < limit // 2 and 
                -c * x + a * y >= -limit // 2 and 
                -c * x + a * y < limit // 2):
                xlist.append(x)
                ylist.append(y)
    klabel = jnp.array([[xlist[i], ylist[i]] for i in range(len(xlist))])
    return klabel

def mn(k, g1, g2):
    """
    Compute the discretized momenta.
    """
    return k[0] * g1 + k[1] * g2

def mn_with_flux(k, g1, g2, phi1, phi2):
    """
    Compute the discretized momenta with inserted flux.
    """
    return (k[0] + phi1) * g1 + (k[1] + phi2) * g2

def projector(k, abs_lattice):
    """
    Compute the projector P(k) = k0 if k = k0 + m*G1 + n*G2.
    """
    m, n = k[0], k[1]
    lattice_size = jnp.abs(jnp.linalg.det(abs_lattice))
    sgn = jnp.sign(jnp.linalg.det(abs_lattice))
    a, b = abs_lattice[0, 0], abs_lattice[0, 1]
    c, d = abs_lattice[1, 0], abs_lattice[1, 1]
    i, j = 0, 0

    G1_limit = m * d - n * b
    G2_limit = n * a - m * c

    while G1_limit > jnp.ceil(lattice_size / 2) - 1:
        G1_limit -= lattice_size
        m -= sgn * a
        n -= sgn * c
        i += 1

    while G2_limit > jnp.ceil(lattice_size / 2) - 1:
        G2_limit -= lattice_size
        m -= sgn * b
        n -= sgn * d
        j += 1

    while G1_limit < jnp.ceil(-lattice_size / 2):
        G1_limit += lattice_size
        m += sgn * a
        n += sgn * c
        i -= 1

    while G2_limit < jnp.ceil(-lattice_size / 2):
        G2_limit += lattice_size
        m += sgn * b
        n += sgn * d
        j -= 1

    return jnp.array([m, n]), jnp.int8(sgn * i), jnp.int8(sgn * j)

def lattice_vecs(a1: jnp.ndarray, a2: jnp.ndarray, Tmatrix: jnp.ndarray) -> jnp.ndarray:
    """
    Return the basis T1, T2 of the super-cell built from the unit cell lattice vectors a1 and a2.
    """
    T1 = Tmatrix[0, 0] * a1 + Tmatrix[0, 1] * a2
    T2 = Tmatrix[1, 0] * a1 + Tmatrix[1, 1] * a2
    return jnp.column_stack([T1, T2])

def reciprocal_vecs(a1: jnp.ndarray, a2: jnp.ndarray, Tmatrix: jnp.ndarray) -> jnp.ndarray:
    """
    Return the reciprocal basis vectors g1, g2 such that T_i · g_j = 2π δ_ij,
    where T1, T2 are the supercell vectors built from the unit cell vectors a1 and a2.
    """
    # Construct supercell lattice vectors T1, T2
    T1 = Tmatrix[0, 0] * a1 + Tmatrix[0, 1] * a2
    T2 = Tmatrix[1, 0] * a1 + Tmatrix[1, 1] * a2

    # Compute the oriented area A = T1 x T2 (scalar in 2D)
    A = T1[0] * T2[1] - T1[1] * T2[0]

    # Compute reciprocal vectors satisfying T_i · g_j = 2π δ_ij
    g1 = 2 * jnp.pi / A * jnp.array([T2[1], -T2[0]])
    g2 = 2 * jnp.pi / A * jnp.array([-T1[1], T1[0]])

    return jnp.column_stack([g1, g2])

def find_all_translations_in_supercell(lattice, unit_cell_vectors, searchlimit=10, tolerance=1e-3):
    T1, T2 = lattice[:, 0], lattice[:, 1]
    lattice_inverse = jnp.linalg.inv(lattice)

    # Generate all combinations of x and y in the range [-searchlimit, searchlimit]
    x = jnp.arange(-searchlimit, searchlimit + 1)
    y = jnp.arange(-searchlimit, searchlimit + 1)
    grid_x, grid_y = jnp.meshgrid(x, y, indexing="ij")
    grid_x = grid_x.ravel()
    grid_y = grid_y.ravel()

    # Compute translations for all combinations
    def compute_translation(x, y):
        translation = x * unit_cell_vectors[0] + y * unit_cell_vectors[1]
        fractional_coords = jnp.dot(translation, lattice_inverse.T)
        fractional_coords_rounded = jnp.round(fractional_coords, 5)
        return translation, fractional_coords_rounded

    translations, fractional_coords = jax.vmap(compute_translation)(grid_x, grid_y)

    # Filter translations where fractional coordinates are within [0, 1)
    valid_mask = jnp.logical_and(
        jnp.logical_and(0 <= fractional_coords[:, 0], fractional_coords[:, 0] < 1),
        jnp.logical_and(0 <= fractional_coords[:, 1], fractional_coords[:, 1] < 1),
    )
    valid_translations = translations[valid_mask]

    return valid_translations


def construct_momeigstate(f, targetmom, mom_kwargs):
    """
    Construct the momentum eigenstate wavefunction based on the FermiNet-like model and lattice parameters.

    Args:
        f: A FermiNet-like model that returns phase and magnitude.
        abs_lattice: Lattice parameters (e.g., lattice size or shape).
        unit_cell_vectors: Basis vectors of the unit cell.

    Returns:
        A function `mom_eig_state` that computes the momentum eigenstate wavefunction.
    """
    abs_lattice = mom_kwargs['abs_lattice']
    unit_cell_vectors = mom_kwargs['unit_cell_vectors']
    logsumtrick = mom_kwargs.get('logsumtrick', True)
    # Compute lattice and reciprocal lattice vectors
    lattice = lattice_vecs(unit_cell_vectors[0], unit_cell_vectors[1], abs_lattice)
    reciprocal = reciprocal_vecs(unit_cell_vectors[0], unit_cell_vectors[1], np.array([[1,0], [0, 1]]))
    g1, g2, _, _, _ = g1g2(abs_lattice, reciprocal[:,0], reciprocal[:,1])
    klabels = kpoints(abs_lattice)

    # Find all translations in the supercell
    translations = find_all_translations_in_supercell(lattice, unit_cell_vectors)
    def log_network_original(*args, **kwargs):
        """
        Compute the log of the network output.
        """
        phase, mag = f(*args, **kwargs)
        return mag + 1.j * phase

    # Sum over all translations to enforce periodicity using log-sum-exp trick
    def new_log_network(params, pos, spins, atoms, charges):
        ndim = 2
        N = pos.shape[0] // ndim  # Number of particles (assuming ndim is globally defined)

        # Reshape pos to (N, ndim) for particle-wise operations
        reshaped_pos = pos.reshape(N, ndim)

        # Ensure mn(klabels[targetmom], g1, g2) is a JAX array
        mn_result = jnp.array(mn(klabels[targetmom], g1, g2))

        # Compute the phase shift for all translations at once
        phase_shifts = 1j * jnp.dot(translations, mn_result)  # Shape: (num_translations,)

        # Tile translations to match the positions
        tiled_translations = jnp.repeat(translations[:, None, :], N, axis=1)  # Shape: (num_translations, N, ndim)
        translated_positions = reshaped_pos[None, :, :] + tiled_translations  # Shape: (num_translations, N, ndim)

        # Flatten translated_positions back to (num_translations, N * ndim)
        flattened_translated_positions = translated_positions.reshape(translated_positions.shape[0], -1)

        # Compute the log of the network output for all translations
        log_network_outputs = jax.vmap(
            lambda t_pos: log_network_original(params, t_pos, spins, atoms, charges)
        )(flattened_translated_positions)  # Shape: (num_translations,)

        if logsumtrick:
            # Use log-sum-exp trick for numerical stability
            translated_psi_sum = logsumexp(phase_shifts + log_network_outputs)
            return translated_psi_sum + jnp.log(1 / translations.shape[0])  # Normalize
        else:
            # Direct summation
            translated_psi_sum = jnp.sum(jnp.exp(phase_shifts + log_network_outputs))
            return jnp.log(translated_psi_sum) - jnp.log(translations.shape[0])  # Normalize

    return new_log_network




