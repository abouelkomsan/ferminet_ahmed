# Copyright 2022 DeepMind Technologies Limited.
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
# limitations under the License

"""Multiplicative envelopes appropriate for periodic boundary conditions.

See Cassella, G., Sutterud, H., Azadi, S., Drummond, N.D., Pfau, D.,
Spencer, J.S. and Foulkes, W.M.C., 2022. Discovering Quantum Phase Transitions
with Fermionic Neural Networks. arXiv preprint arXiv:2202.05183.
"""

import itertools
from typing import Mapping, Optional, Sequence, Tuple, Union

from ferminet import envelopes
import jax
import jax.numpy as jnp
import numpy as np
from ferminet import ellipticfunctions
import kfac_jax



def make_multiwave_envelope(kpoints: jnp.ndarray) -> envelopes.Envelope:
  """Returns an oscillatory envelope.

  Envelope consists of a sum of truncated 3D Fourier series, one centered on
  each atom, with Fourier frequencies given by kpoints:

    sigma_{2i}*cos(kpoints_i.r_{ae}) + sigma_{2i+1}*sin(kpoints_i.r_{ae})

  Initialization sets the coefficient of the first term in each
  series to 1, and all other coefficients to 0. This corresponds to the
  cosine of the first entry in kpoints. If this is [0, 0, 0], the envelope
  will evaluate to unity at the beginning of training.

  Args:
    kpoints: Reciprocal lattice vectors of terms included in the Fourier
      series. Shape (nkpoints, ndim) (Note that ndim=3 is currently
      a hard-coded default).

  Returns:
    An instance of ferminet.envelopes.Envelope with apply_type
    envelopes.EnvelopeType.PRE_DETERMINANT
  """

  def init(
      natom: int, output_dims: Sequence[int], ndim: int = 3
  ) -> Sequence[Mapping[str, jnp.ndarray]]:
    """See ferminet.envelopes.EnvelopeInit."""
    del natom, ndim  # unused
    params = []
    nk = kpoints.shape[0]
    for output_dim in output_dims:
      params.append({'sigma': jnp.zeros((2 * nk, output_dim))})
      params[-1]['sigma'] = params[-1]['sigma'].at[0, :].set(1.0)
    return params

  def apply(*, ae: jnp.ndarray, r_ae: jnp.ndarray, r_ee: jnp.ndarray,
            sigma: jnp.ndarray) -> jnp.ndarray:
    """See ferminet.envelopes.EnvelopeApply."""
    del r_ae, r_ee  # unused
    phase_coords = ae @ kpoints.T
    waves = jnp.concatenate((jnp.cos(phase_coords), jnp.sin(phase_coords)),
                            axis=2)
    env = waves @ (sigma**2.0)
    return jnp.sum(env, axis=1)

  return envelopes.Envelope(envelopes.EnvelopeType.PRE_DETERMINANT, init, apply)


def make_kpoints(
    lattice: Union[np.ndarray, jnp.ndarray],
    spins: Tuple[int, int],
    min_kpoints: Optional[int] = None,
) -> jnp.ndarray:
  """Generates an array of reciprocal lattice vectors.

  Args:
    lattice: Matrix whose columns are the primitive lattice vectors of the
      system, shape (ndim, ndim). (Note that ndim=3 is currently
      a hard-coded default).
    spins: Tuple of the number of spin-up and spin-down electrons.
    min_kpoints: If specified, the number of kpoints which must be included in
      the output. The number of kpoints returned will be the
      first filled shell which is larger than this value. Defaults to None,
      which results in min_kpoints == sum(spins).

  Raises:
    ValueError: Fewer kpoints requested by min_kpoints than number of
      electrons in the system.

  Returns:
    jnp.ndarray, shape (nkpoints, ndim), an array of reciprocal lattice
      vectors sorted in ascending order according to length.
  """
  rec_lattice = 2 * jnp.pi * jnp.linalg.inv(lattice)
  # Calculate required no. of k points
  if min_kpoints is None:
    min_kpoints = sum(spins)
  elif min_kpoints < sum(spins):
    raise ValueError(
        'Number of kpoints must be equal or greater than number of electrons')

  dk = 1 + 1e-5
  # Generate ordinals of the lowest min_kpoints kpoints
  max_k = int(jnp.ceil(min_kpoints * dk)**(1 / 3.))
  ordinals = sorted(range(-max_k, max_k+1), key=abs)
  ordinals = jnp.asarray(list(itertools.product(ordinals, repeat=3)))

  kpoints = ordinals @ rec_lattice.T
  kpoints = jnp.asarray(sorted(kpoints, key=jnp.linalg.norm))
  k_norms = jnp.linalg.norm(kpoints, axis=1)

  return kpoints[k_norms <= k_norms[min_kpoints - 1] * dk]


def make_kpoints_2d(
    lattice: Union[np.ndarray, jnp.ndarray],
    spins: Tuple[int, int],
    min_kpoints: Optional[int] = None,
) -> jnp.ndarray:
  """Generates an array of 2D reciprocal lattice vectors.

  Args:
    lattice: Matrix whose columns are the primitive lattice vectors of the
      system, shape (2, 2).
    spins: Tuple of the number of spin-up and spin-down electrons.
    min_kpoints: If specified, the number of kpoints which must be included in
      the output. The number of kpoints returned will be the first filled shell
      which is larger than this value. Defaults to sum(spins).

  Returns:
    jnp.ndarray, shape (nkpoints, 2), reciprocal lattice vectors sorted by
    ascending |k|, including the Gamma point as the first entry.
  """
  lattice = jnp.asarray(lattice, dtype=jnp.float32)
  rec_lattice = 2.0 * jnp.pi * jnp.linalg.inv(lattice)   # (2,2)

  if min_kpoints is None:
    min_kpoints = int(sum(spins))
  elif min_kpoints < sum(spins):
    raise ValueError(
        "Number of kpoints must be equal or greater than number of electrons"
    )

  dk = 1.0 + 1e-5

  # 2D shell estimate (similar logic to 3D version but with exponent 1/2)
  max_k = int(jnp.ceil(min_kpoints * dk) ** (1.0 / 2.0))

  ordinals_1d = sorted(range(-max_k, max_k + 1), key=abs)
  ordinals = jnp.asarray(list(itertools.product(ordinals_1d, repeat=2)))  # (n_ord, 2)

  kpoints = ordinals @ rec_lattice.T  # (n_ord, 2)
  # sort by norm, so (0,0) is first
  kpoints = jnp.asarray(sorted(list(kpoints), key=lambda v: float(jnp.linalg.norm(v))))
  k_norms = jnp.linalg.norm(kpoints, axis=1)

  # keep all k with |k| <= |k|min_kpoints-1| * dk
  return kpoints[k_norms <= k_norms[min_kpoints - 1] * dk]
def make_multiwave_envelope_2d(kpoints: jnp.ndarray) -> envelopes.Envelope:
    """Returns an oscillatory envelope for 2D systems.

    Envelope consists of a sum of truncated 2D Fourier series, one centered on
    each atom, with Fourier frequencies given by kpoints:

      sigma_{2i}*cos(kpoints_i.r_{ae}) + sigma_{2i+1}*sin(kpoints_i.r_{ae})

    Initialization sets the coefficient of the first term in each
    series to 1, and all other coefficients to 0. This corresponds to the
    cosine of the first entry in kpoints. If this is [0, 0], the envelope
    will evaluate to unity at the beginning of training.

    Args:
        kpoints: Reciprocal lattice vectors of terms included in the Fourier
          series. Shape (nkpoints, ndim) (Note that ndim=2 for 2D systems).

    Returns:
        An instance of ferminet.envelopes.Envelope with apply_type
        envelopes.EnvelopeType.PRE_DETERMINANT
    """

    def init(
        natom: int, output_dims: Sequence[int], ndim: int = 2
    ) -> Sequence[Mapping[str, jnp.ndarray]]:
        """See ferminet.envelopes.EnvelopeInit."""
        del natom, ndim  # unused
        params = []
        nk = kpoints.shape[0]
        for output_dim in output_dims:
            params.append({'sigma': jnp.zeros((2 * nk, output_dim))})
            params[-1]['sigma'] = params[-1]['sigma'].at[0, :].set(1.0)
        return params

    def apply(*, ae: jnp.ndarray, r_ae: jnp.ndarray, r_ee: jnp.ndarray,
              sigma: jnp.ndarray) -> jnp.ndarray:
        """See ferminet.envelopes.EnvelopeApply."""
        del r_ae, r_ee  # unused
        phase_coords = ae @ kpoints.T
        waves = jnp.concatenate((jnp.cos(phase_coords), jnp.sin(phase_coords)),
                                axis=2)
        env = waves @ (sigma**2.0)
        return jnp.sum(env, axis=1)

    return envelopes.Envelope(envelopes.EnvelopeType.PRE_DETERMINANT, init, apply)


# def make_magnetic_envelope_2d(
#     lattice: jnp.ndarray,
#     elliptic_log_sigma,  # callable: log_weierstrass_sigma(z, N_phi, L1, L2, almost)
#     z_scale: float = 1.0,  # optional scale on (x + i y) before passing to σ
#     magfield_kwargs = {},
# ) -> envelopes.Envelope:
#   """Envelope based on Weierstrass σ with exact quasiperiods (2D).

#   Args:
#     lattice: (2, 2) matrix whose columns are primitive vectors L1, L2.
#     elliptic_log_sigma: function log_weierstrass_sigma(z, N_phi, L1, L2, almost).
#     z_scale: optional scaling on (x + i y) before passing to σ (default 1.0).

#   Returns:
#     ferminet.envelopes.Envelope with type PRE_DETERMINANT.
#   """
#   L1 = lattice[:, 0]
#   L2 = lattice[:, 1]
#   N_phi = magfield_kwargs["N_phi"]
#   almost = jnp.array(0.0)
#   zeros = magfield_kwargs["zeros"]

#   # ---------- init ----------
#   def init(natom: int, output_dims: Sequence[int], ndim: int = 2
#            ) -> Sequence[Mapping[str, jnp.ndarray]]:
#     del natom, ndim
#     params = []
#     for output_dim in output_dims:
#       params.append({'gain': jnp.ones((output_dim,))})
#     return params

#   # ---------- apply ----------
#   def apply(*, ae: jnp.ndarray, r_ae: jnp.ndarray, r_ee: jnp.ndarray,
#             gain: jnp.ndarray) -> jnp.ndarray:
#     """ae: (nelectron, natom, 2) -> returns (nelectron, output_dim)."""
#     del r_ae, r_ee  # unused

#     # Map ae -> complex z per electron–atom vector
#     # shape: (nelectron, natom)
#     z = z_scale * (ae[..., 0] + 1j * ae[..., 1]) / jnp.sqrt(2.0)

#     # Vectorized log σ for all (nelectron, natom) entries:
#     # vmap over atoms (axis=1) inside vmap over electrons (axis=0)
#     log_sigma_ea = jax.vmap(                   # over electrons
#         jax.vmap(lambda zz: elliptic_log_sigma(zz, N_phi, L1, L2, almost,zeros),
#                  in_axes=0),                   # over atoms
#         in_axes=0
#     )(z)

#     # Envelope per electron: sum over atoms of exp(log σ)
#     # shape: (nelectron,)
#     env_scalar = jnp.sum(jnp.exp(log_sigma_ea), axis=1)

#     # Broadcast to output_dim with trainable gain
#     # shape: (nelectron, output_dim)
#     return env_scalar[:, None] * gain[None, :]

#   return envelopes.Envelope(envelopes.EnvelopeType.PRE_DETERMINANT, init, apply)

def make_magnetic_envelope_2d(
    lattice: jnp.ndarray,
    elliptic_log_sigma=None,  # kept for API compatibility (unused)
    z_scale: float = 1.0,
    magfield_kwargs=None,
) -> envelopes.Envelope:
  """
  Global magnetic envelope based on cached Weierstrass σ / LLL implementation.

  - Uses ellipticfunctions._LLL_with_zeros_log_cached(z, zeros, consts)
    with a *fixed* set of zeros from magfield_kwargs["zeros"].
  - No trainable zeros, no holomorphic/anti-holomorphic split.
  - Envelope is PRE_DETERMINANT: gives a scalar per electron, broadcast over
    output_dim with a learnable gain.

  Expected magfield_kwargs:
    - "N_phi": integer flux quanta through the supercell
    - "zeros": array of complex zeros (shape (N_phi,) or (n_zero,)), in the
               same LLL complex convention as z = z_scale * (x + i y) / sqrt(2).
  """
  if magfield_kwargs is None:
    magfield_kwargs = {}

  # --- flux & zeros ---
  N_phi = int(magfield_kwargs["N_phi"])
  zeros = jnp.asarray(magfield_kwargs["zeros"], jnp.complex64)  # (n_zero,)

  # --- geometry constants (shared, cached like in mixed2) ---
  L1 = lattice[:, 0].astype(jnp.float32)
  L2 = lattice[:, 1].astype(jnp.float32)

  L1com = ellipticfunctions.to_cplx_divsqrt2(L1)
  L2com = ellipticfunctions.to_cplx_divsqrt2(L2)
  w1 = (L1com / 2.0).astype(jnp.complex64)
  w2 = (L2com / 2.0).astype(jnp.complex64)
  tau = (w2 / w1).astype(jnp.complex64)
  pi_c = jnp.asarray(jnp.pi, jnp.float32).astype(jnp.complex64)

  # Precompute θ₁ series coefficients once
  theta_coeffs = ellipticfunctions._precompute_theta_coeffs(tau, max_terms=15)
  t1p0, t1ppp0 = ellipticfunctions._theta_derivs0_from_coeffs(theta_coeffs)
  c = - (pi_c * pi_c) / (24.0 * w1 * w1) * (t1ppp0 / t1p0)

  # "almost" term (same as in your LLL envelopes)
  almost = ellipticfunctions.almost_modular(L1, L2, N_phi)

  consts_env = {
      "L1": L1,
      "L2": L2,
      "w1": w1,
      "w2": w2,
      "tau": tau,
      "pi": pi_c,
      "theta_coeffs": theta_coeffs,
      "t1p0": t1p0,
      "t1ppp0": t1ppp0,
      "c": c,
      "N_phi": jnp.asarray(N_phi, jnp.float32),
      "almost": almost,
  }
  consts_env = jax.tree.map(lambda x: jax.lax.stop_gradient(jnp.asarray(x)), consts_env)

  # ---------- init ----------
  def init(natom: int, output_dims: Sequence[int], ndim: int = 2
           ) -> Sequence[Mapping[str, jnp.ndarray]]:
    del natom, ndim
    params = []
    for odim in output_dims:
      params.append({"gain": jnp.ones((odim,), dtype=jnp.float32)})
    return params

  # ---------- apply ----------
  def apply(
      *,
      ae: jnp.ndarray,
      r_ae: jnp.ndarray,
      r_ee: jnp.ndarray,
      gain: jnp.ndarray,
  ) -> jnp.ndarray:
    """
    ae: (nelectron, natom, 2)
    returns: (nelectron, output_dim), complex64
    """
    del r_ae, r_ee  # unused

    # Map ae -> complex z per electron–atom vector
    # z: (nelectron, natom), complex
    ae_f = ae.astype(jnp.float32)
    z = z_scale * (ae_f[..., 0] + 1j * ae_f[..., 1]) 
    z = z.astype(jnp.complex64)

    # log ψ_env(z) using cached LLL-with-zeros log for each (e, A)
    def one_z(zz):
      # zz: scalar complex
      return ellipticfunctions._LLL_with_zeros_log_cached(zz, zeros, consts_env)

    # vmap over atoms then electrons
    log_env_ea = jax.vmap(                     # over electrons
        jax.vmap(one_z, in_axes=0),           # over atoms
        in_axes=0,
    )(z)  # shape (nelectron, natom)

    # Envelope per electron: sum over atoms of exp(log ψ_env)
    env_scalar = jnp.sum(jnp.exp(log_env_ea), axis=1)  # (nelectron,)

    # Broadcast to output_dim with trainable gain
    env_scalar = env_scalar.astype(jnp.complex64)
    gain_c = gain.astype(env_scalar.dtype)
    env_eo = env_scalar[:, None] * gain_c[None, :]      # (nelectron, output_dim)

    return env_eo

  return envelopes.Envelope(envelopes.EnvelopeType.PRE_DETERMINANT, init, apply)


# def make_magnetic_envelope_2d(
#     lattice: jnp.ndarray,
#     elliptic_log_sigma,  # callable: log_weierstrass_sigma(z, N_phi, L1, L2, almost)
#     z_scale: float = 1.0,  # optional scale on (x + i y) before passing to σ
#     magfield_kwargs = {},
# ) -> envelopes.Envelope:
#   """Envelope based on Weierstrass σ with exact quasiperiods (2D).

#   Args:
#     lattice: (2, 2) matrix whose columns are primitive vectors L1, L2.
#     elliptic_log_sigma: function log_weierstrass_sigma(z, N_phi, L1, L2, almost).
#     z_scale: optional scaling on (x + i y) before passing to σ (default 1.0).

#   Returns:
#     ferminet.envelopes.Envelope with type PRE_DETERMINANT.
#   """
#   L1 = lattice[:, 0]
#   L2 = lattice[:, 1]
#   N_phi = magfield_kwargs["N_phi"]
#   almost = jnp.array(0.0)
#   zeros = magfield_kwargs["zeros"]

#   # ---------- init ----------
#   def init(natom: int, output_dims: Sequence[int], ndim: int = 2
#            ) -> Sequence[Mapping[str, jnp.ndarray]]:
#     del natom, ndim
#     params = []
#     for output_dim in output_dims:
#       params.append({'gain': jnp.ones((output_dim,))})
#     return params

#   # ---------- apply ----------
#   def apply(*, ae: jnp.ndarray, r_ae: jnp.ndarray, r_ee: jnp.ndarray,
#             gain: jnp.ndarray) -> jnp.ndarray:
#     """ae: (nelectron, natom, 2) -> returns (nelectron, output_dim)."""
#     del r_ae, r_ee  # unused

#     # Map ae -> complex z per electron–atom vector
#     # shape: (nelectron, natom)
#     z = z_scale * (ae[..., 0] + 1j * ae[..., 1]) / jnp.sqrt(2.0)

#     # Vectorized log σ for all (nelectron, natom) entries:
#     # vmap over atoms (axis=1) inside vmap over electrons (axis=0)
#     log_sigma_ea = jax.vmap(                   # over electrons
#         jax.vmap(lambda zz: elliptic_log_sigma(zz, N_phi, L1, L2, almost,zeros),
#                  in_axes=0),                   # over atoms
#         in_axes=0
#     )(z)

#     # Envelope per electron: sum over atoms of exp(log σ)
#     # shape: (nelectron,)
#     env_scalar = jnp.sum(jnp.exp(log_sigma_ea), axis=1)

#     # Broadcast to output_dim with trainable gain
#     # shape: (nelectron, output_dim)
#     return env_scalar[:, None] * gain[None, :]

#   return envelopes.Envelope(envelopes.EnvelopeType.PRE_DETERMINANT, init, apply)

def make_magnetic_envelope_2d_mod(
    lattice: jnp.ndarray,
    elliptic_log_sigma,  # callable: log_weierstrass_sigma(z, N_phi, L1, L2, almost, zeros)
    z_scale: float = 1.0,  # optional scale on (x + i y) before passing to σ
    magfield_kwargs = {},
) -> envelopes.Envelope:
  """Single-electron, single-atom magnetic σ-envelope (no vmap, no loops)."""

  # lattice columns are primitive vectors L1, L2
  L1 = lattice[:, 0]
  L2 = lattice[:, 1]
  Linv = jnp.linalg.inv(lattice)

  N_phi = magfield_kwargs["N_phi"]
  zeros = magfield_kwargs["zeros"]
  use_nearest_image = True
  almost = jnp.array(0.0, dtype=jnp.float32)

  def init(natom: int, output_dims: Sequence[int], ndim: int = 2
           ) -> Sequence[Mapping[str, jnp.ndarray]]:
    del natom, ndim
    return [{'gain': jnp.ones((odim,))} for odim in output_dims]

  def apply(*, ae: jnp.ndarray, r_ae: jnp.ndarray, r_ee: jnp.ndarray,
            gain: jnp.ndarray) -> jnp.ndarray:
    """ae: (1,1,2) -> returns (1, output_dim)."""
    del r_ae, r_ee

    # pull the single displacement r (shape (2,))
    r = ae[0, 0, :]
    rdtype = r.dtype

    # fractional lattice coords t: r = t1 L1 + t2 L2
    t = Linv @ r  # (2,)
    n = jnp.floor(t + 0.5) if use_nearest_image else jnp.floor(t)
    n = jax.lax.stop_gradient(n).astype(jnp.int32)

    # lattice translation L = n1 L1 + n2 L2
    L_vec = lattice @ n.astype(rdtype)
    L_vec = jax.lax.stop_gradient(L_vec)

    # reduced position r' = r - L
    r_prime = r - L_vec

    # η = +1 iff both windings are even, else −1
    both_even = jnp.all((n & 1) == 0)
    eta = jnp.where(both_even, jnp.array(1.0, rdtype), jnp.array(-1.0, rdtype))

    # phase = η * exp(i/2 * (r'×L)), with 2D scalar cross r'×L = r'_x L_y − r'_y L_x
    theta = jnp.array(0.5, rdtype) * (r_prime[0] * L_vec[1] - r_prime[1] * L_vec[0])
    phase = (eta + 0j) * jnp.exp(-1j * theta)

    # complex LLL coordinate z' = z_scale * (x'+ i y') / √2
    sqrt2 = jnp.sqrt(jnp.array(2.0, rdtype))
    z_prime = z_scale * (r_prime[0] + 1j * r_prime[1]) / sqrt2

    # log σ̂ with exact quasiperiods (single call, no vmap)
    log_sigma = elliptic_log_sigma(z_prime, N_phi, L1, L2, almost, zeros)

    # envelope scalar for the single electron
    env_scalar = phase * jnp.exp(log_sigma)  # complex scalar

    # expand to (1, output_dim)
    gain_c = gain.astype(env_scalar.real.dtype) + 0j
    out_row = env_scalar * gain_c  # (output_dim,)
    return out_row[None, :]

  return envelopes.Envelope(envelopes.EnvelopeType.PRE_DETERMINANT, init, apply)



# def make_magnetic_envelope_2d_trainable_zeros(
#     lattice: jnp.ndarray,
#     elliptic_log_sigma,              # callable: log_weierstrass_sigma(z, N_phi, L1, L2, almost, zeros)
#     z_scale: float = 1.0,
#     magfield_kwargs = {},
# ) -> envelopes.Envelope:
#   """Envelope based on Weierstrass σ with exact quasiperiods (2D) and
#   trainable zeros constrained to sum to zero.

#   Args:
#     lattice: (2, 2) matrix whose columns are primitive vectors L1, L2.
#     elliptic_log_sigma: function log_weierstrass_sigma(z, N_phi, L1, L2, almost, zeros).
#     z_scale: optional scaling on (x + i y) before passing to σ.
#     magfield_kwargs: expects:
#         - "N_phi": int, number of flux quanta (zeros count).
#         - optional "zeros": initial complex array of shape (N_phi,).

#   Returns:
#     ferminet.envelopes.Envelope with type PRE_DETERMINANT.
#   """
#   L1 = lattice[:, 0]
#   L2 = lattice[:, 1]
#   N_phi = magfield_kwargs["N_phi"]
#   almost = jnp.array(0.0)

#   zeros_init = magfield_kwargs.get("zeros", None)
#   if zeros_init is not None:
#     zeros_init = jnp.asarray(zeros_init)
#     if zeros_init.shape != (N_phi,):
#       raise ValueError(f"zeros init must have shape ({N_phi},), got {zeros_init.shape}")

#   # ---------- init ----------
#   def init(natom: int, output_dims: Sequence[int], ndim: int = 2
#            ) -> Sequence[Mapping[str, jnp.ndarray]]:
#     del natom, ndim
#     params = []
#     for _odim in output_dims:
#       if zeros_init is None:
#         # start all zeros; you can add tiny deterministic offsets if desired
#         zr = jnp.zeros((N_phi,), dtype=jnp.complex64)
#       else:
#         zr = zeros_init.astype(jnp.complex64)
#       params.append({
#           'gain': jnp.ones((_odim,)),
#           'zeros_raw': zr,    # trainable complex parameters
#       })
#     return params

#   # ---------- apply ----------
#   def apply(*, ae: jnp.ndarray, r_ae: jnp.ndarray, r_ee: jnp.ndarray,
#             gain: jnp.ndarray, zeros_raw: jnp.ndarray) -> jnp.ndarray:
#     """ae: (nelectron, natom, 2) -> returns (nelectron, output_dim)."""
#     del r_ae, r_ee  # unused

#     # Map ae -> complex z per electron–atom vector
#     # shape: (nelectron, natom)
#     z = z_scale * (ae[..., 0] + 1j * ae[..., 1]) / jnp.sqrt(2.0)

#     # Enforce sum_k zeros_k = 0 exactly each pass
#     zeros_centered = zeros_raw - jnp.mean(zeros_raw)

#     # (Optional) If you want zeros wrapped into the fundamental cell, uncomment:
#     Linv = jnp.linalg.inv(lattice)
#     def wrap_to_cell(zc):
#       # complex -> 2-vector
#       r = jnp.array([jnp.real(zc), jnp.imag(zc)]) * jnp.sqrt(2.0) / z_scale
#       t = Linv @ r                    # fractional coords
#       t = t - jnp.floor(t)            # [0,1)
#       r_wrapped = lattice @ t
#       z_wrapped = (r_wrapped[0] + 1j * r_wrapped[1]) / jnp.sqrt(2.0) * z_scale
#       return z_wrapped
#     zeros_centered = jax.vmap(wrap_to_cell)(zeros_centered)
#     zeros_centered = zeros_centered - jnp.mean(zeros_centered)

#     # Vectorized log σ for all (nelectron, natom)
#     log_sigma_ea = jax.vmap(                   # over electrons
#         jax.vmap(lambda zz: elliptic_log_sigma(zz, N_phi, L1, L2, almost, zeros_centered),
#                  in_axes=0),                   # over atoms
#         in_axes=0
#     )(z)

#     # Envelope per electron: sum over atoms of exp(log σ)
#     env_scalar = jnp.sum(jnp.exp(log_sigma_ea), axis=1)  # (nelectron,)

#     # Broadcast to output_dim with trainable gain
#     return env_scalar[:, None] * gain[None, :]

#   return envelopes.Envelope(envelopes.EnvelopeType.PRE_DETERMINANT, init, apply)

def make_magnetic_envelope_2d_trainable_zeros(
    lattice: jnp.ndarray,
    elliptic_log_sigma,     # log_weierstrass_sigma(z, N_phi, L1, L2, almost, zeros)
    z_scale: float = 1.0,
    magfield_kwargs = {},
) -> envelopes.Envelope:
  L1 = lattice[:, 0]
  L2 = lattice[:, 1]
  N_phi = int(magfield_kwargs["N_phi"])
  almost = jnp.asarray(0.0, jnp.float32)

  # initial zeros: complex array provided by you (shape [nz,])
  init_zeros_c = jnp.asarray(magfield_kwargs["zeros"], jnp.complex64)
  nz = int(init_zeros_c.shape[0])

  # ---------- init ----------
  def init(natom: int, output_dims: Sequence[int], ndim: int = 2
           ) -> Sequence[Mapping[str, jnp.ndarray]]:
    del natom, ndim
    # store zeros as REAL params: (nz, 2) float32
    zeros_xy0 = jnp.stack([jnp.real(init_zeros_c), jnp.imag(init_zeros_c)], axis=-1).astype(jnp.float32)
    zeros_xy0 = zeros_xy0 - jnp.mean(zeros_xy0, axis=0, keepdims=True)  # enforce sum=0 at init

    params = []
    for output_dim in output_dims:
      params.append({
        'gain': jnp.ones((output_dim,), dtype=jnp.float32),
        'zeros_xy': zeros_xy0,                     # trainable, real, fixed shape
      })
    return params

  # ---------- apply ----------
  def apply(*, ae: jnp.ndarray, r_ae: jnp.ndarray, r_ee: jnp.ndarray,
            gain: jnp.ndarray, zeros_xy: jnp.ndarray) -> jnp.ndarray:
    del r_ae, r_ee
    # map ae -> complex z
    z = z_scale * (ae[..., 0] + 1j * ae[..., 1]) / jnp.sqrt(jnp.array(2.0, dtype=jnp.float32))

    # reconstruct complex zeros from real params; enforce sum=0 each call
    zeros_c = (zeros_xy[..., 0].astype(jnp.float32) + 1j * zeros_xy[..., 1].astype(jnp.float32)).astype(jnp.complex64)
    #zeros_c = zeros_c - jnp.mean(zeros_c)  # enforce Σ a = 0 (periodic BCs)

    # vmapped log σ with trainable zeros
    log_sigma_ea = jax.vmap(
        jax.vmap(lambda zz: elliptic_log_sigma(
            zz, jnp.asarray(N_phi, jnp.int32), L1, L2, almost, zeros_c),
            in_axes=0),
        in_axes=0
    )(z)  # (ne, na) complex

    env_scalar = jnp.sum(jnp.exp(log_sigma_ea), axis=1)        # complex
    gain_c = gain.astype(jnp.float32) + 0j                     # complex
    return env_scalar[:, None] * gain_c[None, :]

  return envelopes.Envelope(envelopes.EnvelopeType.PRE_DETERMINANT, init, apply)



# def make_LLL_envelope_2d_trainable_zeros(
#     lattice: jnp.ndarray,
#     elliptic_log_sigma,      # e.g. LLL_with_zeros_log(z, N_phi, L1, L2, almost, zeros)
#     z_scale: float = 1.0,
#     magfield_kwargs = None,
# ) -> envelopes.Envelope:
#   """LLL envelope with per-orbital trainable zeros (2D, PRE_DETERMINANT).

#   Args:
#     lattice: (2, 2) matrix whose columns are primitive vectors L1, L2.
#     elliptic_log_sigma: callable(z, N_phi, L1, L2, almost, zeros)
#       returning log(envelope) for a *single* electron and a given set of zeros.
#     z_scale: optional scale factor on (x + i y) before passing to σ.
#     magfield_kwargs: dict with at least:
#         - "N_phi": int (flux quanta)
#         - "zeros_init": array (nzeros, 2) real, initial zeros in complex plane
#         - optional "almost": scalar (the modular/G2 correction term)

#   Returns:
#     ferminet.envelopes.Envelope with type PRE_DETERMINANT.
#     For each electron and orbital k, the envelope is:
#         env[e, k] = exp(elliptic_log_sigma(z_e, N_phi, L1, L2, almost, zeros_k))
#   """

#   if magfield_kwargs is None:
#     magfield_kwargs = {}

#   L1 = lattice[:, 0]
#   L2 = lattice[:, 1]

#   # N_phi as scalar float32 (elliptic_log_sigma can cast as needed)
#   N_phi = jnp.asarray(magfield_kwargs["N_phi"], dtype=jnp.float32)

#   # Almost term (Gaussian/modular correction); default 0.0 if not provided
#   almost = jnp.asarray(magfield_kwargs.get("almost", 0.0), dtype=jnp.float32)

#   # Initial zeros: (nzeros, 2) real
#   #zeros_init = jnp.asarray(magfield_kwargs["zeros_init"], dtype=jnp.float32)
#   #nzeros = zeros_init.shape[0]

#   # ---------- init ----------
#   def init(natom: int, output_dims: Sequence[int], ndim: int = 2
#           ) -> Sequence[Mapping[str, jnp.ndarray]]:
#       del natom, ndim

#       params = []
#       key = jax.random.PRNGKey(1234)    # you may pass this in instead

#       #N_phi = magfield_kwargs["N_phi"]
#       N_phi = int(jnp.asarray(magfield_kwargs["N_phi"], dtype=jnp.int32))

#       for odim in output_dims:
#           # One trainable zero-set per orbital
#           # Shape: (odim, N_phi, 2)

#           # Split rng for each orbital block
#           key, subkey = jax.random.split(key)

#           # Sample zeros uniformly inside supercell spanned by L1,L2
#           # u,v ∈ [0,1)
#           uv = jax.random.uniform(subkey, (odim, N_phi, 2), dtype=jnp.float32)

#           # Convert from lattice coords (u,v) to real 2D coordinates
#           # r = u L1 + v L2
#           zeros_init = (uv[..., 0:1] * L1[None,None,:] +
#                         uv[..., 1:2] * L2[None,None,:])

#           # Enforce zero-mean per orbital
#           zeros_mean = jnp.mean(zeros_init, axis=1, keepdims=True)
#           zeros_centered = zeros_init - zeros_mean

#           params.append({
#               'gain': jnp.ones((odim,), dtype=jnp.float32),
#               'zeros_unconstrained': zeros_centered.astype(jnp.float32)
#           })

#       return params

#   # ---------- apply ----------
#   def apply(*, ae: jnp.ndarray, r_ae: jnp.ndarray, r_ee: jnp.ndarray,
#             gain: jnp.ndarray,
#             zeros_unconstrained: jnp.ndarray) -> jnp.ndarray:
#     """ae: (nelectron, natom, 2) -> returns (nelectron, output_dim)."""
#     del r_ae, r_ee  # unused

#     # We assume a single dummy atom at the origin: use ae[..., 0, :]
#     # positions: (nelectron, 2)
#     pos = ae[:, 0, :].astype(jnp.float32)

#     # Map to complex LLL coordinate z = (x + i y)/sqrt(2)
#     z = z_scale * (pos[..., 0] + 1j * pos[..., 1]) / jnp.sqrt(2.0)
#     z = z.astype(jnp.complex64)          # (ne,)

#     ne = z.shape[0]
#     odim = gain.shape[0]

#     # Enforce Σ_a zeros_a = 0 separately for each orbital:
#     # zeros_unconstrained: (odim, nzeros, 2)
#     mean_zeros = jnp.mean(zeros_unconstrained, axis=1, keepdims=True)    # (odim, 1, 2)
#     zeros_centered = zeros_unconstrained - mean_zeros                    # (odim, nzeros, 2)

#     # Convert to complex: (odim, nzeros)
#     zeros_complex = (zeros_centered[..., 0] + 1j * zeros_centered[..., 1]).astype(jnp.complex64)

#     # Define one-orbital, many-electron log-envelope
#     def log_env_one_orbital(zeros_k: jnp.ndarray) -> jnp.ndarray:
#       """zeros_k: (nzeros,) complex64 -> log_env(e) shape (ne,) complex."""
#       def log_env_one_electron(z_e):
#         return elliptic_log_sigma(z_e, N_phi, L1, L2, almost, zeros_k)
#       return jax.vmap(log_env_one_electron)(z)      # (ne,)

#     # Vectorize over orbitals: (odim, ne)
#     log_env_od = jax.vmap(log_env_one_orbital, in_axes=0)(zeros_complex)  # (odim, ne)
#     log_env_eo = jnp.swapaxes(log_env_od, 0, 1)                           # (ne, odim)

#     # Envelope per electron and orbital
#     env_eo = jnp.exp(log_env_eo).astype(jnp.complex64)    # (ne, odim)
#     gain_c = gain.astype(env_eo.dtype)                    # real -> complex-safe
#     env_eo = env_eo * gain_c[None, :]

#     return env_eo

#   return envelopes.Envelope(envelopes.EnvelopeType.PRE_DETERMINANT, init, apply)

def make_LLL_envelope_2d_trainable_zeros(
    lattice: jnp.ndarray,
    elliptic_log_sigma,
    z_scale: float = 1.0,
    magfield_kwargs = None,
) -> envelopes.Envelope:
  if magfield_kwargs is None:
    magfield_kwargs = {}

  L1 = lattice[:, 0]
  L2 = lattice[:, 1]

  N_phi = jnp.asarray(magfield_kwargs["N_phi"], dtype=jnp.float32)
  almost = jnp.asarray(magfield_kwargs.get("almost", 0.0), dtype=jnp.float32)

  def init(natom: int, output_dims: Sequence[int], ndim: int = 2
          ) -> Sequence[Mapping[str, jnp.ndarray]]:
    del natom, ndim
    params = []
    key = jax.random.PRNGKey(1234)

    N_phi_int = int(jnp.asarray(magfield_kwargs["N_phi"], dtype=jnp.int32))

    for odim in output_dims:
      key, subkey = jax.random.split(key)
      uv = jax.random.uniform(subkey, (odim, N_phi_int, 2), dtype=jnp.float32)
      zeros_init = (uv[..., 0:1] * L1[None, None, :] +
                    uv[..., 1:2] * L2[None, None, :])
      zeros_mean = jnp.mean(zeros_init, axis=1, keepdims=True)
      zeros_centered = zeros_init - zeros_mean

      params.append({
          'gain': jnp.ones((odim,), dtype=jnp.float32),
          'zeros_unconstrained': zeros_centered.astype(jnp.float32),
      })

    return params

  def apply(*, ae: jnp.ndarray, r_ae: jnp.ndarray, r_ee: jnp.ndarray,
            gain: jnp.ndarray,
            zeros_unconstrained: jnp.ndarray) -> jnp.ndarray:
    """ae: (nelectron, natom, 2) -> returns (nelectron, output_dim)."""
    del r_ae, r_ee

    pos = ae[:, 0, :].astype(jnp.float32)  # (ne, 2)

    z = z_scale * (pos[..., 0] + 1j * pos[..., 1]) / jnp.sqrt(2.0)
    z = z.astype(jnp.complex64)  # (ne,)

    ne = int(z.shape[0])
    odim = int(gain.shape[0])

    mean_zeros = jnp.mean(zeros_unconstrained, axis=1, keepdims=True)
    zeros_centered = zeros_unconstrained - mean_zeros
    zeros_complex = (zeros_centered[..., 0] + 1j * zeros_centered[..., 1]).astype(
        jnp.complex64
    )  # (odim, nzeros)

    env_eo = jnp.zeros((ne, odim), dtype=jnp.complex64)

    N_phi_f32 = N_phi

    # plain Python loops (JAX will unroll them; no scan/fori/vmap)
    for k in range(odim):
      zeros_k = zeros_complex[k]  # (nzeros,)
      for i in range(ne):
        z_e = z[i]
        log_val = elliptic_log_sigma(z_e, N_phi_f32, L1, L2, almost, zeros_k)
        val = jnp.exp(log_val).astype(jnp.complex64)
        env_eo = env_eo.at[i, k].set(val)

    gain_c = gain.astype(env_eo.dtype)
    env_eo = env_eo * gain_c[None, :]

    return env_eo

  return envelopes.Envelope(envelopes.EnvelopeType.PRE_DETERMINANT, init, apply)



def make_LLL_envelope_2d_trainable_zeros_mixed(
    lattice: jnp.ndarray,
    elliptic_log_sigma,      # e.g. LLL_with_zeros_log(z, N_phi, L1, L2, almost, zeros)
    z_scale: float = 1.0,
    magfield_kwargs = None,
) -> envelopes.Envelope:
  """LLL-like envelope with per-orbital trainable holo + anti-holo zeros (2D).

  Args:
    lattice: (2, 2) matrix whose columns are primitive vectors L1, L2.
    elliptic_log_sigma: callable(z, N_phi, L1, L2, almost, zeros)
      returning log(envelope) for a *single* electron and a given set of zeros.
      This is treated as the holomorphic building block; the anti-holo part
      is modeled via complex conjugation.
    z_scale: optional scale factor on (x + i y) before passing to σ.
    magfield_kwargs: dict with at least:
        - "N_phi": int (flux quanta)
        - "N_holo": int, number of holomorphic zeros per orbital
        - "N_anti": int, number of anti-holomorphic zeros per orbital
        - optional "almost": scalar (the modular/G2 correction term)

      We enforce N_holo - N_anti == N_phi.

  Returns:
    ferminet.envelopes.Envelope with type PRE_DETERMINANT.
    For each electron and orbital k, the envelope is:
        env[e, k] = exp( log_env_holo(z_e; zeros_holo_k)
                         + log_env_anti(z_e; zeros_anti_k) )
  """

  if magfield_kwargs is None:
    magfield_kwargs = {}

  # Lattice vectors as float32
  L1 = lattice[:, 0].astype(jnp.float32)
  L2 = lattice[:, 1].astype(jnp.float32)

  # Flux
  N_phi = int(magfield_kwargs["N_phi"])

  # Split into holo / anti-holo zeros (default: all holomorphic)
  N_holo = int(magfield_kwargs.get("N_holo", N_phi))
  N_anti = int(magfield_kwargs.get("N_anti", 0))

  assert N_holo - N_anti == N_phi, "Require N_holo + N_anti == N_phi"

  # Almost term (Gaussian/modular correction)
  almost = jnp.asarray(magfield_kwargs.get("almost", 0.0), dtype=jnp.float32)

  # ---------- init ----------
  def init(natom: int, output_dims: Sequence[int], ndim: int = 2
          ) -> Sequence[Mapping[str, jnp.ndarray]]:
    del natom, ndim

    params = []
    key = jax.random.PRNGKey(99)    # deterministic; can be externalised

    for odim in output_dims:
      # Split rng for this envelope block
      key, subkey_h = jax.random.split(key)
      key, subkey_a = jax.random.split(key)

      # ---------------- holomorphic zeros ----------------
      if N_holo > 0:
        uv_h = jax.random.uniform(subkey_h, (odim, N_holo, 2), dtype=jnp.float32)
        zeros_h_init = (uv_h[..., 0:1] * L1[None, None, :] +
                        uv_h[..., 1:2] * L2[None, None, :])
        zeros_h_mean = jnp.mean(zeros_h_init, axis=1, keepdims=True)
        zeros_h_centered = zeros_h_init - zeros_h_mean
      else:
        zeros_h_centered = jnp.zeros((odim, 0, 2), dtype=jnp.float32)

      # ---------------- anti-holomorphic zeros ----------------
      if N_anti > 0:
        uv_a = jax.random.uniform(subkey_a, (odim, N_anti, 2), dtype=jnp.float32)
        zeros_a_init = (uv_a[..., 0:1] * L1[None, None, :] +
                        uv_a[..., 1:2] * L2[None, None, :])
        zeros_a_mean = jnp.mean(zeros_a_init, axis=1, keepdims=True)
        zeros_a_centered = zeros_a_init - zeros_a_mean
      else:
        zeros_a_centered = jnp.zeros((odim, 0, 2), dtype=jnp.float32)

      params.append({
          'gain': jnp.ones((odim,), dtype=jnp.float32),
          'zeros_holo_unconstrained': zeros_h_centered.astype(jnp.float32),
          'zeros_anti_unconstrained': zeros_a_centered.astype(jnp.float32),
      })

    return params

  # ---------- apply ----------
  def apply(*, ae: jnp.ndarray, r_ae: jnp.ndarray, r_ee: jnp.ndarray,
            gain: jnp.ndarray,
            zeros_holo_unconstrained: jnp.ndarray,
            zeros_anti_unconstrained: jnp.ndarray) -> jnp.ndarray:
    """ae: (nelectron, natom, 2) -> returns (nelectron, output_dim)."""
    del r_ae, r_ee  # unused

    # Assume a single dummy atom at origin: ae[:, 0, :]
    pos = ae[:, 0, :].astype(jnp.float32)  # (ne, 2)

    # Complex LLL coordinate
    z = z_scale * (pos[..., 0] + 1j * pos[..., 1]) / jnp.sqrt(2.0)
    z = z.astype(jnp.complex64)                       # (ne,)

    ne = z.shape[0]
    odim = gain.shape[0]

    # Enforce Σ zeros = 0 separately for each orbital and each sector

    # --- holomorphic zeros ---
    if N_holo > 0:
      mean_h = jnp.mean(zeros_holo_unconstrained, axis=1, keepdims=True)   # (odim, 1, 2)
      zeros_h_centered = zeros_holo_unconstrained - mean_h                 # (odim, N_holo, 2)
      zeros_h_c = (zeros_h_centered[..., 0] + 1j * zeros_h_centered[..., 1]).astype(jnp.complex64)  # (odim, N_holo)
    else:
      zeros_h_c = jnp.zeros((odim, 0), dtype=jnp.complex64)

    # --- anti-holomorphic zeros ---
    if N_anti > 0:
      mean_a = jnp.mean(zeros_anti_unconstrained, axis=1, keepdims=True)   # (odim, 1, 2)
      zeros_a_centered = zeros_anti_unconstrained - mean_a                 # (odim, N_anti, 2)
      zeros_a_c = (zeros_a_centered[..., 0] + 1j * zeros_a_centered[..., 1]).astype(jnp.complex64)  # (odim, N_anti)
    else:
      zeros_a_c = jnp.zeros((odim, 0), dtype=jnp.complex64)

    # Helper: holomorphic log-envelope, many electrons for one orbital
    N_phi_f32 = jnp.asarray(N_phi, dtype=jnp.float32)

    def log_env_holo_one_orbital(zeros_k: jnp.ndarray) -> jnp.ndarray:
      """zeros_k: (N_holo,) → log_env_holo(e) shape (ne,) complex."""
      if zeros_k.shape[0] == 0:
        return jnp.zeros((ne,), dtype=jnp.complex64)

      def one_e(z_e):
        return elliptic_log_sigma(z_e, N_phi_f32, L1, L2, almost, zeros_k)
      return jax.vmap(one_e)(z)  # (ne,)

    # Anti-holomorphic part modeled as conjugate of holo-like piece at conj(z)
    def log_env_anti_one_orbital(zeros_k: jnp.ndarray) -> jnp.ndarray:
      """zeros_k: (N_anti,) → log_env_anti(e) shape (ne,) complex."""
      if zeros_k.shape[0] == 0:
        return jnp.zeros((ne,), dtype=jnp.complex64)

      z_conj = jnp.conj(z)

      def one_e(z_e_conj):
        # treat z_e_conj as the "holomorphic" variable, then conjugate result
        val = elliptic_log_sigma(z_e_conj, N_phi_f32, L1, L2, almost, zeros_k)
        return jnp.conj(val)

      return jax.vmap(one_e)(z_conj)  # (ne,)

    # Vectorize over orbitals → (odim, ne)
    log_env_holo_od = jax.vmap(log_env_holo_one_orbital, in_axes=0)(zeros_h_c)   # (odim, ne)
    log_env_anti_od = jax.vmap(log_env_anti_one_orbital, in_axes=0)(zeros_a_c)   # (odim, ne)

    # Swap to (ne, odim)
    log_env_holo_eo = jnp.swapaxes(log_env_holo_od, 0, 1)  # (ne, odim)
    log_env_anti_eo = jnp.swapaxes(log_env_anti_od, 0, 1)  # (ne, odim)

    # Total log envelope
    log_env_eo = log_env_holo_eo + log_env_anti_eo         # (ne, odim)

    # Convert to envelope and apply gain
    env_eo = jnp.exp(log_env_eo).astype(jnp.complex64)
    gain_c = gain.astype(env_eo.dtype)
    env_eo = env_eo * gain_c[None, :]

    return env_eo

  return envelopes.Envelope(envelopes.EnvelopeType.PRE_DETERMINANT, init, apply)


def make_LLL_envelope_2d_trainable_zeros_mixed2(
    lattice: jnp.ndarray,
    elliptic_log_sigma=None,   # kept for API compatibility
    z_scale: float = 1.0,
    magfield_kwargs=None,
) -> envelopes.Envelope:
  """
  LLL-like envelope with per-orbital trainable holomorphic + anti-holomorphic zeros.

  - Holomorphic sector: ψ_holo ~ LLL_with_zeros_log_cached(z ; zeros_h, N_holo)
  - Anti-holo sector:   ψ_anti ~ conj( LLL_with_zeros_log_cached(conj(z) ; zeros_a, N_anti) )

  The net winding is controlled by:
      N_holo - N_anti = N_phi   (asserted)

  All geometric/theta constants are precomputed once and *not* part of params.
  Trainable params:
    - zeros_holo_unconstrained (per orbital)
    - zeros_anti_unconstrained (per orbital)
    - gain (per orbital)
  """
  if magfield_kwargs is None:
    magfield_kwargs = {}

  # --- flux bookkeeping ---
  N_phi  = int(magfield_kwargs["N_phi"])
  N_holo = int(magfield_kwargs.get("N_holo", N_phi))
  N_anti = int(magfield_kwargs.get("N_anti", 0))

  # Your corrected condition:
  assert N_holo - N_anti == N_phi, (
      f"Require N_holo - N_anti == N_phi, got N_holo={N_holo}, "
      f"N_anti={N_anti}, N_phi={N_phi}"
  )

  # --- geometry constants (shared) ---
  L1 = lattice[:, 0].astype(jnp.float32)
  L2 = lattice[:, 1].astype(jnp.float32)

  L1com = ellipticfunctions.to_cplx_divsqrt2(L1)
  L2com = ellipticfunctions.to_cplx_divsqrt2(L2)
  w1 = (L1com / 2.0).astype(jnp.complex64)
  w2 = (L2com / 2.0).astype(jnp.complex64)
  tau = (w2 / w1).astype(jnp.complex64)
  pi_c = jnp.asarray(jnp.pi, jnp.float32).astype(jnp.complex64)

  # Precompute θ₁ series coefficients once
  theta_coeffs = ellipticfunctions._precompute_theta_coeffs(tau, max_terms=15)
  t1p0, t1ppp0 = ellipticfunctions._theta_derivs0_from_coeffs(theta_coeffs)
  c = - (pi_c * pi_c) / (24.0 * w1 * w1) * (t1ppp0 / t1p0)

  # "almost" term, shared for both sectors
  #almost = jnp.asarray(magfield_kwargs.get("almost", 0.0), jnp.complex64)
  almost = ellipticfunctions.almost_modular(L1,L2,magfield_kwargs["N_phi"])

  # consts for holo and anti sectors (differ only by N_phi)
  consts_holo = {
      'L1': L1,
      'L2': L2,
      'w1': w1,
      'w2': w2,
      'tau': tau,
      'pi': pi_c,
      'theta_coeffs': theta_coeffs,
      't1p0': t1p0,
      't1ppp0': t1ppp0,
      'c': c,
      'N_phi': jnp.asarray(N_holo, jnp.float32),
      'almost': almost,
  }
  consts_anti = {
      'L1': L1,
      'L2': L2,
      'w1': w1,
      'w2': w2,
      'tau': tau,
      'pi': pi_c,
      'theta_coeffs': theta_coeffs,
      't1p0': t1p0,
      't1ppp0': t1ppp0,
      'c': c,
      'N_phi': jnp.asarray(max(N_anti, 1), jnp.float32),  # avoid div-by-zero; we won't use if N_anti==0
      'almost': almost,
  }

  consts_holo = jax.tree.map(lambda x: jax.lax.stop_gradient(jnp.asarray(x)), consts_holo)
  consts_anti = jax.tree.map(lambda x: jax.lax.stop_gradient(jnp.asarray(x)), consts_anti)

  # ---------- init: sample zeros per orbital in each sector ----------
  def init(natom: int, output_dims: Sequence[int], ndim: int = 2
          ) -> Sequence[Mapping[str, jnp.ndarray]]:
    del natom, ndim

    params = []
    key = jax.random.PRNGKey(1234)

    for odim in output_dims:
      key, subkey_h = jax.random.split(key)
      key, subkey_a = jax.random.split(key)

      # --- holomorphic zeros ---
      if N_holo > 0:
        uv_h = jax.random.uniform(subkey_h, (odim, N_holo, 2), dtype=jnp.float32)
        zeros_h_init = (uv_h[..., 0:1] * L1[None, None, :] +
                        uv_h[..., 1:2] * L2[None, None, :])
        zeros_h_mean = jnp.mean(zeros_h_init, axis=1, keepdims=True)
        zeros_h_centered = zeros_h_init - zeros_h_mean
      else:
        zeros_h_centered = jnp.zeros((odim, 0, 2), dtype=jnp.float32)

      # --- anti-holomorphic zeros ---
      if N_anti > 0:
        uv_a = jax.random.uniform(subkey_a, (odim, N_anti, 2), dtype=jnp.float32)
        zeros_a_init = (uv_a[..., 0:1] * L1[None, None, :] +
                        uv_a[..., 1:2] * L2[None, None, :])
        zeros_a_mean = jnp.mean(zeros_a_init, axis=1, keepdims=True)
        zeros_a_centered = zeros_a_init - zeros_a_mean
      else:
        zeros_a_centered = jnp.zeros((odim, 0, 2), dtype=jnp.float32)

      params.append({
          'gain': jnp.ones((odim,), dtype=jnp.float32),
          'zeros_holo_unconstrained': zeros_h_centered.astype(jnp.float32),
          'zeros_anti_unconstrained': zeros_a_centered.astype(jnp.float32),
      })

    return params

  # ---------- apply: combine holo + anti sectors ----------
  def apply(*, ae: jnp.ndarray, r_ae: jnp.ndarray, r_ee: jnp.ndarray,
            gain: jnp.ndarray,
            zeros_holo_unconstrained: jnp.ndarray,
            zeros_anti_unconstrained: jnp.ndarray) -> jnp.ndarray:
    """ae: (nelectron, natom, 2) -> returns (nelectron, output_dim)."""
    del r_ae, r_ee

    # positions (ne,2)
    pos = ae[:, 0, :].astype(jnp.float32)
    z = z_scale * (pos[..., 0] + 1j * pos[..., 1]) / jnp.sqrt(2.0)
    z = z.astype(jnp.complex64)  # (ne,)
    ne = z.shape[0]
    odim = gain.shape[0]

    # --- center zeros separately in each sector/orbital ---

    # holo
    if N_holo > 0:
      mean_h = jnp.mean(zeros_holo_unconstrained, axis=1, keepdims=True)
      zeros_h_centered = zeros_holo_unconstrained - mean_h
      zeros_h_c = (zeros_h_centered[..., 0] + 1j * zeros_h_centered[..., 1]).astype(jnp.complex64)
    else:
      zeros_h_c = jnp.zeros((odim, 0), dtype=jnp.complex64)

    # anti
    if N_anti > 0:
      mean_a = jnp.mean(zeros_anti_unconstrained, axis=1, keepdims=True)
      zeros_a_centered = zeros_anti_unconstrained - mean_a
      zeros_a_c = (zeros_a_centered[..., 0] + 1j * zeros_a_centered[..., 1]).astype(jnp.complex64)
    else:
      zeros_a_c = jnp.zeros((odim, 0), dtype=jnp.complex64)

    # --- log envelopes per orbital ---

    def log_env_holo_one_orbital(zeros_k: jnp.ndarray) -> jnp.ndarray:
      """Holomorphic sector for one orbital: (ne,) complex."""
      if zeros_k.shape[0] == 0 or N_holo == 0:
        return jnp.zeros((ne,), dtype=jnp.complex64)
      return jax.vmap(lambda z_e: ellipticfunctions._LLL_with_zeros_log_cached(z_e, zeros_k, consts_holo))(z)

    def log_env_anti_one_orbital(zeros_k: jnp.ndarray) -> jnp.ndarray:
      """Anti-holomorphic sector for one orbital: (ne,) complex."""
      if zeros_k.shape[0] == 0 or N_anti == 0:
        return jnp.zeros((ne,), dtype=jnp.complex64)

      def one_e(z_e):
        # treat conj(z) as holomorphic variable and then conjugate the log
        val = ellipticfunctions._LLL_with_zeros_log_cached(jnp.conj(z_e), zeros_k, consts_anti) ###check this carefuly
        return val

      return jax.vmap(one_e)(z)

    # vectorize over orbitals -> (odim, ne)
    log_env_holo_od = jax.vmap(log_env_holo_one_orbital, in_axes=0)(zeros_h_c)  # (odim, ne)
    log_env_anti_od = jax.vmap(log_env_anti_one_orbital, in_axes=0)(zeros_a_c)  # (odim, ne)

    # swap to (ne, odim)
    log_env_holo_eo = jnp.swapaxes(log_env_holo_od, 0, 1)  # (ne, odim)
    log_env_anti_eo = jnp.swapaxes(log_env_anti_od, 0, 1)  # (ne, odim)

    # total
    log_env_eo = log_env_holo_eo + log_env_anti_eo  # (ne, odim)

    env_eo = jnp.exp(log_env_eo).astype(jnp.complex64)
    gain_c = gain.astype(env_eo.dtype)
    env_eo = env_eo * gain_c[None, :]

    return env_eo

  return envelopes.Envelope(envelopes.EnvelopeType.PRE_DETERMINANT, init, apply)



def make_LLL_envelope_2d_trainable_zeros_mixed3(
    lattice: jnp.ndarray,
    elliptic_log_sigma=None,   # kept for API compatibility
    z_scale: float = 1.0,
    magfield_kwargs=None,
    use_kfac_generic: bool = True,  # NEW: tell KFAC to see these params
) -> envelopes.Envelope:
  """
  LLL-like envelope with per-orbital trainable holomorphic + anti-holomorphic zeros.

  - Holomorphic sector: ψ_holo ~ LLL_with_zeros_log_cached(z ; zeros_h, N_holo)
  - Anti-holo sector:   ψ_anti ~ conj( LLL_with_zeros_log_cached(conj(z) ; zeros_a, N_anti) )

  The net winding is controlled by:
      N_holo - N_anti = N_phi   (asserted)

  All geometric/theta constants are precomputed once and *not* part of params.
  Trainable params:
    - zeros_holo_unconstrained (per orbital)
    - zeros_anti_unconstrained (per orbital)
    - gain (per orbital)
  """
  if magfield_kwargs is None:
    magfield_kwargs = {}

  # --- flux bookkeeping ---
  N_phi  = int(magfield_kwargs["N_phi"])
  N_holo = int(magfield_kwargs.get("N_holo", N_phi))
  N_anti = int(magfield_kwargs.get("N_anti", 0))

  # Your corrected condition:
  assert N_holo - N_anti == N_phi, (
      f"Require N_holo - N_anti == N_phi, got N_holo={N_holo}, "
      f"N_anti={N_anti}, N_phi={N_phi}"
  )

  # --- geometry constants (shared) ---
  L1 = lattice[:, 0].astype(jnp.float32)
  L2 = lattice[:, 1].astype(jnp.float32)

  L1com = ellipticfunctions.to_cplx_divsqrt2(L1)
  L2com = ellipticfunctions.to_cplx_divsqrt2(L2)
  w1 = (L1com / 2.0).astype(jnp.complex64)
  w2 = (L2com / 2.0).astype(jnp.complex64)
  tau = (w2 / w1).astype(jnp.complex64)
  pi_c = jnp.asarray(jnp.pi, jnp.float32).astype(jnp.complex64)

  # Precompute θ₁ series coefficients once
  theta_coeffs = ellipticfunctions._precompute_theta_coeffs(tau, max_terms=15)
  t1p0, t1ppp0 = ellipticfunctions._theta_derivs0_from_coeffs(theta_coeffs)
  c = - (pi_c * pi_c) / (24.0 * w1 * w1) * (t1ppp0 / t1p0)

  # "almost" term, shared for both sectors
  almost = ellipticfunctions.almost_modular(L1, L2, magfield_kwargs["N_phi"])

  # consts for holo and anti sectors (differ only by N_phi)
  consts_holo = {
      'L1': L1,
      'L2': L2,
      'w1': w1,
      'w2': w2,
      'tau': tau,
      'pi': pi_c,
      'theta_coeffs': theta_coeffs,
      't1p0': t1p0,
      't1ppp0': t1ppp0,
      'c': c,
      'N_phi': jnp.asarray(N_holo, jnp.float32),
      'almost': almost,
  }
  consts_anti = {
      'L1': L1,
      'L2': L2,
      'w1': w1,
      'w2': w2,
      'tau': tau,
      'pi': pi_c,
      'theta_coeffs': theta_coeffs,
      't1p0': t1p0,
      't1ppp0': t1ppp0,
      'c': c,
      'N_phi': jnp.asarray(max(N_anti, 1), jnp.float32),  # avoid div-by-zero; unused if N_anti==0
      'almost': almost,
  }

  consts_holo = jax.tree.map(lambda x: jax.lax.stop_gradient(jnp.asarray(x)), consts_holo)
  consts_anti = jax.tree.map(lambda x: jax.lax.stop_gradient(jnp.asarray(x)), consts_anti)

  # ---------- init: sample zeros per orbital in each sector ----------
  def init(natom: int, output_dims: Sequence[int], ndim: int = 2
          ) -> Sequence[Mapping[str, jnp.ndarray]]:
    del natom, ndim

    params = []
    key = jax.random.PRNGKey(95)

    for odim in output_dims:
      key, subkey_h = jax.random.split(key)
      key, subkey_a = jax.random.split(key)

      # --- holomorphic zeros ---
      if N_holo > 0:
        uv_h = jax.random.uniform(subkey_h, (odim, N_holo, 2), dtype=jnp.float32)
        zeros_h_init = (uv_h[..., 0:1] * L1[None, None, :] +
                        uv_h[..., 1:2] * L2[None, None, :])
        zeros_h_mean = jnp.mean(zeros_h_init, axis=1, keepdims=True)
        zeros_h_centered = zeros_h_init - zeros_h_mean
      else:
        zeros_h_centered = jnp.zeros((odim, 0, 2), dtype=jnp.float32)

      # --- anti-holomorphic zeros ---
      if N_anti > 0:
        uv_a = jax.random.uniform(subkey_a, (odim, N_anti, 2), dtype=jnp.float32)
        zeros_a_init = (uv_a[..., 0:1] * L1[None, None, :] +
                        uv_a[..., 1:2] * L2[None, None, :])
        zeros_a_mean = jnp.mean(zeros_a_init, axis=1, keepdims=True)
        zeros_a_centered = zeros_a_init - zeros_a_mean
      else:
        zeros_a_centered = jnp.zeros((odim, 0, 2), dtype=jnp.float32)

      params.append({
          'gain': jnp.ones((odim,), dtype=jnp.float32),
          'zeros_holo_unconstrained': zeros_h_centered.astype(jnp.float32),
          'zeros_anti_unconstrained': zeros_a_centered.astype(jnp.float32),
      })

    return params

  # ---------- apply: combine holo + anti sectors ----------
  def apply(*, ae: jnp.ndarray, r_ae: jnp.ndarray, r_ee: jnp.ndarray,
            gain: jnp.ndarray,
            zeros_holo_unconstrained: jnp.ndarray,
            zeros_anti_unconstrained: jnp.ndarray) -> jnp.ndarray:
    """ae: (nelectron, natom, 2) -> returns (nelectron, output_dim)."""
    del r_ae, r_ee

    # --- KFAC registration for envelope params ---
    if use_kfac_generic:
      zeros_holo_unconstrained = kfac_jax.register_generic(zeros_holo_unconstrained)
      zeros_anti_unconstrained = kfac_jax.register_generic(zeros_anti_unconstrained)
      # If you also want gain preconditioned separately, uncomment:
      # gain = kfac_jax.register_generic(gain)

    # positions (ne,2)
    pos = ae[:, 0, :].astype(jnp.float32)
    z = z_scale * (pos[..., 0] + 1j * pos[..., 1]) / jnp.sqrt(2.0)
    z = z.astype(jnp.complex64)  # (ne,)
    ne = z.shape[0]
    odim = gain.shape[0]

    # --- center zeros separately in each sector/orbital ---

    # holo
    if N_holo > 0:
      mean_h = jnp.mean(zeros_holo_unconstrained, axis=1, keepdims=True)
      zeros_h_centered = zeros_holo_unconstrained - mean_h
      zeros_h_c = (zeros_h_centered[..., 0] + 1j * zeros_h_centered[..., 1]).astype(jnp.complex64)
    else:
      zeros_h_c = jnp.zeros((odim, 0), dtype=jnp.complex64)

    # anti
    if N_anti > 0:
      mean_a = jnp.mean(zeros_anti_unconstrained, axis=1, keepdims=True)
      zeros_a_centered = zeros_anti_unconstrained - mean_a
      zeros_a_c = (zeros_a_centered[..., 0] + 1j * zeros_a_centered[..., 1]).astype(jnp.complex64)
    else:
      zeros_a_c = jnp.zeros((odim, 0), dtype=jnp.complex64)

    # --- log envelopes per orbital ---

    def log_env_holo_one_orbital(zeros_k: jnp.ndarray) -> jnp.ndarray:
      """Holomorphic sector for one orbital: (ne,) complex."""
      if zeros_k.shape[0] == 0 or N_holo == 0:
        return jnp.zeros((ne,), dtype=jnp.complex64)
      return jax.vmap(
          lambda z_e: ellipticfunctions._LLL_with_zeros_log_cached(z_e, zeros_k, consts_holo)
      )(z)

    def log_env_anti_one_orbital(zeros_k: jnp.ndarray) -> jnp.ndarray:
      """Anti-holomorphic sector for one orbital: (ne,) complex."""
      if zeros_k.shape[0] == 0 or N_anti == 0:
        return jnp.zeros((ne,), dtype=jnp.complex64)

      def one_e(z_e):
        # treat conj(z) as holomorphic variable and then conjugate the log
        val = ellipticfunctions._LLL_with_zeros_log_cached(jnp.conj(z_e), zeros_k, consts_anti)
        return jnp.conj(val)

      return jax.vmap(one_e)(z)

    # vectorize over orbitals -> (odim, ne)
    log_env_holo_od = jax.vmap(log_env_holo_one_orbital, in_axes=0)(zeros_h_c)  # (odim, ne)
    log_env_anti_od = jax.vmap(log_env_anti_one_orbital, in_axes=0)(zeros_a_c)  # (odim, ne)

    # swap to (ne, odim)
    log_env_holo_eo = jnp.swapaxes(log_env_holo_od, 0, 1)  # (ne, odim)
    log_env_anti_eo = jnp.swapaxes(log_env_anti_od, 0, 1)  # (ne, odim)

    # total
    log_env_eo = log_env_holo_eo + log_env_anti_eo  # (ne, odim)

    env_eo = jnp.exp(log_env_eo).astype(jnp.complex64)
    gain_c = gain.astype(env_eo.dtype)
    env_eo = env_eo * gain_c[None, :]

    return env_eo

  return envelopes.Envelope(envelopes.EnvelopeType.PRE_DETERMINANT, init, apply)



# def make_LLL_envelope_2d_trainable_zeros_mixed4(
#     lattice: jnp.ndarray,
#     elliptic_log_sigma=None,   # kept for API compatibility
#     z_scale: float = 1.0,
#     magfield_kwargs=None,
#     use_kfac_dense: bool = False,
#     kfac_dense_eps: float = 1e-4,
# ) -> envelopes.Envelope:
#   """
#   LLL-like envelope with per-orbital trainable holomorphic + anti-holomorphic zeros.

#   - Holomorphic sector: ψ_holo ~ LLL_with_zeros_log_cached(z ; zeros_h, N_holo)
#   - Anti-holo sector:   ψ_anti ~ conj( LLL_with_zeros_log_cached(conj(z) ; zeros_a, N_anti) )

#   The net winding is controlled by:
#       N_holo - N_anti = N_phi   (asserted)

#   Trainable params per spin channel:
#     - zeros_holo_unconstrained : (N_holo*2, odim)
#     - zeros_anti_unconstrained : (N_anti*2, odim)
#     - gain                     : (odim,)

#   We:
#     * Reshape these weights back into zero coordinates (odim, N_holo, 2) for physics,
#       then center them by subtracting the per-orbital mean.
#     * Also tag them as a dense layer for KFAC with `register_dense`, using a tiny
#       coupling (kfac_dense_eps) into log_env_eo so their Fisher block is nonzero.
#   """
#   if magfield_kwargs is None:
#     magfield_kwargs = {}

#   # --- flux bookkeeping ---
#   N_phi  = int(magfield_kwargs["N_phi"])
#   N_holo = int(magfield_kwargs.get("N_holo", N_phi))
#   N_anti = int(magfield_kwargs.get("N_anti", 0))

#   assert N_holo - N_anti == N_phi, (
#       f"Require N_holo - N_anti == N_phi, got N_holo={N_holo}, "
#       f"N_anti={N_anti}, N_phi={N_phi}"
#   )

#   # --- geometry constants (shared) ---
#   L1 = lattice[:, 0].astype(jnp.float32)
#   L2 = lattice[:, 1].astype(jnp.float32)

#   L1com = ellipticfunctions.to_cplx_divsqrt2(L1)
#   L2com = ellipticfunctions.to_cplx_divsqrt2(L2)
#   w1 = (L1com / 2.0).astype(jnp.complex64)
#   w2 = (L2com / 2.0).astype(jnp.complex64)
#   tau = (w2 / w1).astype(jnp.complex64)
#   pi_c = jnp.asarray(jnp.pi, jnp.float32).astype(jnp.complex64)

#   # Precompute θ₁ series coefficients once
#   theta_coeffs = ellipticfunctions._precompute_theta_coeffs(tau, max_terms=15)
#   t1p0, t1ppp0 = ellipticfunctions._theta_derivs0_from_coeffs(theta_coeffs)
#   c = - (pi_c * pi_c) / (24.0 * w1 * w1) * (t1ppp0 / t1p0)

#   # "almost" term, shared for both sectors
#   almost = ellipticfunctions.almost_modular(L1, L2, magfield_kwargs["N_phi"])

#   # consts for holo and anti sectors (differ only by N_phi)
#   consts_holo = {
#       'L1': L1,
#       'L2': L2,
#       'w1': w1,
#       'w2': w2,
#       'tau': tau,
#       'pi': pi_c,
#       'theta_coeffs': theta_coeffs,
#       't1p0': t1p0,
#       't1ppp0': t1ppp0,
#       'c': c,
#       'N_phi': jnp.asarray(N_holo, jnp.float32),
#       'almost': almost,
#   }
#   consts_anti = {
#       'L1': L1,
#       'L2': L2,
#       'w1': w1,
#       'w2': w2,
#       'tau': tau,
#       'pi': pi_c,
#       'theta_coeffs': theta_coeffs,
#       't1p0': t1p0,
#       't1ppp0': t1ppp0,
#       'c': c,
#       'N_phi': jnp.asarray(max(N_anti, 1), jnp.float32),  # avoid div-by-zero if N_anti==0
#       'almost': almost,
#   }

#   consts_holo = jax.tree.map(lambda x: jax.lax.stop_gradient(jnp.asarray(x)), consts_holo)
#   consts_anti = jax.tree.map(lambda x: jax.lax.stop_gradient(jnp.asarray(x)), consts_anti)

#   # ---------- init: sample zeros per orbital in each sector ----------
#   def init(
#       natom: int,
#       output_dims: Sequence[int],
#       ndim: int = 2,
#   ) -> Sequence[Mapping[str, jnp.ndarray]]:
#     del natom, ndim

#     params = []
#     key = jax.random.PRNGKey(95)

#     for odim in output_dims:
#       key, subkey_h = jax.random.split(key)
#       key, subkey_a = jax.random.split(key)

#       # Holomorphic zeros: store as weight matrix (N_holo*2, odim)
#       if N_holo > 0:
#         # sample positions in the supercell first, then encode into weights
#         uv_h = jax.random.uniform(subkey_h, (odim, N_holo, 2), dtype=jnp.float32)
#         zeros_h_init = (uv_h[..., 0:1] * L1[None, None, :] +
#                         uv_h[..., 1:2] * L2[None, None, :])  # (odim, N_holo, 2)
#         zeros_h_mean = jnp.mean(zeros_h_init, axis=1, keepdims=True)
#         zeros_h_centered = zeros_h_init - zeros_h_mean          # (odim, N_holo, 2)
#         # pack into (N_holo*2, odim) as "weight"
#         zeros_h_weight = jnp.transpose(zeros_h_centered, (1, 2, 0))  # (N_holo, 2, odim)
#         zeros_h_weight = zeros_h_weight.reshape(N_holo * 2, odim)    # (N_holo*2, odim)
#       else:
#         zeros_h_weight = jnp.zeros((0, odim), dtype=jnp.float32)

#       # Anti-holomorphic zeros: also weight matrix (N_anti*2, odim)
#       if N_anti > 0:
#         uv_a = jax.random.uniform(subkey_a, (odim, N_anti, 2), dtype=jnp.float32)
#         zeros_a_init = (uv_a[..., 0:1] * L1[None, None, :] +
#                         uv_a[..., 1:2] * L2[None, None, :])  # (odim, N_anti, 2)
#         zeros_a_mean = jnp.mean(zeros_a_init, axis=1, keepdims=True)
#         zeros_a_centered = zeros_a_init - zeros_a_mean         # (odim, N_anti, 2)
#         zeros_a_weight = jnp.transpose(zeros_a_centered, (1, 2, 0))  # (N_anti, 2, odim)
#         zeros_a_weight = zeros_a_weight.reshape(N_anti * 2, odim)    # (N_anti*2, odim)
#       else:
#         zeros_a_weight = jnp.zeros((0, odim), dtype=jnp.float32)

#       params.append({
#           'gain': jnp.ones((odim,), dtype=jnp.float32),
#           'zeros_holo_unconstrained': zeros_h_weight.astype(jnp.float32),
#           'zeros_anti_unconstrained': zeros_a_weight.astype(jnp.float32),
#       })

#     return params

#   # ---------- apply: combine holo + anti sectors ----------
#   def apply(
#       *,
#       ae: jnp.ndarray,
#       r_ae: jnp.ndarray,
#       r_ee: jnp.ndarray,
#       gain: jnp.ndarray,
#       zeros_holo_unconstrained: jnp.ndarray,
#       zeros_anti_unconstrained: jnp.ndarray,
#   ) -> jnp.ndarray:
#     """ae: (nelectron, natom, 2) -> returns (nelectron, output_dim)."""
#     del r_ae, r_ee

#     # positions (ne,2)
#     pos = ae[:, 0, :].astype(jnp.float32)
#     z = z_scale * (pos[..., 0] + 1j * pos[..., 1]) / jnp.sqrt(2.0)
#     z = z.astype(jnp.complex64)  # (ne,)
#     ne = z.shape[0]
#     odim = gain.shape[0]

#     # --- decode weight matrices back into zeros and center them ---

#     # holomorphic sector
#     if N_holo > 0 and zeros_holo_unconstrained.size > 0:
#       # zeros_holo_unconstrained: (N_holo*2, odim)
#       assert zeros_holo_unconstrained.shape == (N_holo * 2, odim)
#       zh = zeros_holo_unconstrained.reshape(N_holo, 2, odim)     # (N_holo, 2, odim)
#       zh = jnp.transpose(zh, (2, 0, 1))                          # (odim, N_holo, 2)
#       mean_h = jnp.mean(zh, axis=1, keepdims=True)               # (odim, 1, 2)
#       zeros_h_centered = zh - mean_h                             # (odim, N_holo, 2)
#       zeros_h_c = (zeros_h_centered[..., 0] +
#                    1j * zeros_h_centered[..., 1]).astype(jnp.complex64)  # (odim, N_holo)
#     else:
#       zeros_h_c = jnp.zeros((odim, 0), dtype=jnp.complex64)

#     # anti-holomorphic sector
#     if N_anti > 0 and zeros_anti_unconstrained.size > 0:
#       assert zeros_anti_unconstrained.shape == (N_anti * 2, odim)
#       za = zeros_anti_unconstrained.reshape(N_anti, 2, odim)     # (N_anti, 2, odim)
#       za = jnp.transpose(za, (2, 0, 1))                          # (odim, N_anti, 2)
#       mean_a = jnp.mean(za, axis=1, keepdims=True)               # (odim, 1, 2)
#       zeros_a_centered = za - mean_a                             # (odim, N_anti, 2)
#       zeros_a_c = (zeros_a_centered[..., 0] +
#                    1j * zeros_a_centered[..., 1]).astype(jnp.complex64)  # (odim, N_anti)
#     else:
#       zeros_a_c = jnp.zeros((odim, 0), dtype=jnp.complex64)

#     # --- log envelopes per orbital ---

#     def log_env_holo_one_orbital(zeros_k: jnp.ndarray) -> jnp.ndarray:
#       """Holomorphic sector for one orbital: (ne,) complex."""
#       if zeros_k.shape[0] == 0 or N_holo == 0:
#         return jnp.zeros((ne,), dtype=jnp.complex64)
#       return jax.vmap(
#           lambda z_e: ellipticfunctions._LLL_with_zeros_log_cached(
#               z_e, zeros_k, consts_holo
#           )
#       )(z)

#     def log_env_anti_one_orbital(zeros_k: jnp.ndarray) -> jnp.ndarray:
#       """Anti-holomorphic sector for one orbital: (ne,) complex."""
#       if zeros_k.shape[0] == 0 or N_anti == 0:
#         return jnp.zeros((ne,), dtype=jnp.complex64)

#       def one_e(z_e):
#         # treat conj(z) as holomorphic variable and then conjugate the log
#         val = ellipticfunctions._LLL_with_zeros_log_cached(
#             jnp.conj(z_e), zeros_k, consts_anti
#         )
#         return jnp.conj(val)

#       return jax.vmap(one_e)(z)

#     # vectorize over orbitals -> (odim, ne)
#     log_env_holo_od = jax.vmap(log_env_holo_one_orbital, in_axes=0)(zeros_h_c)  # (odim, ne)
#     log_env_anti_od = jax.vmap(log_env_anti_one_orbital, in_axes=0)(zeros_a_c)  # (odim, ne)

#     # swap to (ne, odim)
#     log_env_holo_eo = jnp.swapaxes(log_env_holo_od, 0, 1)  # (ne, odim)
#     log_env_anti_eo = jnp.swapaxes(log_env_anti_od, 0, 1)  # (ne, odim)

#     # total log envelope
#     log_env_eo = log_env_holo_eo + log_env_anti_eo  # (ne, odim)

#     # --- KFAC dense tagging: treat zeros as weights of a (fake) dense layer ---
#     # --- log_env_eo is (ne, odim) at this point ---

#     # --- KFAC dense tagging: treat zeros as weights of a (fake) repeated dense layer ---
#     if use_kfac_dense:
#       # Holomorphic zeros
#       if N_holo > 0 and zeros_holo_unconstrained.size > 0:
#         in_dim_h = N_holo * 2

#         # x_h: (ne, in_dim_h); after outer vmap this becomes (batch, ne, in_dim_h)
#         x_h = jnp.ones((z.shape[0], in_dim_h), dtype=jnp.float32)

#         # W_h: (in_dim_h, odim) – this is the actual parameter leaf
#         W_h = zeros_holo_unconstrained

#         # pre_h: (ne, odim); after vmap -> (batch, ne, odim)
#         pre_h = x_h @ W_h

#         # Register as a repeated-dense layer
#         pre_h = kfac_jax.register_dense(
#             pre_h,
#             x_h,
#             W_h,
#             variant="repeated_dense",
#             type="full",
#         )

#         # Tiny coupling into log_env_eo with matching shape (ne, odim)
#         shift_h = kfac_dense_eps * jnp.real(pre_h)   # (ne, odim)
#         log_env_eo = log_env_eo + shift_h

#       # Anti-holomorphic zeros
#       if N_anti > 0 and zeros_anti_unconstrained.size > 0:
#         in_dim_a = N_anti * 2
#         x_a = jnp.ones((z.shape[0], in_dim_a), dtype=jnp.float32)
#         W_a = zeros_anti_unconstrained
#         pre_a = x_a @ W_a  # (ne, odim)
#         pre_a = kfac_jax.register_dense(
#             pre_a,
#             x_a,
#             W_a,
#             variant="repeated_dense",
#             type="full",
#         )
#         shift_a = kfac_dense_eps * jnp.real(pre_a)
#         log_env_eo = log_env_eo + shift_a

#     # --- exponentiate and apply gain ---
#     env_eo = jnp.exp(log_env_eo).astype(jnp.complex64)
#     gain_c = gain.astype(env_eo.dtype)
#     env_eo = env_eo * gain_c[None, :]

#     return env_eo

#   return envelopes.Envelope(envelopes.EnvelopeType.PRE_DETERMINANT, init, apply)



def map_to_supercell(positions: jnp.ndarray,
                     lattice_vectors: jnp.ndarray) -> jnp.ndarray:
    """
    Maps positions back to the supercell for a general lattice (e.g., triangular).

    - positions: array of shape (..., 2) or flat length 2*N
    - lattice_vectors: 2x2 array whose columns are the primitive lattice vectors
    - returns: positions mapped back into the unit cell, with the same shape as input

    Fractional coordinates are wrapped into [0,1) along each lattice direction.
    """
    positions = jnp.asarray(positions)
    lattice_vectors = jnp.asarray(lattice_vectors)

    orig_shape = positions.shape
    flat = positions.reshape(-1)                      # (2 * N,)
    num_particles = flat.shape[0] // 2
    reshaped_positions = flat.reshape((num_particles, 2))  # (N, 2)

    # fractional coordinates
    lattice_inverse = jnp.linalg.inv(lattice_vectors)      # (2, 2)
    frac = reshaped_positions @ lattice_inverse.T          # (N, 2)

    # wrap to [0, 1): x -> x - floor(x)
    frac_wrapped = frac - jnp.floor(frac)

    # back to Cartesian
    supercell_positions = frac_wrapped @ lattice_vectors.T  # (N, 2)

    return supercell_positions.reshape(orig_shape)


def make_LLL_envelope_2d_trainable_zeros_mixed4(
    lattice: jnp.ndarray,
    elliptic_log_sigma=None,   # kept for API compatibility
    z_scale: float = 1.0,
    magfield_kwargs=None,
    use_kfac_dense: bool = False,
    kfac_dense_eps: float = 1e-4,
) -> "envelopes.Envelope":
  """
  LLL-like envelope with per-orbital trainable holomorphic + anti-holomorphic zeros.

  - Holomorphic sector: ψ_holo ~ LLL_with_zeros_log_cached(z ; zeros_h, N_holo)
  - Anti-holo sector:   ψ_anti ~ conj( LLL_with_zeros_log_cached(conj(z) ; zeros_a, N_anti) )

  The net winding is controlled by:
      N_holo - N_anti = N_phi   (asserted)

  Trainable params per spin channel:
    - zeros_holo_unconstrained : (N_holo*2, odim)
    - zeros_anti_unconstrained : (N_anti*2, odim)
    - gain                     : (odim,)

  We:
    * Reshape these weights back into zero coordinates (odim, N_holo, 2) for physics,
      then center them by subtracting the per-orbital mean (after mapping to supercell).
    * Also tag them as a dense layer for KFAC with `register_dense`, using a tiny
      coupling (kfac_dense_eps) into log_env_eo so their Fisher block is nonzero.
  """
  if magfield_kwargs is None:
    magfield_kwargs = {}

  # --- flux bookkeeping ---
  N_phi  = int(magfield_kwargs["N_phi"])
  N_holo = int(magfield_kwargs.get("N_holo", N_phi))
  N_anti = int(magfield_kwargs.get("N_anti", 0))

  assert N_holo - N_anti == N_phi, (
      f"Require N_holo - N_anti == N_phi, got N_holo={N_holo}, "
      f"N_anti={N_anti}, N_phi={N_phi}"
  )

  # --- geometry constants (shared) ---
  L1 = lattice[:, 0].astype(jnp.float32)
  L2 = lattice[:, 1].astype(jnp.float32)

  L1com = ellipticfunctions.to_cplx_divsqrt2(L1)
  L2com = ellipticfunctions.to_cplx_divsqrt2(L2)
  w1 = (L1com / 2.0).astype(jnp.complex64)
  w2 = (L2com / 2.0).astype(jnp.complex64)
  tau = (w2 / w1).astype(jnp.complex64)
  pi_c = jnp.asarray(jnp.pi, jnp.float32).astype(jnp.complex64)

  # Precompute θ₁ series coefficients once
  theta_coeffs = ellipticfunctions._precompute_theta_coeffs(tau, max_terms=15)
  t1p0, t1ppp0 = ellipticfunctions._theta_derivs0_from_coeffs(theta_coeffs)
  c = - (pi_c * pi_c) / (24.0 * w1 * w1) * (t1ppp0 / t1p0)

  # "almost" term, shared for both sectors
  almost = ellipticfunctions.almost_modular(L1, L2, magfield_kwargs["N_phi"])

  # consts for holo and anti sectors (differ only by N_phi)
  consts_holo = {
      'L1': L1,
      'L2': L2,
      'w1': w1,
      'w2': w2,
      'tau': tau,
      'pi': pi_c,
      'theta_coeffs': theta_coeffs,
      't1p0': t1p0,
      't1ppp0': t1ppp0,
      'c': c,
      'N_phi': jnp.asarray(N_holo, jnp.float32),
      'almost': almost,
  }
  consts_anti = {
      'L1': L1,
      'L2': L2,
      'w1': w1,
      'w2': w2,
      'tau': tau,
      'pi': pi_c,
      'theta_coeffs': theta_coeffs,
      't1p0': t1p0,
      't1ppp0': t1ppp0,
      'c': c,
      'N_phi': jnp.asarray(max(N_anti, 1), jnp.float32),  # avoid div-by-zero if N_anti==0
      'almost': almost,
  }

  consts_holo = jax.tree.map(lambda x: jax.lax.stop_gradient(jnp.asarray(x)), consts_holo)
  consts_anti = jax.tree.map(lambda x: jax.lax.stop_gradient(jnp.asarray(x)), consts_anti)

  # ---------- init: sample zeros per orbital in each sector ----------
  def init(
      natom: int,
      output_dims: Sequence[int],
      ndim: int = 2,
  ) -> Sequence[Mapping[str, jnp.ndarray]]:
    del natom, ndim

    params = []
    key = jax.random.PRNGKey(1234)

    for odim in output_dims:
      key, subkey_h = jax.random.split(key)
      key, subkey_a = jax.random.split(key)

      # Holomorphic zeros: store as weight matrix (N_holo*2, odim)
      if N_holo > 0:
        # sample positions in the supercell first, then encode into weights
        uv_h = jax.random.uniform(subkey_h, (odim, N_holo, 2), dtype=jnp.float32)
        zeros_h_init = (uv_h[..., 0:1] * L1[None, None, :] +
                        uv_h[..., 1:2] * L2[None, None, :])  # (odim, N_holo, 2)
        zeros_h_mean = jnp.mean(zeros_h_init, axis=1, keepdims=True)
        zeros_h_centered = zeros_h_init - zeros_h_mean          # (odim, N_holo, 2)
        # pack into (N_holo*2, odim) as "weight"
        zeros_h_weight = jnp.transpose(zeros_h_centered, (1, 2, 0))  # (N_holo, 2, odim)
        zeros_h_weight = zeros_h_weight.reshape(N_holo * 2, odim)    # (N_holo*2, odim)
      else:
        zeros_h_weight = jnp.zeros((0, odim), dtype=jnp.float32)

      # Anti-holomorphic zeros: also weight matrix (N_anti*2, odim)
      if N_anti > 0:
        uv_a = jax.random.uniform(subkey_a, (odim, N_anti, 2), dtype=jnp.float32)
        zeros_a_init = (uv_a[..., 0:1] * L1[None, None, :] +
                        uv_a[..., 1:2] * L2[None, None, :])  # (odim, N_anti, 2)
        zeros_a_mean = jnp.mean(zeros_a_init, axis=1, keepdims=True)
        zeros_a_centered = zeros_a_init - zeros_a_mean         # (odim, N_anti, 2)
        zeros_a_weight = jnp.transpose(zeros_a_centered, (1, 2, 0))  # (N_anti, 2, odim)
        zeros_a_weight = zeros_a_weight.reshape(N_anti * 2, odim)    # (N_anti*2, odim)
      else:
        zeros_a_weight = jnp.zeros((0, odim), dtype=jnp.float32)

      params.append({
          'gain': jnp.ones((odim,), dtype=jnp.float32),
          'zeros_holo_unconstrained': zeros_h_weight.astype(jnp.float32),
          'zeros_anti_unconstrained': zeros_a_weight.astype(jnp.float32),
      })

    return params

  # ---------- apply: combine holo + anti sectors ----------
  def apply(
      *,
      ae: jnp.ndarray,
      r_ae: jnp.ndarray,
      r_ee: jnp.ndarray,
      gain: jnp.ndarray,
      zeros_holo_unconstrained: jnp.ndarray,
      zeros_anti_unconstrained: jnp.ndarray,
  ) -> jnp.ndarray:
    """ae: (nelectron, natom, 2) -> returns (nelectron, output_dim)."""
    del r_ae, r_ee

    # positions (ne,2)
    pos = ae[:, 0, :].astype(jnp.float32)
    z = z_scale * (pos[..., 0] + 1j * pos[..., 1]) / jnp.sqrt(2.0)
    z = z.astype(jnp.complex64)  # (ne,)
    ne = z.shape[0]
    odim = gain.shape[0]

    # --- decode weight matrices back into zeros and center them,
    #     AFTER mapping them to the supercell ---

    # holomorphic sector
    if N_holo > 0 and zeros_holo_unconstrained.size > 0:
      # zeros_holo_unconstrained: (N_holo*2, odim)
      assert zeros_holo_unconstrained.shape == (N_holo * 2, odim)
      zh = zeros_holo_unconstrained.reshape(N_holo, 2, odim)     # (N_holo, 2, odim)
      zh = jnp.transpose(zh, (2, 0, 1))                          # (odim, N_holo, 2)

      # map each orbital's zeros to the supercell before centering
      # zh_super = jax.vmap(
      #     lambda z_orb: map_to_supercell(z_orb, lattice)
      # )(zh)                                                      # (odim, N_holo, 2)

      mean_h = jnp.mean(zh, axis=1, keepdims=True)         # (odim, 1, 2)
      zeros_h_centered = zh - mean_h                       # (odim, N_holo, 2)
      zeros_h_c = (zeros_h_centered[..., 0] +
                   1j * zeros_h_centered[..., 1]).astype(jnp.complex64)  # (odim, N_holo)
    else:
      zeros_h_c = jnp.zeros((odim, 0), dtype=jnp.complex64)

    # anti-holomorphic sector
    if N_anti > 0 and zeros_anti_unconstrained.size > 0:
      assert zeros_anti_unconstrained.shape == (N_anti * 2, odim)
      za = zeros_anti_unconstrained.reshape(N_anti, 2, odim)     # (N_anti, 2, odim)
      za = jnp.transpose(za, (2, 0, 1))                          # (odim, N_anti, 2)

      # map each orbital's zeros to the supercell before centering
      # za_super = jax.vmap(
      #     lambda z_orb: map_to_supercell(z_orb, lattice)
      # )(za)                                                      # (odim, N_anti, 2)

      mean_a = jnp.mean(za, axis=1, keepdims=True)         # (odim, 1, 2)
      zeros_a_centered = za - mean_a                       # (odim, N_anti, 2)
      zeros_a_c = (zeros_a_centered[..., 0] +
                   1j * zeros_a_centered[..., 1]).astype(jnp.complex64)  # (odim, N_anti)
    else:
      zeros_a_c = jnp.zeros((odim, 0), dtype=jnp.complex64)

    # --- log envelopes per orbital ---

    def log_env_holo_one_orbital(zeros_k: jnp.ndarray) -> jnp.ndarray:
      """Holomorphic sector for one orbital: (ne,) complex."""
      if zeros_k.shape[0] == 0 or N_holo == 0:
        return jnp.zeros((ne,), dtype=jnp.complex64)
      return jax.vmap(
          lambda z_e: ellipticfunctions._LLL_with_zeros_log_cached(
              z_e, zeros_k, consts_holo
          )
      )(z)

    def log_env_anti_one_orbital(zeros_k: jnp.ndarray) -> jnp.ndarray:
      """Anti-holomorphic sector for one orbital: (ne,) complex."""
      if zeros_k.shape[0] == 0 or N_anti == 0:
        return jnp.zeros((ne,), dtype=jnp.complex64)

      def one_e(z_e):
        # treat conj(z) as holomorphic variable and then conjugate the log
        val = ellipticfunctions._LLL_with_zeros_log_cached(
            jnp.conj(z_e), zeros_k, consts_anti
        )
        return jnp.conj(val)

      return jax.vmap(one_e)(z)

    # vectorize over orbitals -> (odim, ne)
    log_env_holo_od = jax.vmap(log_env_holo_one_orbital, in_axes=0)(zeros_h_c)  # (odim, ne)
    log_env_anti_od = jax.vmap(log_env_anti_one_orbital, in_axes=0)(zeros_a_c)  # (odim, ne)

    # swap to (ne, odim)
    log_env_holo_eo = jnp.swapaxes(log_env_holo_od, 0, 1)  # (ne, odim)
    log_env_anti_eo = jnp.swapaxes(log_env_anti_od, 0, 1)  # (ne, odim)

    # total log envelope
    log_env_eo = log_env_holo_eo + log_env_anti_eo  # (ne, odim)

    # --- KFAC dense tagging: treat zeros as weights of a (fake) repeated dense layer ---
    if use_kfac_dense:
      # Holomorphic zeros
      if N_holo > 0 and zeros_holo_unconstrained.size > 0:
        in_dim_h = N_holo * 2

        # x_h: (ne, in_dim_h); after outer vmap this becomes (batch, ne, in_dim_h)
        x_h = jnp.ones((z.shape[0], in_dim_h), dtype=jnp.float32)

        # W_h: (in_dim_h, odim) – this is the actual parameter leaf
        W_h = zeros_holo_unconstrained

        # pre_h: (ne, odim); after vmap -> (batch, ne, odim)
        pre_h = x_h @ W_h

        # Register as a repeated-dense layer
        pre_h = kfac_jax.register_dense(
            pre_h,
            x_h,
            W_h,
            variant="repeated_dense",
            type="full",
        )

        # Tiny coupling into log_env_eo with matching shape (ne, odim)
        shift_h = kfac_dense_eps * jnp.real(pre_h)   # (ne, odim)
        log_env_eo = log_env_eo + shift_h

      # Anti-holomorphic zeros
      if N_anti > 0 and zeros_anti_unconstrained.size > 0:
        in_dim_a = N_anti * 2
        x_a = jnp.ones((z.shape[0], in_dim_a), dtype=jnp.float32)
        W_a = zeros_anti_unconstrained
        pre_a = x_a @ W_a  # (ne, odim)
        pre_a = kfac_jax.register_dense(
            pre_a,
            x_a,
            W_a,
            variant="repeated_dense",
            type="full",
        )
        shift_a = kfac_dense_eps * jnp.real(pre_a)
        log_env_eo = log_env_eo + shift_a

    # --- exponentiate and apply gain ---
    #log_env_eo = log_env_eo - (L1comm*(L2comm) - conj(L1comm)*conj(L2comm))/2
    env_eo = jnp.exp(log_env_eo).astype(jnp.complex64)
    gain_c = gain.astype(env_eo.dtype)
    #env_eo = env_eo * gain_c[None, :]

    return env_eo

  return envelopes.Envelope(envelopes.EnvelopeType.PRE_DETERMINANT, init, apply)

def make_vortexformer_envelope(
    lattice: jnp.ndarray,
    elliptic_log_sigma=None,   # kept for API compatibility
    z_scale: float = 1.0,
    magfield_kwargs=None,
    use_kfac_dense: bool = False,
    kfac_dense_eps: float = 1e-4,
) -> "envelopes.Envelope":
  """
  LLL-like envelope with per-orbital trainable holomorphic + anti-holomorphic zeros.
  """

  if magfield_kwargs is None:
    magfield_kwargs = {}

  # --- flux bookkeeping ---
  N_phi  = int(magfield_kwargs["N_phi"])
  N_holo = int(magfield_kwargs.get("N_holo", N_phi))
  N_anti = int(magfield_kwargs.get("N_anti", 0))

  assert N_holo - N_anti == N_phi, (
      f"Require N_holo - N_anti == N_phi, got N_holo={N_holo}, "
      f"N_anti={N_anti}, N_phi={N_phi}"
  )

  # --- geometry constants (shared) ---
  L1 = lattice[:, 0].astype(jnp.float32)
  L2 = lattice[:, 1].astype(jnp.float32)

  L1com = ellipticfunctions.to_cplx_divsqrt2(L1)
  L2com = ellipticfunctions.to_cplx_divsqrt2(L2)
  w1 = (L1com / 2.0).astype(jnp.complex64)
  w2 = (L2com / 2.0).astype(jnp.complex64)
  tau = (w2 / w1).astype(jnp.complex64)
  pi_c = jnp.asarray(jnp.pi, jnp.float32).astype(jnp.complex64)

  # Precompute θ₁ series coefficients once
  theta_coeffs = ellipticfunctions._precompute_theta_coeffs(tau, max_terms=15)
  t1p0, t1ppp0 = ellipticfunctions._theta_derivs0_from_coeffs(theta_coeffs)
  c = - (pi_c * pi_c) / (24.0 * w1 * w1) * (t1ppp0 / t1p0)

  # "almost" term, shared for both sectors
  almost = ellipticfunctions.almost_modular(L1, L2, magfield_kwargs["N_phi"])

  # consts for holo and anti sectors (differ only by N_phi)
  consts_holo = {
      'L1': L1,
      'L2': L2,
      'w1': w1,
      'w2': w2,
      'tau': tau,
      'pi': pi_c,
      'theta_coeffs': theta_coeffs,
      't1p0': t1p0,
      't1ppp0': t1ppp0,
      'c': c,
      'N_phi': jnp.asarray(N_holo, jnp.float32),
      'almost': almost,
  }
  consts_anti = {
      'L1': L1,
      'L2': L2,
      'w1': w1,
      'w2': w2,
      'tau': tau,
      'pi': pi_c,
      'theta_coeffs': theta_coeffs,
      't1p0': t1p0,
      't1ppp0': t1ppp0,
      'c': c,
      'N_phi': jnp.asarray(max(N_anti, 1), jnp.float32),  # avoid div-by-zero if N_anti==0
      'almost': almost,
  }

  consts_holo = jax.tree.map(lambda x: jax.lax.stop_gradient(jnp.asarray(x)), consts_holo)
  consts_anti = jax.tree.map(lambda x: jax.lax.stop_gradient(jnp.asarray(x)), consts_anti)

  # ---------- init ----------
  def init(
      natom: int,
      output_dims: Sequence[int],
      ndim: int = 2,
  ) -> Sequence[Mapping[str, jnp.ndarray]]:
    del natom, ndim

    params = []
    key = jax.random.PRNGKey(95)

    for odim in output_dims:
      key, subkey_h = jax.random.split(key)
      key, subkey_a = jax.random.split(key)

      # Holomorphic zeros: weight matrix (N_holo*2, odim)
      if N_holo > 0:
        uv_h = jax.random.uniform(subkey_h, (odim, N_holo, 2), dtype=jnp.float32)
        zeros_h_init = (uv_h[..., 0:1] * L1[None, None, :] +
                        uv_h[..., 1:2] * L2[None, None, :])  # (odim, N_holo, 2)
        zeros_h_mean = jnp.mean(zeros_h_init, axis=1, keepdims=True)
        zeros_h_centered = zeros_h_init - zeros_h_mean          # (odim, N_holo, 2)
        zeros_h_weight = jnp.transpose(zeros_h_centered, (1, 2, 0))  # (N_holo, 2, odim)
        zeros_h_weight = zeros_h_weight.reshape(N_holo * 2, odim)    # (N_holo*2, odim)
      else:
        zeros_h_weight = jnp.zeros((0, odim), dtype=jnp.float32)

      # Anti-holomorphic zeros: weight matrix (N_anti*2, odim)
      if N_anti > 0:
        uv_a = jax.random.uniform(subkey_a, (odim, N_anti, 2), dtype=jnp.float32)
        zeros_a_init = (uv_a[..., 0:1] * L1[None, None, :] +
                        uv_a[..., 1:2] * L2[None, None, :])  # (odim, N_anti, 2)
        zeros_a_mean = jnp.mean(zeros_a_init, axis=1, keepdims=True)
        zeros_a_centered = zeros_a_init - zeros_a_mean         # (odim, N_anti, 2)
        zeros_a_weight = jnp.transpose(zeros_a_centered, (1, 2, 0))  # (N_anti, 2, odim)
        zeros_a_weight = zeros_a_weight.reshape(N_anti * 2, odim)    # (N_anti*2, odim)
      else:
        zeros_a_weight = jnp.zeros((0, odim), dtype=jnp.float32)

      params.append({
          'gain': jnp.ones((odim,), dtype=jnp.float32),
          'zeros_holo_unconstrained': zeros_h_weight.astype(jnp.float32),
          'zeros_anti_unconstrained': zeros_a_weight.astype(jnp.float32),
      })

    return params

  # ---------- apply ----------
  def apply(
      *,
      ae: jnp.ndarray,
      r_ae: jnp.ndarray,
      r_ee: jnp.ndarray,
      gain: jnp.ndarray,
      zeros_holo_unconstrained: jnp.ndarray,
      zeros_anti_unconstrained: jnp.ndarray,
  ) -> jnp.ndarray:
    """ae: (nelectron, natom, 2) -> returns (nelectron, output_dim)."""
    del r_ae, r_ee

    # positions (ne,2)
    pos = ae[:, 0, :].astype(jnp.float32)
    z = z_scale * (pos[..., 0] + 1j * pos[..., 1]) / jnp.sqrt(2.0)
    z = z.astype(jnp.complex64)  # (ne,)
    ne = z.shape[0]
    odim = gain.shape[0]

    # ---------- holomorphic sector ----------
    if N_holo > 0 and zeros_holo_unconstrained.size > 0:
      assert zeros_holo_unconstrained.shape == (N_holo * 2, odim)
      zh = zeros_holo_unconstrained.reshape(N_holo, 2, odim)  # (N_holo, 2, odim)
      zh = jnp.transpose(zh, (2, 0, 1))                        # (odim, N_holo, 2)

      # center in real space
      mean_h = jnp.mean(zh, axis=1, keepdims=True)             # (odim, 1, 2)
      zeros_h_centered = zh - mean_h                           # (odim, N_holo, 2)

      # project *centered* zeros back to the first supercell
      zeros_h_centered = map_to_supercell(zeros_h_centered, lattice)  # same shape

      zeros_h_c = (zeros_h_centered[..., 0] +
                   1j * zeros_h_centered[..., 1]).astype(jnp.complex64)  # (odim, N_holo)
    else:
      zeros_h_c = jnp.zeros((odim, 0), dtype=jnp.complex64)

    # ---------- anti-holomorphic sector ----------
    if N_anti > 0 and zeros_anti_unconstrained.size > 0:
      assert zeros_anti_unconstrained.shape == (N_anti * 2, odim)
      za = zeros_anti_unconstrained.reshape(N_anti, 2, odim)   # (N_anti, 2, odim)
      za = jnp.transpose(za, (2, 0, 1))                        # (odim, N_anti, 2)

      # center in real space
      mean_a = jnp.mean(za, axis=1, keepdims=True)             # (odim, 1, 2)
      zeros_a_centered = za - mean_a                           # (odim, N_anti, 2)

      # project *centered* zeros back to the first supercell
      zeros_a_centered = map_to_supercell(zeros_a_centered, lattice)

      zeros_a_c = (zeros_a_centered[..., 0] +
                   1j * zeros_a_centered[..., 1]).astype(jnp.complex64)  # (odim, N_anti)
    else:
      zeros_a_c = jnp.zeros((odim, 0), dtype=jnp.complex64)

    # --- log envelopes per orbital ---

    def log_env_holo_one_orbital(zeros_k: jnp.ndarray) -> jnp.ndarray:
      if zeros_k.shape[0] == 0 or N_holo == 0:
        return jnp.zeros((ne,), dtype=jnp.complex64)
      return jax.vmap(
          lambda z_e: ellipticfunctions._LLL_with_zeros_log_cached(
              z_e, zeros_k, consts_holo
          )
      )(z)

    def log_env_anti_one_orbital(zeros_k: jnp.ndarray) -> jnp.ndarray:
      if zeros_k.shape[0] == 0 or N_anti == 0:
        return jnp.zeros((ne,), dtype=jnp.complex64)

      def one_e(z_e):
        val = ellipticfunctions._LLL_with_zeros_log_cached(
            jnp.conj(z_e), zeros_k, consts_anti
        )
        return jnp.conj(val)

      return jax.vmap(one_e)(z)

    # vectorize over orbitals -> (odim, ne)
    log_env_holo_od = jax.vmap(log_env_holo_one_orbital, in_axes=0)(zeros_h_c)   # (odim, ne)
    log_env_anti_od = jax.vmap(log_env_anti_one_orbital, in_axes=0)(zeros_a_c)   # (odim, ne)

    # swap to (ne, odim)
    log_env_holo_eo = jnp.swapaxes(log_env_holo_od, 0, 1)  # (ne, odim)
    log_env_anti_eo = jnp.swapaxes(log_env_anti_od, 0, 1)  # (ne, odim)

    log_env_eo = log_env_holo_eo + log_env_anti_eo         # (ne, odim)

    # --- optional KFAC dense tagging (unchanged) ---
    if use_kfac_dense:
      if N_holo > 0 and zeros_holo_unconstrained.size > 0:
        in_dim_h = N_holo * 2
        x_h = jnp.ones((z.shape[0], in_dim_h), dtype=jnp.float32)
        W_h = zeros_holo_unconstrained
        pre_h = x_h @ W_h
        pre_h = kfac_jax.register_dense(
            pre_h,
            x_h,
            W_h,
            variant="repeated_dense",
            type="full",
        )
        shift_h = kfac_dense_eps * jnp.real(pre_h)
        log_env_eo = log_env_eo + shift_h

      if N_anti > 0 and zeros_anti_unconstrained.size > 0:
        in_dim_a = N_anti * 2
        x_a = jnp.ones((z.shape[0], in_dim_a), dtype=jnp.float32)
        W_a = zeros_anti_unconstrained
        pre_a = x_a @ W_a
        pre_a = kfac_jax.register_dense(
            pre_a,
            x_a,
            W_a,
            variant="repeated_dense",
            type="full",
        )
        shift_a = kfac_dense_eps * jnp.real(pre_a)
        log_env_eo = log_env_eo + shift_a

    # --- exponentiate and apply gain ---
    env_eo = jnp.exp(log_env_eo).astype(jnp.complex64)
    gain_c = gain.astype(env_eo.dtype)
    env_eo = env_eo * gain_c[None, :]

    return env_eo

  return envelopes.Envelope(envelopes.EnvelopeType.PRE_DETERMINANT, init, apply)


def make_symmetric_gauge_LLL_envelope_zeros(
    lattice: jnp.ndarray,
    z_scale: float = 1.0,
    magfield_kwargs=None,
    use_kfac_dense: bool = False,
    kfac_dense_eps: float = 1e-4,
) -> "envelopes.Envelope":
  """
  Envelope built from symmetric-gauge periodic LLL orbitals with trainable zeros.

  For each orbital o we define:

      Ψ_o(r) = ∏_{i=1..N_zero} Φ( r - a_{o,i} ),

  where:
    - a_{o,i} are trainable zeros (real 2D positions in the supercell),
    - Φ is the symmetric-gauge periodic LLL block constructed via
      magnetic translations and mapping (r - a_{o,i}) back to the
      supercell, including the appropriate magnetic phase.

  We never evaluate Φ outside the supercell: each (r - a_i) is mapped
  to the fundamental cell before calling symmetric_gauge_periodic_LLL.

  Trainable params per spin channel:
    - zeros_unconstrained : (N_zero * 2, odim)
    - gain                : (odim,)

  Args:
    lattice: (2, 2) real-space primitive vectors as columns [L1, L2].
    z_scale: scaling applied to the electron positions (mostly kept for
             API symmetry; here we work in real r, not z directly).
    magfield_kwargs: dict, must contain:
        - "N_phi": number of flux quanta (we set N_zero = N_phi)
        Optional:
        - "N_zero": override number of zeros per orbital.
        - "nmax": integer cutoff for symmetric_gauge_periodic_LLL translations.
    use_kfac_dense: if True, tag zeros as weights of a repeated dense layer.
    kfac_dense_eps: small coupling strength for KFAC tagging.

  Returns:
    envelopes.Envelope with apply_type PRE_DETERMINANT.
  """
  if magfield_kwargs is None:
    magfield_kwargs = {}

  # --- flux / zero bookkeeping ---
  N_phi = int(magfield_kwargs["N_phi"])
  N_zero = int(magfield_kwargs.get("N_zero", N_phi))
  nmax = int(magfield_kwargs.get("nmax", 20))

  # you may want to keep this constraint for now
  assert N_zero == N_phi, (
      f"Expect N_zero == N_phi for now, got N_zero={N_zero}, N_phi={N_phi}"
  )

  # --- geometry (real-space) ---
  L1 = lattice[:, 0].astype(jnp.float32)  # (2,)
  L2 = lattice[:, 1].astype(jnp.float32)  # (2,)

  # ---------- init: sample zeros per orbital ----------
  def init(
      natom: int,
      output_dims: Sequence[int],
      ndim: int = 2,
  ) -> Sequence[Mapping[str, jnp.ndarray]]:
    del natom, ndim

    params = []
    key = jax.random.PRNGKey(95)

    for odim in output_dims:
      key, subkey = jax.random.split(key)

      if N_zero > 0:
        # Sample zero positions uniformly in the supercell
        # uv: (odim, N_zero, 2) with components in [0,1)
        uv = jax.random.uniform(subkey, (odim, N_zero, 2), dtype=jnp.float32)

        # zeros in real space: u * L1 + v * L2
        zeros_init = (uv[..., 0:1] * L1[None, None, :] +
                      uv[..., 1:2] * L2[None, None, :])  # (odim, N_zero, 2)

        # Optional centering per orbital (keeps COM of zeros near origin)
        zeros_mean = jnp.mean(zeros_init, axis=1, keepdims=True)    # (odim, 1, 2)
        zeros_centered = zeros_init - zeros_mean                    # (odim, N_zero, 2)

        # Pack into (N_zero*2, odim) as "weights":
        #   (odim, N_zero, 2) -> (N_zero, 2, odim) -> (N_zero*2, odim)
        zeros_weight = jnp.transpose(zeros_centered, (1, 2, 0))     # (N_zero, 2, odim)
        zeros_weight = zeros_weight.reshape(N_zero * 2, odim)       # (N_zero*2, odim)
      else:
        zeros_weight = jnp.zeros((0, odim), dtype=jnp.float32)

      params.append({
          "gain": jnp.ones((odim,), dtype=jnp.float32),
          "zeros_holo_unconstrained": zeros_weight.astype(jnp.float32),
      })

    return params

  # ---------- apply: build orbital envelopes ----------
  def apply(
      *,
      ae: jnp.ndarray,
      r_ae: jnp.ndarray,
      r_ee: jnp.ndarray,
      gain: jnp.ndarray,
      zeros_holo_unconstrained: jnp.ndarray,
  ) -> jnp.ndarray:
    """
    ae:  (nelectron, natom, 2)
    returns: (nelectron, output_dim) complex envelope values.
    """
    del r_ae, r_ee

    # electron positions (ne, 2)
    pos = ae[:, 0, :].astype(jnp.float32) * z_scale   # (ne, 2)
    ne = pos.shape[0]
    odim = gain.shape[0]

    # Decode zeros per orbital: (N_zero*2, odim) -> (odim, N_zero, 2)
    if N_zero > 0 and zeros_holo_unconstrained.size > 0:
      assert zeros_holo_unconstrained.shape == (N_zero * 2, odim)
      zt = zeros_holo_unconstrained.reshape(N_zero, 2, odim)   # (N_zero, 2, odim)
      zeros_xy = jnp.transpose(zt, (2, 0, 1))            # (odim, N_zero, 2)
    else:
      zeros_xy = jnp.zeros((odim, 0, 2), dtype=jnp.float32)

    # --- envelope per orbital ---
    def env_one_orbital(zeros_for_orb: jnp.ndarray) -> jnp.ndarray:
      """
      zeros_for_orb: (N_zero, 2) real-space zeros for this orbital.
      returns: (ne,) complex envelope for this orbital.
      """
      if zeros_for_orb.shape[0] == 0:
        # no zeros -> constant envelope = 1
        return jnp.ones((ne,), dtype=jnp.complex64)

      def psi_e(r_e):
        # symmetric-gauge periodic LLL with zeros, using mapped coordinates
        return ellipticfunctions.symmetric_gauge_periodic_LLL_with_zeros_mapped(
            r_e, zeros_for_orb, L1, L2, nmax=nmax
        )

      return jax.vmap(psi_e)(pos).astype(jnp.complex64)

    # vectorize over orbitals -> (odim, ne)
    env_od = jax.vmap(env_one_orbital, in_axes=0)(zeros_xy)  # (odim, ne)
    env_eo = jnp.swapaxes(env_od, 0, 1)                      # (ne, odim)

    # --- KFAC dense tagging (optional, same pattern as before) ---
    if use_kfac_dense and N_zero > 0 and zeros_holo_unconstrained.size > 0:
      in_dim = N_zero * 2

      # x: (ne, in_dim) - dummy inputs (all ones) to define Fisher block
      x = jnp.ones((ne, in_dim), dtype=jnp.float32)

      # W: (in_dim, odim) - actual parameter leaf (zeros_unconstrained)
      W = zeros_unconstrained

      pre = x @ W  # (ne, odim)
      pre = kfac_jax.register_dense(
          pre,
          x,
          W,
          variant="repeated_dense",
          type="full",
      )

      shift = kfac_dense_eps * jnp.real(pre)  # (ne, odim)
      # couple KFAC "pre-activation" into env via a tiny multiplicative factor
      env_eo = env_eo * jnp.exp(shift.astype(env_eo.dtype))

    # --- apply gain ---
    gain_c = gain.astype(env_eo.dtype)    # (odim,)
    env_eo = env_eo * gain_c[None, :]     # (ne, odim)

    return env_eo

  return envelopes.Envelope(envelopes.EnvelopeType.PRE_DETERMINANT, init, apply)
# assumes you already have ellipticfunctions imported
# from your own module
# import ellipticfunctions



# def make_LLL_envelope_2d_trainable_zeros_mixed2_nminus1(
#     lattice: jnp.ndarray,
#     elliptic_log_sigma=None,   # kept for API compatibility
#     z_scale: float = 1.0,
#     magfield_kwargs=None,
# ) -> envelopes.Envelope:
#   """
#   Variant of make_LLL_envelope_2d_trainable_zeros_mixed2 where, in each sector
#   (holomorphic / anti-holomorphic) and for each orbital, we parameterize only
#   N-1 zeros freely and define the last zero as minus the sum of the others:

#       a_N = - Σ_{i=1}^{N-1} a_i

#   so that Σ_{i=1}^N a_i = 0 is enforced exactly by construction.

#   Trainable params:
#     - zeros_holo_free   : (odim, max(N_holo-1, 0), 2)
#     - zeros_anti_free   : (odim, max(N_anti-1, 0), 2)
#     - gain              : (odim,)

#   All geometric/theta constants are precomputed once and *not* trained.
#   """
#   if magfield_kwargs is None:
#     magfield_kwargs = {}

#   # --- flux bookkeeping ---
#   N_phi  = int(magfield_kwargs["N_phi"])
#   N_holo = int(magfield_kwargs.get("N_holo", N_phi))
#   N_anti = int(magfield_kwargs.get("N_anti", 0))

#   assert N_holo - N_anti == N_phi, (
#       f"Require N_holo - N_anti == N_phi, got N_holo={N_holo}, "
#       f"N_anti={N_anti}, N_phi={N_phi}"
#   )

#   # numbers of *free* zeros per sector (the last one is dependent)
#   N_holo_free = max(N_holo - 1, 0)
#   N_anti_free = max(N_anti - 1, 0)

#   # --- geometry constants (shared, not trainable) ---
#   L1 = lattice[:, 0].astype(jnp.float32)
#   L2 = lattice[:, 1].astype(jnp.float32)

#   L1com = ellipticfunctions.to_cplx_divsqrt2(L1)
#   L2com = ellipticfunctions.to_cplx_divsqrt2(L2)
#   w1 = (L1com / 2.0).astype(jnp.complex64)
#   w2 = (L2com / 2.0).astype(jnp.complex64)
#   tau = (w2 / w1).astype(jnp.complex64)
#   pi_c = jnp.asarray(jnp.pi, jnp.float32).astype(jnp.complex64)

#   # theta-series coefficients + derivatives, cached once
#   theta_coeffs = ellipticfunctions._precompute_theta_coeffs(tau, max_terms=15)
#   t1p0, t1ppp0 = ellipticfunctions._theta_derivs0_from_coeffs(theta_coeffs)
#   c = - (pi_c * pi_c) / (24.0 * w1 * w1) * (t1ppp0 / t1p0)

#   # "almost" term
#   almost = ellipticfunctions.almost_modular(L1, L2, magfield_kwargs["N_phi"])

#   consts_holo = {
#       'L1': L1,
#       'L2': L2,
#       'w1': w1,
#       'w2': w2,
#       'tau': tau,
#       'pi': pi_c,
#       'theta_coeffs': theta_coeffs,
#       't1p0': t1p0,
#       't1ppp0': t1ppp0,
#       'c': c,
#       'N_phi': jnp.asarray(N_holo if N_holo > 0 else 1, jnp.float32),
#       'almost': almost,
#   }
#   consts_anti = {
#       'L1': L1,
#       'L2': L2,
#       'w1': w1,
#       'w2': w2,
#       'tau': tau,
#       'pi': pi_c,
#       'theta_coeffs': theta_coeffs,
#       't1p0': t1p0,
#       't1ppp0': t1ppp0,
#       'c': c,
#       'N_phi': jnp.asarray(N_anti if N_anti > 0 else 1, jnp.float32),
#       'almost': almost,
#   }

#   consts_holo = jax.tree.map(lambda x: jax.lax.stop_gradient(jnp.asarray(x)), consts_holo)
#   consts_anti = jax.tree.map(lambda x: jax.lax.stop_gradient(jnp.asarray(x)), consts_anti)

#   # ---------- init: sample N-1 free zeros per orbital in each sector ----------
#   def init(natom: int, output_dims: Sequence[int], ndim: int = 2
#           ) -> Sequence[Mapping[str, jnp.ndarray]]:
#     del natom, ndim

#     params = []
#     key = jax.random.PRNGKey(1395)

#     for odim in output_dims:
#       key, subkey_h = jax.random.split(key)
#       key, subkey_a = jax.random.split(key)

#       # --- holomorphic free zeros (N_holo_free per orbital) ---
#       if N_holo_free > 0:
#         uv_h = jax.random.uniform(subkey_h,
#                                   (odim, N_holo_free, 2),
#                                   dtype=jnp.float32)
#         zeros_h_free_init = (uv_h[..., 0:1] * L1[None, None, :]
#                              + uv_h[..., 1:2] * L2[None, None, :])
#       else:
#         zeros_h_free_init = jnp.zeros((odim, 0, 2), dtype=jnp.float32)

#       # --- anti-holomorphic free zeros (N_anti_free per orbital) ---
#       if N_anti_free > 0:
#         uv_a = jax.random.uniform(subkey_a,
#                                   (odim, N_anti_free, 2),
#                                   dtype=jnp.float32)
#         zeros_a_free_init = (uv_a[..., 0:1] * L1[None, None, :]
#                              + uv_a[..., 1:2] * L2[None, None, :])
#       else:
#         zeros_a_free_init = jnp.zeros((odim, 0, 2), dtype=jnp.float32)

#       params.append({
#           'gain': jnp.ones((odim,), dtype=jnp.float32),
#           'zeros_holo_free': zeros_h_free_init.astype(jnp.float32),
#           'zeros_anti_free': zeros_a_free_init.astype(jnp.float32),
#       })

#     return params

#   # ---------- helper: build full zero set from N-1 free + dependent last -----
#   def _build_full_zeros(zeros_free: jnp.ndarray, N_total: int) -> jnp.ndarray:
#     """
#     zeros_free: (odim, N_free, 2), with N_free = max(N_total - 1, 0).
#     Returns zeros_all: (odim, N_total, 2) with last zero = -sum of free zeros,
#     ensuring sum_a zeros_all[a] = 0.
#     """
#     odim = zeros_free.shape[0]
#     if N_total == 0:
#       return jnp.zeros((odim, 0, 2), dtype=zeros_free.dtype)

#     N_free = zeros_free.shape[1]
#     if N_free > 0:
#       sum_free = jnp.sum(zeros_free, axis=1, keepdims=True)  # (odim, 1, 2)
#       last_zero = -sum_free                                  # (odim, 1, 2)
#       zeros_all = jnp.concatenate([zeros_free, last_zero], axis=1)  # (odim, N_total, 2)
#     else:
#       # N_total == 1 and no free zeros: the single zero is at the origin.
#       zeros_all = jnp.zeros((odim, 1, 2), dtype=zeros_free.dtype)
#     return zeros_all

#   # ---------- apply: combine holo + anti sectors with N-1 parametrization -----
#   def apply(*, ae: jnp.ndarray, r_ae: jnp.ndarray, r_ee: jnp.ndarray,
#             gain: jnp.ndarray,
#             zeros_holo_free: jnp.ndarray,
#             zeros_anti_free: jnp.ndarray) -> jnp.ndarray:
#     """ae: (nelectron, natom, 2) -> returns (nelectron, output_dim)."""
#     del r_ae, r_ee

#     # positions (ne,2)
#     pos = ae[:, 0, :].astype(jnp.float32)
#     z = z_scale * (pos[..., 0] + 1j * pos[..., 1]) / jnp.sqrt(2.0)
#     z = z.astype(jnp.complex64)  # (ne,)
#     ne = z.shape[0]
#     odim = gain.shape[0]

#     # --- reconstruct full zero sets from free + dependent last ---

#     # holomorphic sector
#     if N_holo > 0:
#       zeros_h_all = _build_full_zeros(zeros_holo_free, N_holo)  # (odim, N_holo, 2)
#       zeros_h_c = (zeros_h_all[..., 0] + 1j * zeros_h_all[..., 1]).astype(jnp.complex64)  # (odim, N_holo)
#     else:
#       zeros_h_c = jnp.zeros((odim, 0), dtype=jnp.complex64)

#     # anti-holomorphic sector
#     if N_anti > 0:
#       zeros_a_all = _build_full_zeros(zeros_anti_free, N_anti)  # (odim, N_anti, 2)
#       zeros_a_c = (zeros_a_all[..., 0] + 1j * zeros_a_all[..., 1]).astype(jnp.complex64)  # (odim, N_anti)
#     else:
#       zeros_a_c = jnp.zeros((odim, 0), dtype=jnp.complex64)

#     # --- log envelopes per orbital ---

#     def log_env_holo_one_orbital(zeros_k: jnp.ndarray) -> jnp.ndarray:
#       """Holomorphic sector for one orbital: (ne,) complex."""
#       if zeros_k.shape[0] == 0 or N_holo == 0:
#         return jnp.zeros((ne,), dtype=jnp.complex64)
#       return jax.vmap(
#           lambda z_e: ellipticfunctions._LLL_with_zeros_log_cached(
#               z_e, zeros_k, consts_holo
#           )
#       )(z)

#     def log_env_anti_one_orbital(zeros_k: jnp.ndarray) -> jnp.ndarray:
#       """Anti-holomorphic sector for one orbital: (ne,) complex."""
#       if zeros_k.shape[0] == 0 or N_anti == 0:
#         return jnp.zeros((ne,), dtype=jnp.complex64)

#       def one_e(z_e):
#         # treat conj(z) as holomorphic variable and then conjugate the log
#         val = ellipticfunctions._LLL_with_zeros_log_cached(
#             jnp.conj(z_e), zeros_k, consts_anti
#         )
#         return jnp.conj(val)

#       return jax.vmap(one_e)(z)

#     # vectorize over orbitals -> (odim, ne)
#     log_env_holo_od = jax.vmap(log_env_holo_one_orbital, in_axes=0)(zeros_h_c)  # (odim, ne)
#     log_env_anti_od = jax.vmap(log_env_anti_one_orbital, in_axes=0)(zeros_a_c)  # (odim, ne)

#     # swap to (ne, odim)
#     log_env_holo_eo = jnp.swapaxes(log_env_holo_od, 0, 1)  # (ne, odim)
#     log_env_anti_eo = jnp.swapaxes(log_env_anti_od, 0, 1)  # (ne, odim)

#     # total
#     log_env_eo = log_env_holo_eo + log_env_anti_eo  # (ne, odim)
#     # def _clamp_log(zlog, lo=-60.0, hi=60.0):
#     #   re = jnp.clip(jnp.real(zlog), a_min=lo, a_max=hi)
#     #   return re + 1j * jnp.imag(zlog)

#     # log_env_eo = _clamp_log(log_env_eo)
#     env_eo = jnp.exp(log_env_eo).astype(jnp.complex64)
#     gain_c = gain.astype(env_eo.dtype)
#     env_eo = env_eo * gain_c[None, :]

#     return env_eo

#   return envelopes.Envelope(envelopes.EnvelopeType.PRE_DETERMINANT, init, apply)


def make_LLL_envelope_2d_trainable_zeros_mixed2_nminus1(
    lattice: jnp.ndarray,
    elliptic_log_sigma=None,   # kept for API compatibility
    z_scale: float = 1.0,
    magfield_kwargs=None,
) -> envelopes.Envelope:
  """
  LLL envelope with trainable zeros in holomorphic and anti-holomorphic sectors.

  This version parameterizes zeros via a Fourier-like basis that *automatically*
  enforces sum_a a_a = 0 (exactly) in R^2 for each orbital and sector.

  For each sector (holo / anti) and orbital:
    - We have N_total zeros (N_holo or N_anti).
    - We use N_total - 1 Fourier modes (k = 1..N_total-1) as parameters.
    - zeros_*_free[orb, k, 2] are the real-space Fourier coefficients c_k (x,y).
    - Zeros are built as:
          a_j = sum_{k=1..N-1} c_k cos(2π j k / N),  j=0..N-1
      which guarantees sum_j a_j = 0 componentwise.

  Trainable params per Envelope entry:
    - gain              : (odim,)
    - zeros_holo_free   : (odim, max(N_holo-1, 0), 2)
    - zeros_anti_free   : (odim, max(N_anti-1, 0), 2)
  """
  if magfield_kwargs is None:
    magfield_kwargs = {}

  # --- flux bookkeeping ---
  N_phi  = int(magfield_kwargs["N_phi"])
  N_holo = int(magfield_kwargs.get("N_holo", N_phi))
  N_anti = int(magfield_kwargs.get("N_anti", 0))

  assert N_holo - N_anti == N_phi, (
      f"Require N_holo - N_anti == N_phi, got N_holo={N_holo}, "
      f"N_anti={N_anti}, N_phi={N_phi}"
  )

  # numbers of Fourier modes per sector (N_total-1, if N_total>0)
  N_holo_free = max(N_holo - 1, 0)
  N_anti_free = max(N_anti - 1, 0)

  # --- geometry constants (shared, not trainable) ---
  L1 = lattice[:, 0].astype(jnp.float32)  # (2,)
  L2 = lattice[:, 1].astype(jnp.float32)  # (2,)

  L1com = ellipticfunctions.to_cplx_divsqrt2(L1)
  L2com = ellipticfunctions.to_cplx_divsqrt2(L2)
  w1 = (L1com / 2.0).astype(jnp.complex64)
  w2 = (L2com / 2.0).astype(jnp.complex64)
  tau = (w2 / w1).astype(jnp.complex64)
  pi_c = jnp.asarray(jnp.pi, jnp.float32).astype(jnp.complex64)

  # theta-series coefficients + derivatives, cached once
  theta_coeffs = ellipticfunctions._precompute_theta_coeffs(tau, max_terms=15)
  t1p0, t1ppp0 = ellipticfunctions._theta_derivs0_from_coeffs(theta_coeffs)
  c = - (pi_c * pi_c) / (24.0 * w1 * w1) * (t1ppp0 / t1p0)

  # "almost" term
  almost = ellipticfunctions.almost_modular(L1, L2, magfield_kwargs["N_phi"])

  consts_holo = {
      'L1': L1,
      'L2': L2,
      'w1': w1,
      'w2': w2,
      'tau': tau,
      'pi': pi_c,
      'theta_coeffs': theta_coeffs,
      't1p0': t1p0,
      't1ppp0': t1ppp0,
      'c': c,
      'N_phi': jnp.asarray(N_holo if N_holo > 0 else 1, jnp.float32),
      'almost': almost,
  }
  consts_anti = {
      'L1': L1,
      'L2': L2,
      'w1': w1,
      'w2': w2,
      'tau': tau,
      'pi': pi_c,
      'theta_coeffs': theta_coeffs,
      't1p0': t1p0,
      't1ppp0': t1ppp0,
      'c': c,
      'N_phi': jnp.asarray(N_anti if N_anti > 0 else 1, jnp.float32),
      'almost': almost,
  }

  consts_holo = jax.tree.map(lambda x: jax.lax.stop_gradient(jnp.asarray(x)), consts_holo)
  consts_anti = jax.tree.map(lambda x: jax.lax.stop_gradient(jnp.asarray(x)), consts_anti)

  # ---------- Fourier helper: modes -> zeros with sum_j zeros[j] = 0 ----------

  def _zeros_from_fourier_modes_2d(
      modes: jnp.ndarray,  # (odim, N_modes, 2)
      N_total: int,
  ) -> jnp.ndarray:
    """
    Given Fourier-like coefficients 'modes' (odim, N_modes, 2),
    build zeros_all: (odim, N_total, 2) such that sum_j zeros_all[j] = 0.

    For each orbital and component α = x,y:
        x_j^α = Σ_{k=1..N_total-1} c_k^α cos(2π j k / N_total)
    with N_modes = max(N_total-1, 0).
    """
    odim = modes.shape[0]
    if N_total == 0:
      return jnp.zeros((odim, 0, 2), dtype=modes.dtype)

    if N_total == 1:
      # Single zero: must be exactly 0 to have sum=0
      return jnp.zeros((odim, 1, 2), dtype=modes.dtype)

    N_modes = modes.shape[1]
    assert N_modes == N_total - 1, (
        f"Expected N_modes = N_total - 1, got N_modes={N_modes}, N_total={N_total}"
    )

    # j: 0..N_total-1, k: 1..N_total-1
    j = jnp.arange(N_total, dtype=jnp.float32)           # (N_total,)
    k = jnp.arange(1, N_total, dtype=jnp.float32)        # (N_modes,)

    phase = 2.0 * jnp.pi * (j[:, None] * k[None, :] / float(N_total))  # (N_total, N_modes)
    W = jnp.cos(phase).astype(jnp.float32)                              # (N_total, N_modes)

    # modes: (odim, N_modes, 2)
    # zeros_all: (odim, N_total, 2) = Σ_k W[j,k] * modes[orb,k,α]
    # use einsum: 'jk,okc->ojc'
    zeros_all = jnp.einsum('jk,okc->ojc', W, modes).astype(jnp.float32)  # (odim,N_total,2)

    # By construction, sum over j of each column is zero:
    #   sum_j W[j,k] = 0 for each k>=1 → sum_j zeros_all[j] = 0.
    return zeros_all

  # ---------- init: sample Fourier modes per orbital in each sector ----------
  def init(natom: int, output_dims: Sequence[int], ndim: int = 2
          ) -> Sequence[Mapping[str, jnp.ndarray]]:
    del natom, ndim

    params = []
    key = jax.random.PRNGKey(23)

    for odim in output_dims:
      key, subkey_h = jax.random.split(key)
      key, subkey_a = jax.random.split(key)

      # --- holomorphic Fourier modes (N_holo_free per orbital, 2 components) ---
      if N_holo_free > 0:
        zeros_h_free_init = (0.2) * jax.random.normal(
            subkey_h,
            (odim, N_holo_free, 2),
            dtype=jnp.float32,
        )
      else:
        zeros_h_free_init = jnp.zeros((odim, 0, 2), dtype=jnp.float32)

      # --- anti-holomorphic Fourier modes ---
      if N_anti_free > 0:
        zeros_a_free_init = (0.2) * jax.random.normal(
            subkey_a,
            (odim, N_anti_free, 2),
            dtype=jnp.float32,
        )
      else:
        zeros_a_free_init = jnp.zeros((odim, 0, 2), dtype=jnp.float32)

      params.append({
          'gain': jnp.ones((odim,), dtype=jnp.float32),
          # these are now Fourier modes, not literal zero positions
          'zeros_holo_free': zeros_h_free_init.astype(jnp.float32),
          'zeros_anti_free': zeros_a_free_init.astype(jnp.float32),
      })

    return params

  # ---------- apply: build zeros from Fourier modes and call LLL envelope -----
  def apply(*, ae: jnp.ndarray, r_ae: jnp.ndarray, r_ee: jnp.ndarray,
            gain: jnp.ndarray,
            zeros_holo_free: jnp.ndarray,
            zeros_anti_free: jnp.ndarray) -> jnp.ndarray:
    """ae: (nelectron, natom, 2) -> returns (nelectron, output_dim)."""
    del r_ae, r_ee

    # positions (ne,2) → complex LLL coords
    pos = ae[:, 0, :].astype(jnp.float32)
    z = z_scale * (pos[..., 0] + 1j * pos[..., 1]) / jnp.sqrt(2.0)
    z = z.astype(jnp.complex64)  # (ne,)
    ne = z.shape[0]
    odim = gain.shape[0]

    # --- build real-space zeros for each sector via Fourier modes ---

    # holomorphic sector
    if N_holo > 0:
      zeros_h_all = _zeros_from_fourier_modes_2d(zeros_holo_free, N_holo)  # (odim,N_holo,2)
      zeros_h_c = (zeros_h_all[..., 0] + 1j * zeros_h_all[..., 1]).astype(jnp.complex64)
    else:
      zeros_h_c = jnp.zeros((odim, 0), dtype=jnp.complex64)

    # anti-holomorphic sector
    if N_anti > 0:
      zeros_a_all = _zeros_from_fourier_modes_2d(zeros_anti_free, N_anti)  # (odim,N_anti,2)
      zeros_a_c = (zeros_a_all[..., 0] + 1j * zeros_a_all[..., 1]).astype(jnp.complex64)
    else:
      zeros_a_c = jnp.zeros((odim, 0), dtype=jnp.complex64)

    # --- log envelopes per orbital ---

    def log_env_holo_one_orbital(zeros_k: jnp.ndarray) -> jnp.ndarray:
      """Holomorphic sector for one orbital: (ne,) complex."""
      if zeros_k.shape[0] == 0 or N_holo == 0:
        return jnp.zeros((ne,), dtype=jnp.complex64)
      return jax.vmap(
          lambda z_e: ellipticfunctions._LLL_with_zeros_log_cached(
              z_e, zeros_k, consts_holo
          )
      )(z)

    def log_env_anti_one_orbital(zeros_k: jnp.ndarray) -> jnp.ndarray:
      """Anti-holomorphic sector for one orbital: (ne,) complex."""
      if zeros_k.shape[0] == 0 or N_anti == 0:
        return jnp.zeros((ne,), dtype=jnp.complex64)

      def one_e(z_e):
        # treat conj(z) as holomorphic variable and then conjugate the log
        val = ellipticfunctions._LLL_with_zeros_log_cached(
            jnp.conj(z_e), zeros_k, consts_anti
        )
        return jnp.conj(val)

      return jax.vmap(one_e)(z)

    # vectorize over orbitals -> (odim, ne)
    log_env_holo_od = jax.vmap(log_env_holo_one_orbital, in_axes=0)(zeros_h_c)  # (odim, ne)
    log_env_anti_od = jax.vmap(log_env_anti_one_orbital, in_axes=0)(zeros_a_c)  # (odim, ne)

    # swap to (ne, odim)
    log_env_holo_eo = jnp.swapaxes(log_env_holo_od, 0, 1)  # (ne, odim)
    log_env_anti_eo = jnp.swapaxes(log_env_anti_od, 0, 1)  # (ne, odim)

    # total log envelope
    log_env_eo = log_env_holo_eo + log_env_anti_eo  # (ne, odim)

    env_eo = jnp.exp(log_env_eo).astype(jnp.complex64)
    gain_c = gain.astype(env_eo.dtype)
    env_eo = env_eo * gain_c[None, :]

    return env_eo

  return envelopes.Envelope(envelopes.EnvelopeType.PRE_DETERMINANT, init, apply)




def make_LLL_envelope_2d_backflow(
    lattice: jnp.ndarray,
    elliptic_log_sigma,      # (z, N_phi, L1, L2, almost, zeros) -> log env
    z_scale: float = 1.0,
    magfield_kwargs=None,
) -> envelopes.Envelope:
  """LLL-like envelope with correlated (backflow) zeros, KFAC-safe.

  All scalar hyperparameters (bf_strength, bf_scale) are stored as JAX params,
  not as Python defaults, to avoid the kfac 'Literal' bug.
  """

  if magfield_kwargs is None:
    magfield_kwargs = {}

  # Lattice vectors, flux, almost
  L1 = lattice[:, 0].astype(jnp.float32)
  L2 = lattice[:, 1].astype(jnp.float32)

  N_phi = int(magfield_kwargs["N_phi"])
  almost = jnp.asarray(magfield_kwargs.get("almost", 0.0), dtype=jnp.float32)

  # How many zeros per orbital (for illustration; you can change this)
  N_zeros = int(magfield_kwargs.get("N_zeros", N_phi))

  # Optional initial values for backflow hyperparameters
  bf_strength_init = float(magfield_kwargs.get("bf_strength_init", 0.0))
  bf_scale_init    = float(magfield_kwargs.get("bf_scale_init", 1.0))

  # ---------- init ----------
  def init(natom: int, output_dims: Sequence[int], ndim: int = 2
           ) -> Sequence[Mapping[str, jnp.ndarray]]:
    del natom, ndim

    params = []
    key = jax.random.PRNGKey(1234)

    for odim in output_dims:
      # --- zeros per orbital: (odim, N_zeros, 2) ---
      key, subkey = jax.random.split(key)
      uv = jax.random.uniform(subkey, (odim, N_zeros, 2), dtype=jnp.float32)
      zeros_init = (
          uv[..., 0:1] * L1[None, None, :] +
          uv[..., 1:2] * L2[None, None, :]
      )
      zeros_mean = jnp.mean(zeros_init, axis=1, keepdims=True)
      zeros_centered = zeros_init - zeros_mean

      params.append({
          "gain": jnp.ones((odim,), dtype=jnp.float32),

          # zeros as real (odim, N_zeros, 2)
          "zeros_unconstrained": zeros_centered.astype(jnp.float32),

          # backflow hyperparameters as trainable 1D arrays
          "bf_strength": jnp.full((1,), bf_strength_init, dtype=jnp.float32),
          "bf_scale":    jnp.full((1,), bf_scale_init,    dtype=jnp.float32),
      })

    return params

  # ---------- apply ----------
  def apply(*,
            ae: jnp.ndarray,
            r_ae: jnp.ndarray,
            r_ee: jnp.ndarray,
            gain: jnp.ndarray,
            zeros_unconstrained: jnp.ndarray,
            bf_strength: jnp.ndarray,
            bf_scale: jnp.ndarray) -> jnp.ndarray:
    """ae: (nelectron, natom, 2) -> (nelectron, output_dim)."""
    del r_ae, r_ee  # we reconstruct e–e distances ourselves

    # (1) Extract positions and complex LLL coordinate
    pos = ae[:, 0, :].astype(jnp.float32)                # (ne, 2)
    z = z_scale * (pos[..., 0] + 1j * pos[..., 1]) / jnp.sqrt(2.0)
    z = z.astype(jnp.complex64)                          # (ne,)

    ne = z.shape[0]
    odim = gain.shape[0]

    # (2) construct electron–electron relative coords for backflow
    # rel[i,j] = r_i - r_j  -> (ne, ne, 2)
    rel = pos[:, None, :] - pos[None, :, :]              # (ne, ne, 2)

    lam2 = (bf_scale[0] ** 2).astype(jnp.float32) + 1e-6
    d2 = jnp.sum(rel**2, axis=-1)                        # (ne, ne)
    # simple scalar backflow field per electron
    f_e = jnp.sum(jnp.exp(-d2 / lam2), axis=1)           # (ne,)

    # (3) shift z by backflow field
    z_bf = z + bf_strength[0].astype(jnp.complex64) * f_e.astype(jnp.complex64)

    # (4) enforce zero-mean zeros per orbital
    mean_zeros = jnp.mean(zeros_unconstrained, axis=1, keepdims=True)   # (odim,1,2)
    zeros_centered = zeros_unconstrained - mean_zeros                   # (odim,N_zeros,2)
    zeros_c = (zeros_centered[..., 0] + 1j * zeros_centered[..., 1]).astype(jnp.complex64)  # (odim,N_zeros)

    N_phi_f32 = jnp.asarray(N_phi, dtype=jnp.float32)

    # (5) per-orbital log-envelope, vmapped over orbitals
    def log_env_one_orbital(zeros_k: jnp.ndarray) -> jnp.ndarray:
      """zeros_k: (N_zeros,) -> log_env(e)  (ne,)"""
      if zeros_k.shape[0] == 0:
        return jnp.zeros((ne,), dtype=jnp.complex64)

      def one_e(z_e):
        return elliptic_log_sigma(z_e, N_phi_f32, L1, L2, almost, zeros_k)

      return jax.vmap(one_e)(z_bf)  # (ne,)

    # (odim, ne)
    log_env_od = jax.vmap(log_env_one_orbital, in_axes=0)(zeros_c)
    # (ne, odim)
    log_env_eo = jnp.swapaxes(log_env_od, 0, 1)

    env_eo = jnp.exp(log_env_eo).astype(jnp.complex64)
    gain_c = gain.astype(env_eo.dtype)
    env_eo = env_eo * gain_c[None, :]

    return env_eo

  return envelopes.Envelope(envelopes.EnvelopeType.PRE_DETERMINANT, init, apply)


def make_LLL_envelope_2d_backflow_cached(
    lattice: jnp.ndarray,
    elliptic_log_sigma=None,      # kept only for API compatibility, not used
    z_scale: float = 1.0,
    magfield_kwargs=None,
) -> envelopes.Envelope:
  """LLL-like envelope with correlated (backflow) zeros using cached LLL.

  - Uses ellipticfunctions._LLL_with_zeros_log_cached(z; zeros_k, consts_holo)
    with precomputed theta coefficients etc.
  - Only a holomorphic sector (no anti-holomorphic part).
  - Zeros are backflow-shifted via a scalar field depending on e–e distances.

  Trainable params per spin channel:
    - zeros_unconstrained : (odim, N_zeros, 2)
    - bf_strength         : (1,)
    - bf_scale            : (1,)
    - gain                : (odim,)
  """

  if magfield_kwargs is None:
    magfield_kwargs = {}

  # ---------------------------------------------------------------------------
  # Geometry / flux / constants (shared, non-trainable)
  # ---------------------------------------------------------------------------
  L1 = lattice[:, 0].astype(jnp.float32)
  L2 = lattice[:, 1].astype(jnp.float32)

  N_phi = int(magfield_kwargs["N_phi"])
  N_zeros = int(magfield_kwargs.get("N_zeros", N_phi))

  # Complex periods
  L1com = ellipticfunctions.to_cplx_divsqrt2(L1)
  L2com = ellipticfunctions.to_cplx_divsqrt2(L2)
  w1 = (L1com / 2.0).astype(jnp.complex64)
  w2 = (L2com / 2.0).astype(jnp.complex64)
  tau = (w2 / w1).astype(jnp.complex64)
  pi_c = jnp.asarray(jnp.pi, jnp.float32).astype(jnp.complex64)

  # θ1-series + derivatives
  theta_coeffs = ellipticfunctions._precompute_theta_coeffs(tau, max_terms=15)
  t1p0, t1ppp0 = ellipticfunctions._theta_derivs0_from_coeffs(theta_coeffs)
  c = - (pi_c * pi_c) / (24.0 * w1 * w1) * (t1ppp0 / t1p0)

  # "almost" term (modular correction)
  almost = ellipticfunctions.almost_modular(L1, L2, N_phi)

  consts_holo = {
      "L1": L1,
      "L2": L2,
      "w1": w1,
      "w2": w2,
      "tau": tau,
      "pi": pi_c,
      "theta_coeffs": theta_coeffs,
      "t1p0": t1p0,
      "t1ppp0": t1ppp0,
      "c": c,
      "N_phi": jnp.asarray(N_phi, jnp.float32),
      "almost": almost,
  }

  # Stop gradients through geometry / theta data
  consts_holo = jax.tree.map(
      lambda x: jax.lax.stop_gradient(jnp.asarray(x)),
      consts_holo,
  )

  # Optional initial values for backflow hyperparameters
  bf_strength_init = float(magfield_kwargs.get("bf_strength_init", 0.0))
  bf_scale_init    = float(magfield_kwargs.get("bf_scale_init", 1.0))

  # ---------------------------------------------------------------------------
  # init
  # ---------------------------------------------------------------------------
  def init(
      natom: int,
      output_dims: Sequence[int],
      ndim: int = 2,
  ) -> Sequence[Mapping[str, jnp.ndarray]]:
    del natom, ndim

    params = []
    key = jax.random.PRNGKey(1234)

    for odim in output_dims:
      key, subkey = jax.random.split(key)

      # zeros per orbital: (odim, N_zeros, 2)
      uv = jax.random.uniform(subkey, (odim, N_zeros, 2), dtype=jnp.float32)
      zeros_init = (
          uv[..., 0:1] * L1[None, None, :] +
          uv[..., 1:2] * L2[None, None, :]
      )
      # center per orbital: sum of zeros = 0
      zeros_mean = jnp.mean(zeros_init, axis=1, keepdims=True)
      zeros_centered = zeros_init - zeros_mean

      params.append({
          "gain": jnp.ones((odim,), dtype=jnp.float32),

          # zeros as real (odim, N_zeros, 2)
          "zeros_unconstrained": zeros_centered.astype(jnp.float32),

          # backflow hyperparameters as trainable 1D arrays
          "bf_strength": jnp.full((1,), bf_strength_init, dtype=jnp.float32),
          "bf_scale":    jnp.full((1,), bf_scale_init,    dtype=jnp.float32),
      })

    return params

  # ---------------------------------------------------------------------------
  # apply
  # ---------------------------------------------------------------------------
  def apply(
      *,
      ae: jnp.ndarray,
      r_ae: jnp.ndarray,
      r_ee: jnp.ndarray,
      gain: jnp.ndarray,
      zeros_unconstrained: jnp.ndarray,
      bf_strength: jnp.ndarray,
      bf_scale: jnp.ndarray,
  ) -> jnp.ndarray:
    """ae: (nelectron, natom, 2) -> (nelectron, output_dim)."""
    del r_ae, r_ee  # we reconstruct e–e distances ourselves

    # (1) Positions and complex LLL coordinate
    pos = ae[:, 0, :].astype(jnp.float32)  # (ne, 2)
    z = z_scale * (pos[..., 0] + 1j * pos[..., 1]) / jnp.sqrt(2.0)
    z = z.astype(jnp.complex64)           # (ne,)

    ne = z.shape[0]
    odim = gain.shape[0]

    # (2) electron–electron relative coords for backflow
    rel = pos[:, None, :] - pos[None, :, :]   # (ne, ne, 2)
    d2 = jnp.sum(rel**2, axis=-1)            # (ne, ne)

    lam2 = (bf_scale[0] ** 2).astype(jnp.float32) + 1e-6
    f_e = jnp.sum(jnp.exp(-d2 / lam2), axis=1)  # (ne,)

    # (3) backflow-shifted complex coordinate
    z_bf = z + bf_strength[0].astype(jnp.complex64) * f_e.astype(jnp.complex64)

    # (4) enforce zero-mean zeros per orbital
    mean_zeros = jnp.mean(zeros_unconstrained, axis=1, keepdims=True)   # (odim,1,2)
    zeros_centered = zeros_unconstrained - mean_zeros                   # (odim,N_zeros,2)
    zeros_c = (zeros_centered[..., 0] + 1j * zeros_centered[..., 1]).astype(
        jnp.complex64
    )  # (odim, N_zeros)

    # (5) per-orbital log-envelope via cached LLL
    def log_env_one_orbital(zeros_k: jnp.ndarray) -> jnp.ndarray:
      """zeros_k: (N_zeros,) -> log_env(e)  (ne,)"""
      if zeros_k.shape[0] == 0:
        return jnp.zeros((ne,), dtype=jnp.complex64)

      def one_e(z_e):
        return ellipticfunctions._LLL_with_zeros_log_cached(
            z_e, zeros_k, consts_holo
        )

      return jax.vmap(one_e)(z_bf)  # (ne,)

    # (odim, ne)
    log_env_od = jax.vmap(log_env_one_orbital, in_axes=0)(zeros_c)
    # (ne, odim)
    log_env_eo = jnp.swapaxes(log_env_od, 0, 1)

    # (6) exponentiate and apply gain
    env_eo = jnp.exp(log_env_eo).astype(jnp.complex64)
    gain_c = gain.astype(env_eo.dtype)
    env_eo = env_eo * gain_c[None, :]

    return env_eo

  return envelopes.Envelope(envelopes.EnvelopeType.PRE_DETERMINANT, init, apply)


# def make_LLL_envelope_2d_trainable_zeros_mixed(
#     lattice: jnp.ndarray,
#     elliptic_log_sigma,
#     z_scale: float = 1.0,
#     magfield_kwargs = None,
# ) -> envelopes.Envelope:
#   if magfield_kwargs is None:
#     magfield_kwargs = {}

#   L1 = lattice[:, 0].astype(jnp.float32)
#   L2 = lattice[:, 1].astype(jnp.float32)

#   N_phi = int(magfield_kwargs["N_phi"])
#   N_holo = int(magfield_kwargs.get("N_holo", N_phi))
#   N_anti = int(magfield_kwargs.get("N_anti", 0))

#   # your constraint:
#   assert N_holo - N_anti == N_phi, "Require N_holo - N_anti == N_phi"

#   almost = jnp.asarray(magfield_kwargs.get("almost", 0.0), dtype=jnp.float32)

#   def init(natom: int, output_dims: Sequence[int], ndim: int = 2
#           ) -> Sequence[Mapping[str, jnp.ndarray]]:
#     del natom, ndim

#     params = []
#     key = jax.random.PRNGKey(1234)

#     for odim in output_dims:
#       key, subkey_h = jax.random.split(key)
#       key, subkey_a = jax.random.split(key)

#       if N_holo > 0:
#         uv_h = jax.random.uniform(subkey_h, (odim, N_holo, 2), dtype=jnp.float32)
#         zeros_h_init = (uv_h[..., 0:1] * L1[None, None, :] +
#                         uv_h[..., 1:2] * L2[None, None, :])
#         zeros_h_mean = jnp.mean(zeros_h_init, axis=1, keepdims=True)
#         zeros_h_centered = zeros_h_init - zeros_h_mean
#       else:
#         zeros_h_centered = jnp.zeros((odim, 0, 2), dtype=jnp.float32)

#       if N_anti > 0:
#         uv_a = jax.random.uniform(subkey_a, (odim, N_anti, 2), dtype=jnp.float32)
#         zeros_a_init = (uv_a[..., 0:1] * L1[None, None, :] +
#                         uv_a[..., 1:2] * L2[None, None, :])
#         zeros_a_mean = jnp.mean(zeros_a_init, axis=1, keepdims=True)
#         zeros_a_centered = zeros_a_init - zeros_a_mean
#       else:
#         zeros_a_centered = jnp.zeros((odim, 0, 2), dtype=jnp.float32)

#       params.append({
#           'gain': jnp.ones((odim,), dtype=jnp.float32),
#           'zeros_holo_unconstrained': zeros_h_centered.astype(jnp.float32),
#           'zeros_anti_unconstrained': zeros_a_centered.astype(jnp.float32),
#       })

#     return params

#   def apply(*, ae: jnp.ndarray, r_ae: jnp.ndarray, r_ee: jnp.ndarray,
#             gain: jnp.ndarray,
#             zeros_holo_unconstrained: jnp.ndarray,
#             zeros_anti_unconstrained: jnp.ndarray) -> jnp.ndarray:
#     """ae: (nelectron, natom, 2) -> returns (nelectron, output_dim)."""
#     del r_ae, r_ee

#     pos = ae[:, 0, :].astype(jnp.float32)  # (ne, 2)

#     z = z_scale * (pos[..., 0] + 1j * pos[..., 1]) / jnp.sqrt(2.0)
#     z = z.astype(jnp.complex64)           # (ne,)
#     z_conj = jnp.conj(z)

#     ne = int(z.shape[0])
#     odim = int(gain.shape[0])

#     # --- holomorphic zeros ---
#     if N_holo > 0:
#       mean_h = jnp.mean(zeros_holo_unconstrained, axis=1, keepdims=True)
#       zeros_h_centered = zeros_holo_unconstrained - mean_h
#       zeros_h_c = (zeros_h_centered[..., 0] + 1j * zeros_h_centered[..., 1]).astype(
#           jnp.complex64
#       )  # (odim, N_holo)
#     else:
#       zeros_h_c = jnp.zeros((odim, 0), dtype=jnp.complex64)

#     # --- anti-holomorphic zeros ---
#     if N_anti > 0:
#       mean_a = jnp.mean(zeros_anti_unconstrained, axis=1, keepdims=True)
#       zeros_a_centered = zeros_anti_unconstrained - mean_a
#       zeros_a_c = (zeros_a_centered[..., 0] + 1j * zeros_a_centered[..., 1]).astype(
#           jnp.complex64
#       )  # (odim, N_anti)
#     else:
#       zeros_a_c = jnp.zeros((odim, 0), dtype=jnp.complex64)

#     log_env_eo = jnp.zeros((ne, odim), dtype=jnp.complex64)
#     N_phi_f32 = jnp.asarray(N_phi, dtype=jnp.float32)

#     # Again, plain loops → no scan / fori / vmap
#     for k in range(odim):
#       zeros_h_k = zeros_h_c[k]  # (N_holo,)
#       zeros_a_k = zeros_a_c[k]  # (N_anti,)

#       for i in range(ne):
#         z_e = z[i]
#         z_e_conj = z_conj[i]

#         if zeros_h_k.shape[0] > 0:
#           log_h = elliptic_log_sigma(z_e, N_phi_f32, L1, L2, almost, zeros_h_k)
#         else:
#           log_h = jnp.array(0.0 + 0.0j, dtype=jnp.complex64)

#         if zeros_a_k.shape[0] > 0:
#           val_a = elliptic_log_sigma(z_e_conj, N_phi_f32, L1, L2, almost, zeros_a_k)
#           log_a = jnp.conj(val_a)
#         else:
#           log_a = jnp.array(0.0 + 0.0j, dtype=jnp.complex64)

#         log_env_eo = log_env_eo.at[i, k].set(log_h + log_a)

#     env_eo = jnp.exp(log_env_eo).astype(jnp.complex64)
#     gain_c = gain.astype(env_eo.dtype)
#     env_eo = env_eo * gain_c[None, :]

#     return env_eo

#   return envelopes.Envelope(envelopes.EnvelopeType.PRE_DETERMINANT, init, apply)


# def make_magnetic_phase_envelope_2d(
#     lattice: jnp.ndarray,
#     *,
#     use_nearest_image: bool = True,   # if True, infer integer windings from ae via nearest image
# ) -> envelopes.Envelope:
#   """Pure phase envelope enforcing magnetic quasi-periodicity via r×L/2.

#   Args:
#     lattice: (2,2) matrix with columns L1, L2 (primitive vectors).
#     use_nearest_image: if True, compute integer windings (n1,n2) from ae by
#       mapping to the nearest image; if False, assume ae already contains the
#       unwrapped displacement and infer n1,n2 by floor.

#   Returns:
#     ferminet.envelopes.Envelope with type PRE_DETERMINANT.
#   """
#   # Columns are L1, L2
#   L1 = lattice[:, 0]
#   L2 = lattice[:, 1]

#   # Precompute the solver that maps Cartesian -> lattice coords.
#   # For vectors r, fractional lattice coords t solve lattice @ t = r.
#   # (equivalently t = solve(lattice, r))
#   Linv = jnp.linalg.inv(lattice)

#   # ---------- init ----------
#   def init(natom: int, output_dims: Sequence[int], ndim: int = 2
#            ) -> Sequence[Mapping[str, jnp.ndarray]]:
#     del natom, ndim
#     params = []
#     for output_dim in output_dims:
#       params.append({'gain': jnp.ones((output_dim,))})
#     return params

#   # ---------- helpers (no side effects) ----------
#   def _infer_windings(r_xy: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
#     """Given a single 2D vector r, return (n, L_vec, r_cross_L).

#     n: (2,) int32 with windings (n1, n2)
#     L_vec: (2,) = n1*L1 + n2*L2
#     r_cross_L: scalar = r_x * L_y - r_y * L_x
#     """
#     # fractional lattice coords t = (t1, t2) such that r = t1 L1 + t2 L2
#     t = Linv @ r_xy  # shape (2,)

#     # Choose integers (n1, n2)
#     if use_nearest_image:
#       # nearest-image integers: round to nearest integer
#       n = jnp.floor(t + 0.5)
#     else:
#       # unwrapped-to-cell split: integer part by floor (keeps residual in [0,1))
#       n = jnp.floor(t)

#     # Stop gradients through the discrete choice
#     n = jax.lax.stop_gradient(n).astype(jnp.int32)

#     # Build lattice translation L = n1 L1 + n2 L2
#     L_vec = lattice @ n.astype(r_xy.dtype)

#     # 2D scalar cross product r × L = r_x L_y - r_y L_x
#     r_cross_L = r_xy[0] * L_vec[1] - r_xy[1] * L_vec[0]
#     return n, L_vec, r_cross_L

#   def _eta_from_n(n: jnp.ndarray) -> jnp.ndarray:
#     """η = +1 if n1,n2 both even (L/2 in lattice), else -1."""
#     n1_even = (n[0] & 1) == 0
#     n2_even = (n[1] & 1) == 0
#     both_even = jnp.logical_and(n1_even, n2_even)
#     # map True -> +1, False -> -1
#     return jnp.where(both_even, jnp.array(1.0, dtype=jnp.float32), jnp.array(-1.0, dtype=jnp.float32))

#   # ---------- apply ----------
#   def apply(*, ae: jnp.ndarray, r_ae: jnp.ndarray, r_ee: jnp.ndarray,
#             gain: jnp.ndarray) -> jnp.ndarray:
#     """ae: (nelectron, natom, 2) -> returns (nelectron, output_dim)."""
#     del r_ae, r_ee  # unused

#     # Vectorized per (electron, atom) phase calculation.
#     def phase_one(r_xy: jnp.ndarray):
#       n, _, r_cross_L = _infer_windings(r_xy)
#       eta = _eta_from_n(n).astype(r_xy.dtype)
#       return eta * jnp.exp(0.5j * r_cross_L)

#     # (nelectron, natom) complex array of phases
#     phase_ea = jax.vmap(                 # over electrons (axis 0)
#         jax.vmap(phase_one, in_axes=0),  # over atoms (axis 1)
#         in_axes=0
#     )(ae)

#     # Per-electron envelope: combine over atoms (same reduction as your σ version)
#     # You can choose sum or mean; we keep sum for drop-in compatibility.
#     env_scalar = jnp.sum(phase_ea, axis=1)  # shape (nelectron,)

#     # Broadcast to output_dim with trainable gain (gain is real; result is complex)
#     return env_scalar[:, None] * gain[None, :]

#   return envelopes.Envelope(envelopes.EnvelopeType.PRE_DETERMINANT, init, apply)


def make_magnetic_pair_zero_envelope_2d(
    lattice: jnp.ndarray,
    num_pair_zeros: int,
    *,
    use_nearest_image: bool = True,
    eps: float = 0.0,  # set >0 for numerical regularization if needed
) -> envelopes.Envelope:
  """Envelope = magnetic phase × even pair factor with trainable zeros.

  - Magnetic phase: for each electron coordinate r, we write
      r = r' + L,  L = n1 L1 + n2 L2,
    and multiply by η exp(i/2 r'×L), with
      η = +1 if (n1,n2) both even, else -1.

  - Pair factor: built from relative coordinates z_ij = (x_ij + i y_ij)/√2,
    and trainable zeros b_k in the complex plane. We use an *even* factor
      f_ij = ∏_k |z_ij - b_k|^2
    so that exchanging i ↔ j leaves it invariant.

  - The total pair Jastrow is distributed over electrons without using log:
      env_e[i] *= sqrt(f_ij), env_e[j] *= sqrt(f_ij)
    for each unordered pair (i<j). Then ∏_i env_e[i] = ∏_{i<j} f_ij.

  Args:
    lattice: (2,2) matrix, columns L1, L2.
    num_pair_zeros: number of trainable pair zeros b_k.
    use_nearest_image: if True, choose integer windings n by rounding t,
      else by floor (unwrapped to [0,1) cell).
    eps: small positive number optionally added inside |z_ij - b_k|^2 to
      regularize (eps=0 gives exact zeros but can be numerically harsher).

  Returns:
    ferminet.envelopes.Envelope with type PRE_DETERMINANT.
  """

  lattice = lattice.astype(jnp.float32)
  L1 = lattice[:, 0]
  L2 = lattice[:, 1]
  Linv = jnp.linalg.inv(lattice)

  # --------- helpers for magnetic phase / reduction ---------

  def _infer_rprime_and_phase(r_xy: jnp.ndarray) -> tuple[jnp.ndarray, jnp.complex64]:
    """Given r, compute reduced r' in fundamental cell and phase.

    r = t1 L1 + t2 L2, with t = Linv @ r.
    n = integer windings, L = n1 L1 + n2 L2, r' = r - L.

    Returns:
      r_prime: (2,) reduced coordinate in the chosen cell.
      phase: complex scalar η * exp(i/2 r'×L).
    """
    t = Linv @ r_xy  # fractional lattice coords (2,)

    if use_nearest_image:
      n = jnp.floor(t + 0.5)
    else:
      n = jnp.floor(t)

    n = jax.lax.stop_gradient(n).astype(jnp.int32)

    # L = n1 L1 + n2 L2
    L_vec = lattice @ n.astype(r_xy.dtype)
    r_prime = r_xy - L_vec

    # scalar cross product r' × L
    rprime_cross_L = r_prime[0] * L_vec[1] - r_prime[1] * L_vec[0]

    # η = +1 if n1,n2 even; else -1
    both_even = jnp.all((n & 1) == 0)
    eta_real = jnp.where(both_even, 1.0, -1.0).astype(r_xy.dtype)

    theta = 0.5 * rprime_cross_L
    phase = (eta_real + 0j) * jnp.exp(1j * theta.astype(r_xy.dtype))

    return r_prime.astype(jnp.float32), phase.astype(jnp.complex64)

  # --------- init ---------

  def init(natom: int, output_dims: Sequence[int], ndim: int = 2
           ) -> Sequence[Mapping[str, jnp.ndarray]]:
    del natom, ndim
    params = []
    # one set of zeros per output_dim block (same across electrons)
    for output_dim in output_dims:
      params.append({
          # trainable zeros in real 2D, shape (num_pair_zeros, 2)
          # small random init near origin
          "zeros_xy": 0.01 * jax.random.normal(
              jax.random.PRNGKey(0),
              (num_pair_zeros, 2),
              dtype=jnp.float32,
          ),
          "gain": jnp.ones((output_dim,), dtype=jnp.float32),
      })
    return params

  # --------- apply ---------

  def apply(*, ae: jnp.ndarray, r_ae: jnp.ndarray, r_ee: jnp.ndarray,
            zeros_xy: jnp.ndarray, gain: jnp.ndarray) -> jnp.ndarray:
    """ae: (ne, natom, 2), r_ee: (ne, ne, 2) → (ne, output_dim) complex."""
    del r_ae  # pair factor uses r_ee; electron positions from ae

    ae = ae.astype(jnp.float32)
    r_ee = r_ee.astype(jnp.float32)
    zeros_xy = zeros_xy.astype(jnp.float32)
    gain = gain.astype(jnp.float32)

    ne = ae.shape[0]

    # Electron positions: assume atom 0 at origin, so ae[:,0,:] = r_e
    r_e = ae[:, 0, :]  # (ne,2)

    # Reduce positions and get magnetic phase per electron
    r_prime_e, phase_e = jax.vmap(_infer_rprime_and_phase, in_axes=0, out_axes=(0, 0))(r_e)
    # Complex coordinates z_i = (x'+i y')/√2 for reduced positions
    z_e = (r_prime_e[:, 0] + 1j * r_prime_e[:, 1]) / jnp.sqrt(2.0)

    # Center zeros so their mean is zero (keeps them roughly in-cell)
    zeros_xy = zeros_xy - jnp.mean(zeros_xy, axis=0, keepdims=True)
    zeros_c = (zeros_xy[:, 0] + 1j * zeros_xy[:, 1]) / jnp.sqrt(2.0)  # (num_pair_zeros,)

    # Initialize per-electron pair envelope to 1
    env_e = jnp.ones((ne,), dtype=jnp.complex64)

    # Indices for unordered pairs i<j
    idx_i, idx_j = jnp.triu_indices(ne, k=1)

    def pair_body(env_e_carry, idx_pair):
      i = idx_i[idx_pair]
      j = idx_j[idx_pair]

      zij = z_e[i] - z_e[j]  # complex
      diffs = zij - zeros_c  # (num_pair_zeros,)
      mag2 = diffs.real**2 + diffs.imag**2
      if eps != 0.0:
        mag2 = mag2 + eps

      # Even pair factor: ∏_k |z_ij - b_k|^2
      f_ij = jnp.prod(mag2).astype(jnp.float32)

      # Distribute f_ij symmetrically over electrons (sqrt each)
      sqrt_f = jnp.sqrt(f_ij).astype(jnp.float32)

      env_e_new = env_e_carry.at[i].multiply(sqrt_f)
      env_e_new = env_e_new.at[j].multiply(sqrt_f)
      return env_e_new, None

    # If ne <= 1, no pairs; skip scan
    def do_pairs(env_e_init):
      n_pairs = idx_i.shape[0]
      return jax.lax.scan(pair_body, env_e_init, jnp.arange(n_pairs))[0]

    env_e = jax.lax.cond(
        ne > 1,
        do_pairs,
        lambda x: x,
        env_e,
    )

    # Combine magnetic phase and pair factor per electron
    env_scalar = env_e * phase_e  # (ne,) complex

    # Broadcast to output_dim and apply gain (real)
    gain_c = gain + 0j  # complex view
    return env_scalar[:, None] * gain_c[None, :]

  return envelopes.Envelope(envelopes.EnvelopeType.PRE_DETERMINANT, init, apply)


def make_magnetic_laughlin_envelope_2d(
    lattice: jnp.ndarray,
    *,
    use_nearest_image: bool = True,
) -> envelopes.Envelope:
  """Magnetic envelope × Laughlin Jastrow (z_i - z_j)^2 built from shifted positions.

  - Positions r_e are first reduced:
        r_e = r'_e + L_e,  L_e = n1 L1 + n2 L2,
    with integers (n1, n2) inferred from lattice@t = r.

  - Magnetic phase per electron:
        phase_i = η_i * exp(i/2 * r'_i × L_i),
    with η_i = +1 if (n1,n2) both even, else -1.

  - Laughlin Jastrow:
        z_i  = (x'_i + i y'_i)/√2  (from reduced r'_i)
        J    = ∏_{i<j} (z_i - z_j)^2

    We distribute it over electrons by
        env_e[i] = ∏_{j≠i} (z_i - z_j),
    so that ∏_i env_e[i] = ∏_{i<j} (z_i - z_j)^2 up to a global sign.

  Args:
    lattice: (2,2) matrix with columns L1, L2.
    use_nearest_image: if True, windings n from floor(t+0.5) (nearest image);
                       if False, from floor(t) (0..1-style cell).
  """

  lattice = lattice.astype(jnp.float32)
  L1 = lattice[:, 0]
  L2 = lattice[:, 1]
  Linv = jnp.linalg.inv(lattice)

  # --------- helper: reduce r and get magnetic phase ---------

  def _infer_rprime_and_phase(r_xy: jnp.ndarray) -> tuple[jnp.ndarray, jnp.complex64]:
    """Given r, compute reduced r' and magnetic phase η exp(i/2 r'×L)."""
    r_xy = r_xy.astype(jnp.float32)
    t = Linv @ r_xy  # fractional coords (2,)

    if use_nearest_image:
      n = jnp.floor(t + 0.5)
    else:
      n = jnp.floor(t)

    n = jax.lax.stop_gradient(n).astype(jnp.int32)

    # L = n1 L1 + n2 L2
    L_vec = lattice @ n.astype(jnp.float32)
    r_prime = r_xy - L_vec

    # scalar cross product r' × L
    rprime_cross_L = r_prime[0] * L_vec[1] - r_prime[1] * L_vec[0]

    # η = +1 if both n1,n2 even; else -1
    both_even = jnp.all((n & 1) == 0)
    eta_real = jnp.where(both_even, 1.0, -1.0).astype(jnp.float32)

    theta = -0.5 * rprime_cross_L
    phase = (eta_real + 0j) * jnp.exp(1j * theta.astype(jnp.float32))

    return r_prime.astype(jnp.float32), phase.astype(jnp.complex64)

  # --------- init ---------

  def init(natom: int, output_dims: Sequence[int], ndim: int = 2
           ) -> Sequence[Mapping[str, jnp.ndarray]]:
    del natom, ndim
    params = []
    for odim in output_dims:
      params.append({
          "gain": jnp.ones((odim,), dtype=jnp.float32),
      })
    return params

  # --------- apply ---------

  def apply(*, ae: jnp.ndarray, r_ae: jnp.ndarray, r_ee: jnp.ndarray,
            gain: jnp.ndarray) -> jnp.ndarray:
    """ae: (ne, natom, 2) → (ne, output_dim) complex.

    Assumes natom=1 dummy atom at origin, so ae[:,0,:] ≡ electron positions.
    """
    del r_ae, r_ee

    ae = ae.astype(jnp.float32)
    gain = gain.astype(jnp.float32)

    # electron positions (ne,2); atom 0 is dummy at origin
    r_e = ae[:, 0, :]  # (ne,2)
    ne = r_e.shape[0]

    # reduce positions and get magnetic phase per electron
    r_prime_e, phase_e = jax.vmap(_infer_rprime_and_phase, in_axes=0, out_axes=(0, 0))(r_e)

    # complex coordinates from reduced positions
    z_e = (r_prime_e[:, 0] + 1j * r_prime_e[:, 1]) / jnp.sqrt(jnp.float32(2.0))  # (ne,)

    # build Jastrow: env_e[i] = ∏_{j≠i} (z_i - z_j)
    if ne > 1:
      z_diff = z_e[:, None] - z_e[None, :]                  # (ne, ne)
      mask_offdiag = ~jnp.eye(ne, dtype=bool)               # True where i≠j
      factors = jnp.where(mask_offdiag, z_diff, 1.0 + 0j)   # 1 on diagonal
      env_e = jnp.prod(factors, axis=1)                     # (ne,)
    else:
      env_e = jnp.ones((ne,), dtype=jnp.complex64)

    # combine Jastrow with magnetic phase
    env_scalar = env_e * phase_e  # (ne,) complex

    # broadcast to output_dim and apply real gain
    gain_c = gain + 0j  # complex view
    return env_scalar[:, None] * gain_c[None, :]

  return envelopes.Envelope(envelopes.EnvelopeType.PRE_DETERMINANT, init, apply)


def make_magnetic_phase_envelope_2d(
    lattice: jnp.ndarray,
    *,
    use_nearest_image: bool = False,
) -> envelopes.Envelope:
  """Pure magnetic phase envelope in 2D with correct minimum-image cross term.

  Enforces ψ(r) → η exp(i/2 r'×L) ψ(r') under r = r' + L,
  with η = +1 iff L/2 is a lattice vector (n1,n2 both even), else −1.
  Args:
    lattice: (2,2) with columns L1, L2.
    use_nearest_image: pick windings by rounding (nearest image) if True,
      else by floor (unwrap-to-cell split).
  """
  L1 = lattice[:, 0]
  L2 = lattice[:, 1]
  Linv = jnp.linalg.inv(lattice)

  def _infer_windings_and_cross(r_xy: jnp.ndarray):
    """Return (n, L_vec, rprime_cross_L) for a single 2D vector r."""
    # fractional lattice coords t: r = t1 L1 + t2 L2
    t = Linv @ r_xy  # (2,)
    n = jnp.floor(t + 0.5) if use_nearest_image else jnp.floor(t)
    n = jax.lax.stop_gradient(n).astype(jnp.int32)

    # L = n1 L1 + n2 L2, r' = r - L
    L_vec = lattice @ n.astype(r_xy.dtype)        # (2,)
    r_prime = r_xy - L_vec                         # (2,)

    # r' × L (2D scalar cross)
    rprime_cross_L = r_prime[0] * L_vec[1] - r_prime[1] * L_vec[0]
    return n, L_vec, rprime_cross_L

  def _eta_from_n(n: jnp.ndarray) -> jnp.ndarray:
    """η = +1 iff n1 and n2 are both even (so L/2 is a lattice vector)."""
    both_even = jnp.all((n & 1) == 0)
    return jnp.where(both_even, 1.0, -1.0)

  # ---------- init ----------
  def init(natom: int, output_dims: Sequence[int], ndim: int = 2
           ) -> Sequence[Mapping[str, jnp.ndarray]]:
    del natom, ndim
    return [{'gain': jnp.ones((odim,))} for odim in output_dims]

  # ---------- apply ----------
  def apply(*, ae: jnp.ndarray, r_ae: jnp.ndarray, r_ee: jnp.ndarray,
            gain: jnp.ndarray) -> jnp.ndarray:
    del r_ae, r_ee
    dtype = ae.dtype

    def phase_one(r_xy: jnp.ndarray):
      n, _, rprime_cross_L = _infer_windings_and_cross(r_xy)

      # eta ∈ {+1, -1}  → explicitly complex to avoid any real-only paths
      eta_real = _eta_from_n(n).astype(r_xy.dtype)
      eta_c = eta_real + 0j                     # force complex

      # compute phase in complex as well
      theta = -0.5 * rprime_cross_L.astype(r_xy.dtype)
      phase = jnp.exp(1j * theta)               # complex64/128

      return eta_c * phase


    # (ne, na) complex
    phase_ea = jax.vmap(jax.vmap(phase_one, in_axes=0), in_axes=0)(ae)

    env_scalar = jnp.sum(phase_ea, axis=1)                      # complex
    gain_c = gain.astype(env_scalar.real.dtype) + 0j            # complex
    return env_scalar[:, None] * gain_c[None, :]

  return envelopes.Envelope(envelopes.EnvelopeType.PRE_DETERMINANT, init, apply)

def make_magnetic_phase_envelope_2d_mod(
    lattice: jnp.ndarray,
    *,
    centered_cell: bool = False,  # False: [0,1)×[0,1), True: [-1/2,1/2)×[-1/2,1/2)
) -> envelopes.Envelope:
  """Pure magnetic phase envelope in 2D using modulo to split r = r0 + L.

  Enforces ψ(r) → η * exp(i/2 r0×L) ψ(r0) for r = r0 + L,
  with η = +1 iff L/2 is a lattice vector (n1,n2 both even), else −1.

  Args:
    lattice: (2,2) with columns L1, L2.
    centered_cell: if True, reduce to [-1/2,1/2); else to [0,1).
  """
  L1 = lattice[:, 0]
  L2 = lattice[:, 1]
  Linv = jnp.linalg.inv(lattice)

  def _split_r_with_mod(r_xy: jnp.ndarray):
    """Return (n, L_vec, r0_cross_L) from a single 2D vector r."""
    # fractional lattice coords t: r = t1 L1 + t2 L2
    t = Linv @ r_xy  # shape (2,)

    if not centered_cell:
      # [0,1)-cell via modulo: t = q + r, q integer, r in [0,1)
      q, r = jnp.divmod(t, 1.0)         # q is floor(t), r = t - floor(t)
    else:
      # Centered cell: r ∈ [-1/2,1/2), q = floor(t + 1/2)
      q, r = jnp.divmod(t + 0.5, 1.0)
      r = r - 0.5

    # Integer windings n = q (safe: divmod uses floor-division semantics)
    n = jax.lax.stop_gradient(jnp.floor(q)).astype(jnp.int32)

    # L = n1 L1 + n2 L2, r0 = r1 L1 + r2 L2  (but we only need r0 in Cartesian)
    L_vec = lattice @ n.astype(r_xy.dtype)        # (2,)
    r0_cart = lattice @ r.astype(r_xy.dtype)      # (2,)

    # r0 × L (2D scalar cross)
    r0_cross_L = r0_cart[0] * L_vec[1] - r0_cart[1] * L_vec[0]
    return n, r0_cross_L

  def _eta_from_n(n: jnp.ndarray) -> jnp.ndarray:
    # η = +1 iff n1 and n2 are even (so L/2 is a lattice vector); else −1
    both_even = jnp.all((n & 1) == 0)
    return jnp.where(both_even, 1.0, -1.0)

  # ---------- init ----------
  def init(natom: int, output_dims: Sequence[int], ndim: int = 2):
    del natom, ndim
    return [{'gain': jnp.ones((odim,))} for odim in output_dims]

  # ---------- apply ----------
  def apply(*, ae: jnp.ndarray, r_ae: jnp.ndarray, r_ee: jnp.ndarray,
            gain: jnp.ndarray) -> jnp.ndarray:
    del r_ae, r_ee

    def phase_one(r_xy: jnp.ndarray):
      n, r0_cross_L = _split_r_with_mod(r_xy)
      eta = _eta_from_n(n).astype(r_xy.dtype)
      phase = jnp.exp(-0.5j * r0_cross_L.astype(r_xy.dtype))
      return (eta + 0j) * phase

    # (ne, na) complex
    phase_ea = jax.vmap(jax.vmap(phase_one, in_axes=0), in_axes=0)(ae)

    env_scalar = jnp.sum(phase_ea, axis=1)                 # complex
    gain_c = gain.astype(env_scalar.real.dtype) + 0j       # complex
    return env_scalar[:, None] * gain_c[None, :]

  return envelopes.Envelope(envelopes.EnvelopeType.PRE_DETERMINANT, init, apply)


def make_kpoints_2d(
    lattice: Union[jnp.ndarray, jnp.ndarray],
    spins: Tuple[int, int],
    min_kpoints: Optional[int] = None,
) -> jnp.ndarray:
    """Generates an array of reciprocal lattice vectors for 2D systems.

    Args:
        lattice: Matrix whose columns are the primitive lattice vectors of the
          system, shape (ndim, ndim). (Note that ndim=2 for 2D systems).
        spins: Tuple of the number of spin-up and spin-down electrons.
        min_kpoints: If specified, the number of kpoints which must be included in
          the output. The number of kpoints returned will be the
          first filled shell which is larger than this value. Defaults to None,
          which results in min_kpoints == sum(spins).

    Raises:
        ValueError: Fewer kpoints requested by min_kpoints than number of
          electrons in the system.

    Returns:
        jnp.ndarray, shape (nkpoints, ndim), an array of reciprocal lattice
        vectors sorted in ascending order according to length.
    """
    rec_lattice = 2 * jnp.pi * jnp.linalg.inv(lattice)
    # Calculate required no. of k points
    if min_kpoints is None:
        min_kpoints = sum(spins)
    elif min_kpoints < sum(spins):
        raise ValueError(
            'Number of kpoints must be equal or greater than number of electrons')

    dk = 1 + 1e-5
    # Generate ordinals of the lowest min_kpoints kpoints
    max_k = int(jnp.ceil(min_kpoints * dk)**(1 / 2.))
    ordinals = sorted(range(-max_k, max_k+1), key=abs)
    ordinals = jnp.asarray(list(itertools.product(ordinals, repeat=2)))

    kpoints = ordinals @ rec_lattice.T
    kpoints = jnp.asarray(sorted(kpoints, key=jnp.linalg.norm))
    k_norms = jnp.linalg.norm(kpoints, axis=1)

    return kpoints[k_norms <= k_norms[min_kpoints - 1] * dk]