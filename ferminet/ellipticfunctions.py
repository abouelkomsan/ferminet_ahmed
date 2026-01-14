import jax
from jax import lax
import jax.numpy as jnp
from typing import Sequence, Mapping

def to_cplx_divsqrt2(v2: jnp.ndarray) -> jnp.ndarray:
    """(x,y) → (x+iy)/√2 (LLL complex convention)."""
    return (v2[0] + 1j*v2[1]) / jnp.sqrt(2.0)

def eisenstein(L1com: jnp.ndarray, L2com: jnp.ndarray, nmax: int = 200) -> jnp.ndarray:
    """
    out = (2π^2/L1^2)*(1/6 + Σ_{n=1..nmax} 1/sin^2(nπ L2/L1)).
    """
    ratio = L2com / L1com
    n = jnp.arange(1, nmax+1, dtype=jnp.float64)
    s = jnp.sum(1.0 / (jnp.sin(jnp.pi*n*ratio)**2))
    return (2.0 * (jnp.pi**2) / (L1com*L1com)) * (1.0/6.0 + s)

def almost_modular(L1: jnp.ndarray, L2: jnp.ndarray, N_phi: jnp.ndarray, nmax: int = 200) -> jnp.ndarray:
    """
    almost = G2(L1com,L2com) - (conj(L1com)/L1com)/N_phi, with Li_com=(Lix+iLiy)/√2.
    """
    L1com = to_cplx_divsqrt2(L1).astype(jnp.complex128)
    L2com = to_cplx_divsqrt2(L2).astype(jnp.complex128)
    G2 = eisenstein(L1com, L2com, nmax=nmax)
    return G2 - (jnp.conj(L1com) / (N_phi * L1com))


def round_sg(x):
  """Discrete round with gradients stopped (both input and output)."""
  x = lax.stop_gradient(x)
  return lax.stop_gradient(jnp.rint(x))

def _xcispi(x):  # e^{i π x}
  x_c  = jnp.asarray(x, jnp.complex64)
  pi_c = jnp.asarray(jnp.pi, jnp.complex64)
  return jnp.exp(1j * pi_c * x_c)

def _kahan_add(acc, c, term):
  acc  = acc.astype(jnp.complex64)
  c    = c.astype(jnp.complex64)
  term = term.astype(jnp.complex64)
  y = term - c
  t = acc + y
  c_new = (t - acc) - y
  return t, c_new

def to_cplx_divsqrt2(v2: jnp.ndarray) -> jnp.ndarray:
  """(x,y) -> (x+iy)/sqrt(2) as complex64."""
  return ((v2[0].astype(jnp.float32) + 1j * v2[1].astype(jnp.float32))
          / jnp.sqrt(jnp.asarray(2.0, jnp.float32))).astype(jnp.complex64)


def _reduce_u_and_z(u, tau, w1):
  """Reduce u into central strip using SG-rounded m,n; return (u_red, z_red, log θ-shift)."""
  pi = jnp.asarray(jnp.pi, jnp.float32)
  u   = u.astype(jnp.complex64)
  tau = tau.astype(jnp.complex64)
  w1  = w1.astype(jnp.complex64)

  # choose integers n,m via STOP-GRADIENT round
  n = round_sg(jnp.imag(u) / (pi * jnp.imag(tau)))  # scalar (real)
  u1 = u - n.astype(jnp.complex64) * pi * tau
  m = round_sg(jnp.real(u1) / pi)
  u_red = u1 - m.astype(jnp.complex64) * pi

  # map back to z via z = (2 w1 / π) u
  z_red = (2.0 * w1 / pi.astype(jnp.complex64)) * u_red

  # exact θ1 multiplicative factor (must use u_red)
  log_fac_theta = (
      1j * pi.astype(jnp.complex64) * (m + n).astype(jnp.complex64)
      - 1j * (2.0 * n.astype(jnp.complex64) * u_red
              + (n.astype(jnp.complex64) ** 2)
                * pi.astype(jnp.complex64) * tau)
  )
  return u_red, z_red, log_fac_theta

# ---------- precompute θ-coefficients for fixed τ ----------

def _precompute_theta_coeffs(tau, max_terms=50):
  """Precompute θ₁ series coefficients and derivative coefficients for fixed τ."""
  tau = jnp.asarray(tau, jnp.complex64)
  q   = _xcispi(tau)
  pref = (2.0 * jnp.sqrt(jnp.sqrt(q))).astype(jnp.complex64)

  n = jnp.arange(max_terms, dtype=jnp.int32)        # 0..T-1
  k = (2 * n + 1).astype(jnp.float32)              # (T,)

  n_f = n.astype(jnp.float32)
  qpow = q ** (n_f * (n_f + 1.0))                  # q^{n(n+1)} (T,)

  alt = (1.0 - 2.0 * (n % 2)).astype(jnp.complex64)   # (-1)^n

  coeff_series = (pref * alt * qpow).astype(jnp.complex64)         # for θ1(u)
  coeff_d1     = (pref * alt * qpow * k.astype(jnp.complex64))     # for θ1'(0)
  coeff_d3     = (-pref * alt * qpow * (k.astype(jnp.complex64)**3))  # θ1'''(0)

  return {
      'k': k.astype(jnp.float32),
      'coeff_series': coeff_series,
      'coeff_d1': coeff_d1,
      'coeff_d3': coeff_d3,
  }

def _theta1_series_from_coeffs(u, theta_coeffs):
  """θ₁(u|τ) using precomputed coeffs."""
  u = u.astype(jnp.complex64)
  k  = theta_coeffs['k'].astype(jnp.complex64)          # (T,)
  cs = theta_coeffs['coeff_series']                     # (T,)
  sin_ku = jnp.sin(k * u)                               # (T,)
  return jnp.sum(cs * sin_ku)

def _theta_derivs0_from_coeffs(theta_coeffs):
  """θ₁'(0|τ) and θ₁'''(0|τ) from precomputed coeffs."""
  t1p0  = jnp.sum(theta_coeffs['coeff_d1'])
  t1ppp = jnp.sum(theta_coeffs['coeff_d3'])
  return t1p0, t1ppp

# ---------- log σ with cached geometry + reduce_u_and_z ----------

# def _log_weierstrass_sigma_cached(z, consts, small_u_thresh=1e-6):
#   """
#   Principal-branch log σ(z | w1,w2) with:
#     - u,z reduced via _reduce_u_and_z
#     - θ₁(u) evaluated from precomputed series coefficients.
#   """
#   z   = jnp.asarray(z, jnp.complex64)
#   w1  = consts['w1']
#   tau = consts['tau']
#   pi  = consts['pi']
#   c   = consts['c']
#   theta_coeffs = consts['theta_coeffs']

#   u = pi * z / (2.0 * w1)

#   # lattice reduction; note: this already includes exact quasiperiod θ-factor
#   u_red, z_red, log_fac_theta = _reduce_u_and_z(u, tau, w1)

#   # θ₁(u_red), θ₁'(0), Gaussian constant c already precomputed
#   theta_series = _theta1_series_from_coeffs(u_red, theta_coeffs)
#   t1p0 = consts['t1p0']

#   # log[ θ1(u_red)/θ1'(0) ] with small-u guard
#   log_theta_ratio = jax.lax.cond(
#       (jnp.abs(u_red) < small_u_thresh),
#       lambda a: jnp.log(a[0]),                       # log(u_red)
#       lambda a: jnp.log(_theta1_series_from_coeffs(a[0], a[3])) - jnp.log(a[2]),
#       operand=(u_red, tau, t1p0, theta_coeffs),
#   )

#   log_base = jnp.log(2.0 * w1 / pi) + log_theta_ratio + c * (z_red * z_red)
#   log_corr = log_fac_theta + c * (z * z - z_red * z_red)
#   return (log_base + log_corr).astype(jnp.complex64)

def _log_weierstrass_sigma_cached(z: jnp.ndarray,
                                 consts: Mapping[str, jnp.ndarray],
                                 small_u_thresh: float = 1e-6
                                 ) -> jnp.ndarray:
  """
  Principal-branch log σ(z | w1,w2) with reduction and exact θ-shift (complex64),
  using only jnp.where (no lax.cond) so that folx does not compute full Hessians.
  """
  # unpack precomputed constants (all stop_gradient’ed)
  w1   = consts['w1']        # complex64
  w2   = consts['w2']        # complex64
  tau  = consts['tau']       # complex64
  pi_c = consts['pi']        # complex64
  c    = consts['c']         # complex64
  t1p0 = consts['t1p0']      # complex64
  # if you stored theta_coeffs and use a series-from-coeffs helper:
  theta_coeffs = consts['theta_coeffs']

  z = z.astype(jnp.complex64)

  # u, reduction to central strip
  u = pi_c * z / (2.0 * w1)
  u_red, z_red, log_fac_theta = _reduce_u_and_z(u, tau, w1)

  # θ1(u_red | τ) via cached series coeffs
  theta_series = _theta1_series_from_coeffs(u_red, theta_coeffs)

  # two branches:
  #  small-u: θ1(u)/θ1'(0) ~ u → log ~ log(u)
  #  generic: log( θ1(u_red) / θ1'(0) )
  # add a tiny eps so we don't ever do log(0) numerically
  eps = jnp.asarray(1e-30, jnp.float32).astype(jnp.complex64)
  branch_small   = jnp.log(u_red + eps)
  branch_generic = jnp.log(theta_series) - jnp.log(t1p0)

  mask = (jnp.abs(u_red) < small_u_thresh)

  # jnp.where is elementwise and folx has rules for it
  log_theta_ratio = jnp.where(mask, branch_small, branch_generic)

  # assemble log σ
  log_base = jnp.log(2.0 * w1 / pi_c) + log_theta_ratio + c * (z_red * z_red)
  log_corr = log_fac_theta + c * (z * z - z_red * z_red)
  return (log_base + log_corr).astype(jnp.complex64)


def _flat_xy_to_complex(z_flat: jnp.ndarray) -> jnp.ndarray:
    """
    Convert flattened real coordinates (x1,y1,x2,y2,...,xNe,yNe)
    to complex LLL coordinates z_i = (x_i + i y_i)/sqrt(2).

    Args:
      z_flat: shape (2*Ne,), real32/real64.

    Returns:
      z: shape (Ne,), complex64.
    """
    z_flat = jnp.asarray(z_flat, jnp.float32)
    Ne = z_flat.shape[0] // 2
    coords = z_flat.reshape((Ne, 2))           # (Ne, 2) -> (x_i, y_i)
    # uses your to_cplx_divsqrt2 defined above
    z = jax.vmap(to_cplx_divsqrt2)(coords)     # (Ne,) complex64
    return z


# --- f(z) as before --------------------------------------------------------

def _log_f_torus(z: jnp.ndarray,
                 consts: Mapping[str, jnp.ndarray]) -> jnp.ndarray:
    """
    log f(z) on the torus, with
        f(z) = σ(z) * exp[-|z|^2 / (2 N_phi)].
    """
    z_c = jnp.asarray(z, jnp.complex64)

    log_sigma = _log_weierstrass_sigma_cached(z_c, consts)

    N_phi_f32 = consts['N_phi'].astype(jnp.float32)
    abs2 = (z_c * jnp.conj(z_c)).real.astype(jnp.float32)
    log_gauss_rel = -abs2 / (2.0 * N_phi_f32)
    log_gauss_rel = log_gauss_rel.astype(jnp.complex64)

    return (log_sigma + log_gauss_rel).astype(jnp.complex64)


_vlog_f_torus = jax.vmap(_log_f_torus, in_axes=(0, None))


# --- Laughlin with flattened coordinates -----------------------------------

def laughlin_log_psi_torus(
    z_flat: jnp.ndarray,              # shape (2*Ne,)
    alpha: jnp.ndarray,
    q: int,
    consts: Mapping[str, jnp.ndarray],
) -> jnp.ndarray:
    """
    log Ψ_Laughlin for flattened coordinates.

    Args:
      z_flat: shape (2*Ne,), real; (x1,y1,x2,y2,...,xNe,yNe).
      alpha: shape (q,), complex64, COM zeros {α_k}.
      q: integer Laughlin parameter.
      consts: sigma-geometry constants (must contain 'N_phi' etc.).

    Returns:
      complex64: log Ψ({z_i}).
    """
    # convert to complex LLL coordinates
    z = _flat_xy_to_complex(z_flat)            # (Ne,) complex64
    alpha = jnp.asarray(alpha, jnp.complex64)
    Ne = z.shape[0]

    # pairwise differences z_i - z_j, i<j
    zi = z[None, :]           # (1, Ne)
    zj = z[:, None]           # (Ne, 1)
    diffs = zi - zj           # (Ne, Ne), complex64

    # Instead of boolean indexing diffs[mask], we:
    # 1. flatten the matrix
    # 2. apply log f to every element
    # 3. reshape back to (Ne, Ne)
    # 4. sum only i<j via a numeric mask and jnp.where
    diffs_flat = diffs.reshape(-1)                     # (Ne*Ne,)
    log_f_flat = _vlog_f_torus(diffs_flat, consts)     # (Ne*Ne,)
    log_f_mat  = log_f_flat.reshape(Ne, Ne)            # (Ne, Ne)

    # mask for i<j
    pair_mask = jnp.triu(jnp.ones((Ne, Ne), dtype=jnp.bool_), k=1)
    zero = jnp.asarray(0.0, log_f_mat.dtype)
    log_pair = jnp.asarray(q, jnp.float32).astype(jnp.complex64) * jnp.sum(
        jnp.where(pair_mask, log_f_mat, zero)
    )

    # COM factor (unchanged)
    Z = jnp.sum(z)
    com_args = Z - alpha       # (q,)
    log_f_com = _vlog_f_torus(com_args, consts)
    log_com = jnp.sum(log_f_com)

    return (log_pair + log_com).astype(jnp.complex64)


def laughlin_psi_torus(
    z_flat: jnp.ndarray,              # <-- shape (2*Ne,)
    alpha: jnp.ndarray,
    q: int,
    consts: Mapping[str, jnp.ndarray],
) -> jnp.ndarray:
    """
    Ψ_Laughlin for flattened coordinates.

    Args:
      z_flat: shape (2*Ne,), real; (x1,y1,x2,y2,...,xNe,yNe).

    Returns:
      complex64: Ψ({z_i}).
    """
    return jnp.exp(laughlin_log_psi_torus(z_flat, alpha, q, consts))
# ---------- LLL with zeros using cached log σ ----------

# def _LLL_with_zeros_log_cached(z, zeros, consts):
#   """
#   log ψ(z) = Σ_a log σ(z - a)
#              + [(Σ conj(a)) z - conj(z) (Σ a)]/(2 Nφ)
#              - n_z |z|²/(2 Nφ) - (n_z/2) almost z²

#   All geometry (w1,w2,τ,c,Nφ,almost) comes from consts.
#   """
#   z = jnp.asarray(z, jnp.complex64)
#   zeros = jnp.asarray(zeros, jnp.complex64)
#   nz = zeros.size

#   # sum_a log σ(z - a)
#   def one_zero(a):
#     return _log_weierstrass_sigma_cached(z - a, consts)
#   log_sig = jax.lax.cond(
#       nz > 0,
#       lambda a: jnp.sum(jax.vmap(one_zero)(a)),
#       lambda a: jnp.array(0.0 + 0.0j, jnp.complex64),
#       operand=zeros,
#   )

#   Nphi_f32 = consts['N_phi']
#   Nphi_c   = Nphi_f32.astype(jnp.complex64)
#   almost   = consts['almost']          # complex64

#   A = jnp.sum(zeros).astype(jnp.complex64)
#   log_gauge = (jnp.conj(A) * z - jnp.conj(z) * A) / (2.0 * Nphi_c)

#   abs2_z = (z * jnp.conj(z)).real.astype(jnp.float32)
#   log_gauss = - nz * abs2_z / (2.0 * Nphi_f32)
#   log_gauss = log_gauss.astype(jnp.complex64)

#   log_quad = -0.5 * jnp.asarray(nz, jnp.float32) * almost * (z * z)

#   log_result = (log_sig + log_gauge + log_gauss + log_quad).astype(jnp.complex64)

#   # gentle clamp on Re(log ψ) for fp32 stability
#   def _safe_log(zlog, lo=-80.0, hi=80.0):
#     re = jnp.clip(jnp.real(zlog),
#                   a_min=jnp.asarray(lo, jnp.float32),
#                   a_max=jnp.asarray(hi, jnp.float32))
#     return (re.astype(jnp.float32) + 1j * jnp.imag(zlog).astype(jnp.float32)
#            ).astype(jnp.complex64)

#   return _safe_log(log_result)

def _LLL_with_zeros_log_cached(z, zeros, consts) -> jnp.ndarray:
  """
  Uses log_weierstrass_sigma_cached (no lax.cond).
  """
  w1   = consts['w1']
  w2   = consts['w2']
  Nphi = consts['N_phi']
  almost = consts['almost']

  # sum_a log σ(z - a)
  z_c = z.astype(jnp.complex64)
  zeros_c = jnp.asarray(zeros, jnp.complex64)
  log_sig = jnp.sum(
      jnp.array([_log_weierstrass_sigma_cached(z_c - a, consts)
                 for a in zeros_c],
                dtype=jnp.complex64)
  )

  A = jnp.sum(zeros_c).astype(jnp.complex64)

  Nphi_f = jnp.asarray(Nphi, jnp.float32)
  Nphi_c = Nphi_f.astype(jnp.complex64)

  log_gauge = (jnp.conj(A) * z_c - jnp.conj(z_c) * A) / (2.0 * Nphi_c)
  abs2 = (z_c * jnp.conj(z_c)).real.astype(jnp.float32)
  log_gauss = - zeros_c.size * abs2 / (2.0 * Nphi_f)
  log_gauss = log_gauss.astype(jnp.complex64)
  log_quad = -0.5 * jnp.asarray(zeros_c.size, jnp.float32) * consts['almost'] * (z_c * z_c)

  return (log_sig + log_gauge + log_gauss + log_quad).astype(jnp.complex64)

# --- theta' from cached coeffs: θ1'(u) = Σ coeff_series * k * cos(k u)
def _theta1_prime_from_coeffs(u, theta_coeffs):
  u = u.astype(jnp.complex64)
  k  = theta_coeffs['k'].astype(jnp.complex64)      # (T,)
  cs = theta_coeffs['coeff_series']                 # (T,)
  cos_ku = jnp.cos(k * u)
  return jnp.sum(cs * k * cos_ku)

# --- reduction that also returns the winding integer n (needed for d/dz log_fac_theta)
def _reduce_u_and_z_with_n(u, tau, w1):
  pi_f = jnp.asarray(jnp.pi, jnp.float32)
  u   = u.astype(jnp.complex64)
  tau = tau.astype(jnp.complex64)
  w1  = w1.astype(jnp.complex64)

  n = round_sg(jnp.imag(u) / (pi_f * jnp.imag(tau)))    # real scalar
  u1 = u - n.astype(jnp.complex64) * pi_f.astype(jnp.complex64) * tau
  m = round_sg(jnp.real(u1) / pi_f)
  u_red = u1 - m.astype(jnp.complex64) * pi_f.astype(jnp.complex64)

  z_red = (2.0 * w1 / pi_f.astype(jnp.complex64)) * u_red

  log_fac_theta = (
      1j * pi_f.astype(jnp.complex64) * (m + n).astype(jnp.complex64)
      - 1j * (2.0 * n.astype(jnp.complex64) * u_red
              + (n.astype(jnp.complex64) ** 2) * pi_f.astype(jnp.complex64) * tau)
  )
  return u_red, z_red, log_fac_theta, n

# --- ∂_z log σ(z | w1,w2) in the SAME reduced-strip convention as _log_weierstrass_sigma_cached
def _dlog_weierstrass_sigma_dz_cached(z: jnp.ndarray,
                                      consts: Mapping[str, jnp.ndarray],
                                      small_u_thresh: float = 1e-6) -> jnp.ndarray:
  w1   = consts['w1']          # complex64
  tau  = consts['tau']         # complex64
  pi_c = consts['pi']          # complex64
  c    = consts['c']           # complex64
  theta_coeffs = consts['theta_coeffs']

  z = z.astype(jnp.complex64)

  # u = π z / (2 w1)
  u = pi_c * z / (2.0 * w1)

  # reduce u, and get the winding integer n used in the exact θ-shift factor
  u_red, _, _, n = _reduce_u_and_z_with_n(u, tau, w1)

  # du/dz = π/(2 w1)
  du_dz = pi_c / (2.0 * w1)

  # θ1(u_red), θ1'(u_red)
  theta  = _theta1_series_from_coeffs(u_red, theta_coeffs)
  theta_p = _theta1_prime_from_coeffs(u_red, theta_coeffs)

  # ratio θ1'(u)/θ1(u), with small-u guard (θ1 ~ t1p0*u => ratio ~ 1/u)
  eps = jnp.asarray(1e-30, jnp.float32).astype(jnp.complex64)
  ratio_small   = 1.0 / (u_red + eps)
  ratio_generic = theta_p / (theta + eps)
  ratio = jnp.where(jnp.abs(u_red) < small_u_thresh, ratio_small, ratio_generic)

  # d/dz log θ1(u_red) = (θ'/θ) * du/dz
  dlog_theta = ratio * du_dz

  # d/dz log_fac_theta = - i * 2 n * du/dz   (since m,n are constants under stop_gradient)
  dlog_fac_theta = (-1j) * (2.0 * n.astype(jnp.complex64)) * du_dz

  # c-terms simplify exactly to 2 c z (independent of reduction)
  dlog_c = 2.0 * c * z

  return (dlog_theta + dlog_fac_theta + dlog_c).astype(jnp.complex64)

# --- ∂_z log ψ_LLL(z) for your _LLL_with_zeros_log_cached
def _dlog_LLL_dz_cached(z, zeros, consts) -> jnp.ndarray:
  z_c = z.astype(jnp.complex64)
  zeros_c = jnp.asarray(zeros, jnp.complex64)
  nz = zeros_c.size

  # Σ_a ∂_z log σ(z-a)
  dlog_sig = jnp.sum(jnp.array(
      [_dlog_weierstrass_sigma_dz_cached(z_c - a, consts) for a in zeros_c],
      dtype=jnp.complex64
  ))

  # other terms (from your log ψ definition)
  A = jnp.sum(zeros_c).astype(jnp.complex64)
  Nphi_f = jnp.asarray(consts['N_phi'], jnp.float32)
  Nphi_c = Nphi_f.astype(jnp.complex64)
  almost = consts['almost'].astype(jnp.complex64)

  # ∂_z [ (conj(A) z - conj(z) A)/(2Nphi) ] = conj(A)/(2Nphi)
  dlog_gauge = jnp.conj(A) / (2.0 * Nphi_c)

  # ∂_z [ -nz |z|^2/(2Nphi) ] = -nz * conj(z)/(2Nphi)
  dlog_gauss = - jnp.asarray(nz, jnp.float32).astype(jnp.complex64) * jnp.conj(z_c) / (2.0 * Nphi_c)

  # ∂_z [ -0.5 nz * almost * z^2 ] = -nz * almost * z
  dlog_quad = - jnp.asarray(nz, jnp.float32).astype(jnp.complex64) * almost * z_c

  return (dlog_sig + dlog_gauge + dlog_gauss + dlog_quad).astype(jnp.complex64)

# --- 1LL log-wavefunction in the same style
def _1LL_with_zeros_log_cached(z, zeros, consts,
                               small_u_thresh: float = 1e-6,
                               lo: float = -80.0,
                               hi: float = 80.0) -> jnp.ndarray:
  """
  1LL via raising operator (symmetric gauge, l_B=1 convention):
      ψ1(z) ∝ (1/√2) (conj(z) - 2 ∂_z) ψ0(z)

  Returns complex64 log ψ1 (principal branch). Uses eps to avoid log(0).
  """
  z_c = z.astype(jnp.complex64)

  log_psi0 = _LLL_with_zeros_log_cached(z_c, zeros, consts)
  dlog0    = _dlog_LLL_dz_cached(z_c, zeros, consts)

  pref = (jnp.conj(z_c) - 2.0 * dlog0) / jnp.sqrt(jnp.asarray(2.0, jnp.float32))
  eps_pref = jnp.asarray(1e-30, jnp.float32).astype(jnp.complex64)

  log_psi1 = log_psi0 + jnp.log(pref + eps_pref)

  # optional stability clamp on Re(log ψ)
  re = jnp.clip(jnp.real(log_psi1),
                a_min=jnp.asarray(lo, jnp.float32),
                a_max=jnp.asarray(hi, jnp.float32))
  log_psi1 = (re.astype(jnp.float32) + 1j * jnp.imag(log_psi1).astype(jnp.float32)).astype(jnp.complex64)
  return log_psi1


######## Magnetic Bloch Wavefunctions ########

def wsigma_cached(z: jnp.ndarray, consts) -> jnp.ndarray:
    """
    Weierstrass σ(z | w1,w2) from cached log σ.
    consts must contain the fields expected by _log_weierstrass_sigma_cached,
    in particular 'w1', 'w2', 'tau', 'pi', 'c', 't1p0', 'theta_coeffs'.
    """
    log_sigma = _log_weierstrass_sigma_cached(z, consts)
    return jnp.exp(log_sigma).astype(jnp.complex64)


def magnetic_bloch_wavefunction(
    r: jnp.ndarray,
    k: jnp.ndarray,
    a1: jnp.ndarray,
    a2: jnp.ndarray,
    sigma_consts,
    nmax_eisenstein: int = 200,
) -> jnp.ndarray:
    """
    JAX version of the Julia function `wavefunction(r,k,a1,a2)`:

        z        = (r_x + i r_y) / √2
        kcomplex = -i (k_x + i k_y) / √2
        a1com    = (a1_x + i a1_y) / √2
        a2com    = (a2_x + i a2_y) / √2
        almost   = almost_modular(a1,a2,1)

        ψ(r;k) = σ(z - kcomplex; a1/2,a2/2)
                 * exp(-almost * z^2 / 2)
                 * exp(conj(kcomplex) * z)
                 * exp(-|z|^2 / 2)
                 * exp(-|kcomplex|^2 / 2)

    Here `sigma_consts` should correspond to the lattice with half-periods
    (a1com/2, a2com/2), i.e. the same choice as in the Julia wsigma call.
    """
    # (x,y) -> (x + i y)/sqrt(2)
    z = to_cplx_divsqrt2(r).astype(jnp.complex64)

    # complex momentum: -i (k_x + i k_y)/sqrt(2)
    kcomplex = (-1j * to_cplx_divsqrt2(k)).astype(jnp.complex64)

    # almost-modular coefficient for N_phi = 1
    almost = almost_modular(a1, a2, jnp.asarray(1.0, jnp.float32),
                            nmax=nmax_eisenstein)
    almost = almost.astype(jnp.complex64)

    # σ(z - kcomplex | a1/2, a2/2) using cached log-sigma
    z_minus_k = (z - kcomplex).astype(jnp.complex64)
    sigma_val = wsigma_cached(z_minus_k, sigma_consts)  # complex64

    # exp(-almost * z^2 / 2)
    phase_quad = jnp.exp(-0.5 * almost * (z * z))

    # exp(conj(kcomplex) * z)
    phase_k = jnp.exp(jnp.conj(kcomplex) * z)

    # Gaussian exp(-|z|^2 / 2)
    abs2_z = (z * jnp.conj(z)).real.astype(jnp.float32)
    gauss_z = jnp.exp(-0.5 * abs2_z).astype(jnp.complex64)

    # exp(-|kcomplex|^2 / 2) — overall k-dependent normalization factor
    abs2_k = (kcomplex * jnp.conj(kcomplex)).real.astype(jnp.float32)
    gauss_k = jnp.exp(-0.5 * abs2_k).astype(jnp.complex64)

    # final wavefunction
    return (sigma_val * phase_quad * phase_k * gauss_z * gauss_k).astype(jnp.complex64)

########## Symmetric-gauge LLL and magnetic translations ########

import jax
import jax.numpy as jnp

def symmetric_gauge_LLL_plane(r: jnp.ndarray) -> jnp.ndarray:
    """
    Symmetric-gauge LLL ground-state orbital on the plane, centered at the origin.

    Args:
      r: shape (2,), float32 (x, y).

    Returns:
      complex64: ψ_0(r) ∝ exp(-|z|^2 / 2), with z = (x+iy)/√2.
    """
    r = jnp.asarray(r, jnp.float32)
    x, y = r[0], r[1]
    z = (x + 1j * y) / jnp.sqrt(jnp.asarray(2.0, jnp.float32))
    abs2_z = (z * jnp.conj(z)).real.astype(jnp.float32)
    return jnp.exp(-0.5 * abs2_z).astype(jnp.complex64)


def magnetic_translation_symmetric(
    r: jnp.ndarray,
    L1: jnp.ndarray,
    L2: jnp.ndarray,
    n1: jnp.ndarray,
    n2: jnp.ndarray,
    base_orbital_fn = symmetric_gauge_LLL_plane,
) -> jnp.ndarray:
    """
    Magnetic translation in symmetric gauge for R = n1 L1 + n2 L2:

        [T_R ψ](r) = exp(i/2 * (R × r)) * exp(i π n1 n2) * ψ(r + R),

    where R × r = R_x r_y - R_y r_x.

    Args:
      r: position, shape (2,), float32.
      L1, L2: primitive supercell vectors, each shape (2,), float32.
      n1, n2: integers (can be scalar jnp.int32 or float32) such that R = n1 L1 + n2 L2.
      base_orbital_fn: χ(r) returning LLL orbital at r.

    Returns:
      complex64: (T_R χ)(r).
    """
    r  = jnp.asarray(r,  jnp.float32)
    L1 = jnp.asarray(L1, jnp.float32)
    L2 = jnp.asarray(L2, jnp.float32)

    n1 = jnp.asarray(n1, jnp.float32)
    n2 = jnp.asarray(n2, jnp.float32)

    # R = n1 L1 + n2 L2
    R = n1 * L1 + n2 * L2  # shape (2,)

    # R × r = R_x r_y - R_y r_x
    cross = R[0] * r[1] - R[1] * r[0]
    phase_B = jnp.exp(0.5j * cross.astype(jnp.complex64))  # exp(i/2 R×r)

    # extra torus factor exp(i π n1 n2)
    phase_torus = jnp.exp(1j * jnp.pi * (n1 + n2).astype(jnp.complex64))

    phase = (phase_B * phase_torus).astype(jnp.complex64)

    r_shift = r + R
    psi_shift = base_orbital_fn(r_shift)

    return (phase * psi_shift).astype(jnp.complex64)

def symmetric_gauge_periodic_LLL(
    r: jnp.ndarray,
    L1: jnp.ndarray,
    L2: jnp.ndarray,
    nmax: int = 2,
) -> jnp.ndarray:
    """
    Periodic LLL-like orbital built from symmetric-gauge plane LLL by
    summing magnetic translations over supercell vectors:

        Ψ(r) = Σ_{n1=-nmax..nmax} Σ_{n2=-nmax..nmax}
               exp(i/2 * (R_{n1,n2} × r)) exp(i π n1 n2) ψ_0(r + R_{n1,n2}),

        R_{n1,n2} = n1 L1 + n2 L2.

    Args:
      r: real-space position, shape (2,), float32 (x,y).
      L1, L2: supercell primitive vectors, each shape (2,), float32.
      nmax: integer cutoff for |n1|, |n2| in the double sum.

    Returns:
      complex64 scalar: Ψ(r).
    """
    r  = jnp.asarray(r,  jnp.float32)
    L1 = jnp.asarray(L1, jnp.float32)
    L2 = jnp.asarray(L2, jnp.float32)

    n_vals = jnp.arange(-nmax, nmax + 1, dtype=jnp.int32)
    n1_grid, n2_grid = jnp.meshgrid(n_vals, n_vals, indexing="ij")

    n1_flat = n1_grid.reshape(-1)  # (N^2,)
    n2_flat = n2_grid.reshape(-1)

    def one_term(n1_i, n2_i):
        return magnetic_translation_symmetric(
            r=r,
            L1=L1,
            L2=L2,
            n1=n1_i,
            n2=n2_i,
            base_orbital_fn=symmetric_gauge_LLL_plane,
        )

    contribs = jax.vmap(one_term)(n1_flat, n2_flat)  # (N^2,)
    psi = jnp.sum(contribs)
    return psi.astype(jnp.complex64)


def symmetric_gauge_periodic_LLL_mapped(
    r: jnp.ndarray,
    L1: jnp.ndarray,
    L2: jnp.ndarray,
    nmax: int = 2,
) -> jnp.ndarray:
    """
    Periodic LLL block which:
      1. Maps r -> r0 in the fundamental cell.
      2. Evaluates symmetric_gauge_periodic_LLL(r0).
      3. Multiplies by the appropriate magnetic phase.

    This way we *never* evaluate the building block outside the supercell.
    """
    r0, log_phase = map_r_to_cell_and_phase(r, L1, L2)
    psi_cell = symmetric_gauge_periodic_LLL(r0, L1, L2, nmax=nmax)
    return (jnp.exp(log_phase) * psi_cell).astype(jnp.complex64)

def symmetric_gauge_periodic_LLL_with_zeros_mapped(
    r: jnp.ndarray,
    zeros_xy: jnp.ndarray,
    L1: jnp.ndarray,
    L2: jnp.ndarray,
    nmax: int = 2,
) -> jnp.ndarray:
    """
    Periodic symmetric-gauge LLL with a set of zeros {a_i}, keeping all
    evaluations inside the fundamental supercell:

        Ψ(r) = ∏_i Ψ_block(r - a_i),

    where Ψ_block is symmetric_gauge_periodic_LLL evaluated at the mapped
    coordinate (r - a_i) → r0_i, and we multiply by the corresponding
    magnetic phase for that winding.

    Args:
      r:         electron position, shape (2,), float32.
      zeros_xy:  zero positions a_i in real space, shape (n_zero, 2) or (2,).
      L1, L2:    supercell vectors, shape (2,).
      nmax:      translation cutoff for the periodic block.

    Returns:
      complex64: Ψ(r) with zeros at positions {a_i}, with all blocks evaluated
                 at coordinates in the fundamental cell.
    """
    r = jnp.asarray(r, jnp.float32)
    zeros_xy = jnp.asarray(zeros_xy, jnp.float32)

    if zeros_xy.ndim == 1:
        zeros_xy = zeros_xy[None, :]  # (1, 2)

    def one_zero(a_i):
        # relative coordinate r_rel = r - a_i
        r_rel = r - a_i

        # map r_rel into the cell and pick up the magnetic phase
        r0_i, log_phase_i = map_r_to_cell_and_phase(r_rel, L1, L2)

        # periodic LLL block at r0_i
        psi_cell_i = symmetric_gauge_periodic_LLL(r0_i, L1, L2, nmax=nmax)

        # total factor for this zero
        return jnp.exp(log_phase_i) * psi_cell_i

    vals = jax.vmap(one_zero)(zeros_xy)   # (n_zero,)
    psi = jnp.prod(vals)
    return psi.astype(jnp.complex64)


def map_r_to_cell_and_phase(
    r: jnp.ndarray,
    L1: jnp.ndarray,
    L2: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Decompose r = r0 + n1 L1 + n2 L2, with r0 in the fundamental supercell,
    and return (r0, log_phase) where log_phase encodes the magnetic
    boundary-condition phase for that lattice winding.

      r: shape (2,), float32
      L1, L2: shape (2,), float32

    Returns:
      r0:       (2,), mapped into unit cell
      log_phase: complex64 scalar, so that
                 ψ(r) = exp(log_phase) * ψ_cell(r0)
    """
    r  = jnp.asarray(r, jnp.float32)
    L1 = jnp.asarray(L1, jnp.float32)
    L2 = jnp.asarray(L2, jnp.float32)

    lattice = jnp.stack([L1, L2], axis=1).astype(jnp.float32)   # (2, 2)
    Linv    = jnp.linalg.inv(lattice)

    # fractional coordinates t = n + frac, with n ∈ Z^2
    t = Linv @ r                       # (2,)
    n = jnp.floor(t)                   # (2,)
    n_sg = lax.stop_gradient(n)        # integer-like, no grads

    # R = n1 L1 + n2 L2
    R_vec = lattice @ n_sg.astype(jnp.float32)   # (2,)
    r0    = r - R_vec                             # reduced coordinate in cell

    # Magnetic phase (symmetric gauge): something like exp(i θ), θ ∝ r0 × R
    # You can tweak the sign if you want to match an existing convention.
    r0_cross_R = r0[0] * R_vec[1] - r0[1] * R_vec[0]
    theta = -0.5 * r0_cross_R                    # float32

    # Optional ±1 parity factor depending on winding parity
    n_int = n_sg.astype(jnp.int32)
    both_even = jnp.all((n_int & 1) == 0)
    eta = jnp.where(both_even,
                    jnp.asarray(1.0, jnp.float32),
                    jnp.asarray(-1.0, jnp.float32))
    eta_c = eta.astype(jnp.complex64)

    log_phase = jnp.log(eta_c) + 1j * theta.astype(jnp.complex64)
    return r0, log_phase