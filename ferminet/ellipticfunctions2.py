import jax
import jax.numpy as jnp
from jax import lax
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




# ---------- small helpers ----------

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

def _precompute_theta_coeffs(tau, max_terms=20):
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


def _LLL_with_zeros_log_cached(z, zeros, consts) -> jnp.ndarray:
  """
  Uses log_weierstrass_sigma_cached 
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

##########

