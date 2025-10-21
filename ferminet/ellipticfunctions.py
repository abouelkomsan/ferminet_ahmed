# # ================== Weierstrass sigma σ(z|w1,w2): stable & direct ==================
# import jax.numpy as jnp

# # def _c128(x):
# #     x = jnp.asarray(x)
# #     if x.dtype == jnp.complex64:   return x.astype(jnp.complex128)
# #     if x.dtype == jnp.float32:     return x.astype(jnp.float64) + 0j
# #     if jnp.issubdtype(x.dtype, jnp.floating): return x + 0j
# #     return x

# def _xcispi(x):  # e^{i π x}
#     # x = _c128(x)
#     return jnp.exp(1j * jnp.pi * x)

# # θ1(u' + mπ + nπτ) = (-1)^{m+n} exp[-i (2n u' + n^2 π τ)] θ1(u')
# def _reduce_u_and_z(u, tau, w1):
#     pi = jnp.pi
#     # choose integers n,m to center u' in the fundamental strip
#     n = jnp.rint(jnp.imag(u) / (pi * jnp.imag(tau)))
#     u1 = u - n * pi * tau
#     m = jnp.rint(jnp.real(u1) / pi)
#     u_red = u1 - m * pi
#     z_red = (2.0 * w1 / pi) * u_red
#     # exact θ1 multiplicative factor (MUST use u_red)
#     log_fac_theta = 1j * pi * (m + n) - 1j * (2.0 * n * u_red + n * n * pi * tau)
#     return u_red, z_red, log_fac_theta

# # ---- compensated (Kahan) summation helper ----
# def _kahan_add(acc, c, term):
#     y = term - c
#     t = acc + y
#     c_new = (t - acc) - y
#     return t, c_new

# # θ1(u|τ) = 2 q^{1/4} Σ_{n=0}∞ (-1)^n q^{n(n+1)} sin((2n+1)u)
# def _theta1_series(u, tau, max_terms=20000, rtol=1e-16):
#     q = _xcispi(tau)
#     pref = 2.0 * jnp.sqrt(jnp.sqrt(q))
#     s = 0.0 + 0.0j; c = 0.0 + 0.0j
#     alt = 1.0 + 0.0j
#     qpow = 1.0 + 0.0j  # q^{n(n+1)} at n=0
#     for n in range(max_terms):
#         k = 2*n + 1
#         term = alt * qpow * jnp.sin(k * u)
#         s, c = _kahan_add(s, c, term)
#         if jnp.abs(term) <= rtol * (jnp.abs(s) + 1.0):
#             break
#         alt = -alt
#         qpow = qpow * (q ** (2*n + 2))  # -> q^{(n+1)(n+2)}
#     return pref * s

# # θ1'(0|τ) = 2 q^{1/4} Σ (-1)^n q^{n(n+1)} (2n+1)
# def _theta1_d1_0(tau, max_terms=20000, rtol=1e-16):
#     q = _xcispi(tau)
#     pref = 2.0 * jnp.sqrt(jnp.sqrt(q))
#     s = 0.0 + 0.0j; c = 0.0 + 0.0j
#     alt = 1.0 + 0.0j; qpow = 1.0 + 0.0j
#     for n in range(max_terms):
#         k = 2*n + 1
#         term = alt * qpow * k
#         s, c = _kahan_add(s, c, term)
#         if jnp.abs(term) <= rtol * (jnp.abs(s) + 1.0):
#             break
#         alt = -alt
#         qpow = qpow * (q ** (2*n + 2))
#     return pref * s

# # θ1'''(0|τ) = -2 q^{1/4} Σ (-1)^n q^{n(n+1)} (2n+1)^3
# def _theta1_d3_0(tau, max_terms=20000, rtol=1e-16):
#     q = _xcispi(tau)
#     pref = -2.0 * jnp.sqrt(jnp.sqrt(q))
#     s = 0.0 + 0.0j; c = 0.0 + 0.0j
#     alt = 1.0 + 0.0j; qpow = 1.0 + 0.0j
#     for n in range(max_terms):
#         k = 2*n + 1
#         term = alt * qpow * (k ** 3)
#         s, c = _kahan_add(s, c, term)
#         if jnp.abs(term) <= rtol * (jnp.abs(s) + 1.0):
#             break
#         alt = -alt
#         qpow = qpow * (q ** (2*n + 2))
#     return pref * s

# def weierstrass_sigma(z, w1, w2, small_u_thresh=1e-12):
#     """
#     Direct Weierstrass σ(z | w1, w2), τ = w2/w1 with Im(τ) > 0.
#     - Reduces z into the fundamental cell and evaluates there.
#     - Restores the exact value with the correct θ-shift factor (uses u_red).
#     - σ(0) = 0 exactly.
#     """
#     # z = _c128(z); w1 = _c128(w1); w2 = _c128(w2)

#     # exact zero
#     if bool((jnp.real(z) == 0.0) & (jnp.imag(z) == 0.0)):
#         return jnp.array(0.0 + 0.0j, dtype=jnp.complex128)

#     tau = w2 / w1  # ensure Im(tau) > 0 in your inputs
#     u   = jnp.pi * z / (2.0 * w1)

#     # reduce to central strip
#     u_red, z_red, log_fac_theta = _reduce_u_and_z(u, tau, w1)

#     # θ-derivatives at 0
#     t1p0   = _theta1_d1_0(tau)
#     t1ppp0 = _theta1_d3_0(tau)

#     # *** CORRECT gaussian constant (matches reference):  c = -(π^2/(24 w1^2)) * θ1'''(0)/θ1'(0)
#     c = - (jnp.pi * jnp.pi) / (24.0 * w1 * w1) * (t1ppp0 / t1p0)

#     # θ1(u_red)/θ1'(0) with small-u guard
#     if bool(jnp.abs(u_red) < small_u_thresh):
#         theta_ratio = u_red
#     else:
#         theta_ratio = _theta1_series(u_red, tau) / t1p0

#     # base at reduced point
#     sigma_base = (2.0 * w1 / jnp.pi) * theta_ratio * jnp.exp(c * (z_red * z_red))
#     # exact correction back to z (θ shift + gaussian correction)
#     corr = jnp.exp(log_fac_theta) * jnp.exp(c * (z * z - z_red * z_red))
#     out = sigma_base * corr
#     return out


# def log_weierstrass_sigma(z, w1, w2, small_u_thresh=1e-12):
#     """
#     Principal-branch log σ(z | w1, w2), τ = w2/w1 with Im(τ) > 0.

#     Stable for large |z| via lattice reduction (evaluate at reduced z' and
#     restore with exact quasiperiod factor). Uses the correct Gaussian constant:
#         c = -(π^2 / (24 w1^2)) * θ1'''(0|τ) / θ1'(0|τ)

#     Returns:
#       complex128: log σ(z). At z=0 returns -inf + 0j exactly.
#     """
#     # z  = _c128(z)
#     # w1 = _c128(w1)
#     # w2 = _c128(w2)

#     # exact zero
#     # if bool((jnp.real(z) == 0.0) & (jnp.imag(z) == 0.0)):
#     #     return jnp.array(-jnp.inf + 0.0j, dtype=jnp.complex64)

#     tau = w2 / w1  # ensure Im(tau) > 0 in your inputs
#     u   = jnp.pi * z / (2.0 * w1)

#     # reduce (u, z) into central strip & get exact θ1 multiplier (uses u_red!)
#     u_red, z_red, log_fac_theta = _reduce_u_and_z(u, tau, w1)

#     # θ1'(0), θ1'''(0) and Gaussian constant
#     t1p0   = _theta1_d1_0(tau)
#     t1ppp0 = _theta1_d3_0(tau)
#     c = - (jnp.pi * jnp.pi) / (24.0 * w1 * w1) * (t1ppp0 / t1p0)

#     # log[ θ1(u_red) / θ1'(0) ] with small-u guard
#     if bool(jnp.abs(u_red) < small_u_thresh):
#         # log θ1(u) ≈ log θ1'(0) + log u  => ratio log = log u
#         log_theta_ratio = jnp.log(u_red)
#     else:
#         log_theta_ratio = jnp.log(_theta1_series(u_red, tau)) - jnp.log(t1p0)

#     # log σ at reduced point + exact correction back to z
#     log_base = jnp.log(2.0 * w1 / jnp.pi) + log_theta_ratio + c * (z_red * z_red)
#     log_corr = log_fac_theta + c * (z * z - z_red * z_red)

#     out = log_base + log_corr
#     return out

# # ================== Weierstrass sigma σ(z|w1,w2): stable & direct (JAX-safe) ==================
# # ================== Weierstrass sigma σ(z|w1,w2): stable & direct (JAX-safe, minimal edits) ==================

# ================== Weierstrass sigma σ(z|w1,w2): JAX-compatible ==================
# ================== Weierstrass sigma σ(z|w1,w2): JAX-compatible & optimized ==================
import jax.numpy as jnp
from jax import lax

def _xcispi(x):  # e^{i π x}
    return jnp.exp(1j * jnp.pi * x)

def _reduce_u_and_z(u, tau, w1):
    pi = jnp.pi
    n = jnp.rint(jnp.imag(u) / (pi * jnp.imag(tau)))
    u1 = u - n * pi * tau
    m = jnp.rint(jnp.real(u1) / pi)
    u_red = u1 - m * pi
    z_red = (2.0 * w1 / pi) * u_red
    log_fac_theta = 1j * pi * (m + n) - 1j * (2.0 * n * u_red + n * n * pi * tau)
    return u_red, z_red, log_fac_theta

# Vectorized theta series computation - MUCH faster!
def _theta1_series_vectorized(u, tau, max_terms=100):
    """Vectorized version - computes all terms at once"""
    q = _xcispi(tau)
    pref = 2.0 * jnp.sqrt(jnp.sqrt(q))
    
    # Vectorized computation
    n = jnp.arange(max_terms)
    k = 2 * n + 1
    alt = (-1.0) ** n
    
    # Compute q^{n(n+1)} efficiently
    exponents = n * (n + 1)
    qpow = q ** exponents
    
    terms = alt * qpow * jnp.sin(k * u)
    s = jnp.sum(terms)
    
    return pref * s

def _theta1_d1_0_vectorized(tau, max_terms=100):
    """Vectorized θ1'(0) computation"""
    q = _xcispi(tau)
    pref = 2.0 * jnp.sqrt(jnp.sqrt(q))
    
    n = jnp.arange(max_terms)
    k = 2 * n + 1
    alt = (-1.0) ** n
    exponents = n * (n + 1)
    qpow = q ** exponents
    
    terms = alt * qpow * k
    s = jnp.sum(terms)
    
    return pref * s

def _theta1_d3_0_vectorized(tau, max_terms=100):
    """Vectorized θ1'''(0) computation"""
    q = _xcispi(tau)
    pref = -2.0 * jnp.sqrt(jnp.sqrt(q))
    
    n = jnp.arange(max_terms)
    k = 2 * n + 1
    alt = (-1.0) ** n
    exponents = n * (n + 1)
    qpow = q ** exponents
    
    terms = alt * qpow * (k ** 3)
    s = jnp.sum(terms)
    
    return pref * s

def weierstrass_sigma(z, w1, w2, small_u_thresh=1e-12, max_terms=100):
    """
    Fast JAX-compatible Weierstrass σ(z | w1, w2).
    
    Args:
        z: Point at which to evaluate
        w1, w2: Period lattice generators (Im(w2/w1) > 0)
        small_u_thresh: Threshold for small-u approximation
        max_terms: Number of terms in theta series (default 100 is usually sufficient)
    """
    tau = w2 / w1
    u = jnp.pi * z / (2.0 * w1)
    
    u_red, z_red, log_fac_theta = _reduce_u_and_z(u, tau, w1)
    
    t1p0 = _theta1_d1_0_vectorized(tau, max_terms)
    t1ppp0 = _theta1_d3_0_vectorized(tau, max_terms)
    
    c = -(jnp.pi * jnp.pi) / (24.0 * w1 * w1) * (t1ppp0 / t1p0)
    
    # Small-u guard using lax.cond
    def small_u_case(_):
        return u_red
    
    def large_u_case(_):
        return _theta1_series_vectorized(u_red, tau, max_terms) / t1p0
    
    theta_ratio = lax.cond(
        jnp.abs(u_red) < small_u_thresh,
        small_u_case,
        large_u_case,
        operand=None
    )
    
    sigma_base = (2.0 * w1 / jnp.pi) * theta_ratio * jnp.exp(c * (z_red * z_red))
    corr = jnp.exp(log_fac_theta) * jnp.exp(c * (z * z - z_red * z_red))
    out = sigma_base * corr
    
    is_zero = (jnp.real(z) == 0.0) & (jnp.imag(z) == 0.0)
    return jnp.where(is_zero, 0.0 + 0.0j, out)


def log_weierstrass_sigma(z, w1, w2, small_u_thresh=1e-12, max_terms=100):
    """
    Fast JAX-compatible log σ(z | w1, w2).
    
    Args:
        z: Point at which to evaluate
        w1, w2: Period lattice generators (Im(w2/w1) > 0)
        small_u_thresh: Threshold for small-u approximation
        max_terms: Number of terms in theta series (default 100 is usually sufficient)
    """
    tau = w2 / w1
    u = jnp.pi * z / (2.0 * w1)
    
    u_red, z_red, log_fac_theta = _reduce_u_and_z(u, tau, w1)
    
    t1p0 = _theta1_d1_0_vectorized(tau, max_terms)
    t1ppp0 = _theta1_d3_0_vectorized(tau, max_terms)
    c = -(jnp.pi * jnp.pi) / (24.0 * w1 * w1) * (t1ppp0 / t1p0)
    
    def small_u_case(_):
        return jnp.log(u_red)
    
    def large_u_case(_):
        return jnp.log(_theta1_series_vectorized(u_red, tau, max_terms)) - jnp.log(t1p0)
    
    log_theta_ratio = lax.cond(
        jnp.abs(u_red) < small_u_thresh,
        small_u_case,
        large_u_case,
        operand=None
    )
    
    log_base = jnp.log(2.0 * w1 / jnp.pi) + log_theta_ratio + c * (z_red * z_red)
    log_corr = log_fac_theta + c * (z * z - z_red * z_red)
    
    out = log_base + log_corr
    
    is_zero = (jnp.real(z) == 0.0) & (jnp.imag(z) == 0.0)
    return jnp.where(is_zero, -jnp.inf + 0.0j, out)


# ============ Batched versions for even better performance ============

def weierstrass_sigma_batched(z_array, w1, w2, small_u_thresh=1e-12, max_terms=100):
    """
    Batched version for computing σ at multiple points efficiently.
    
    Args:
        z_array: Array of points (shape: [n,])
        w1, w2: Period lattice generators
        
    Returns:
        Array of σ(z) values (shape: [n,])
    """
    return jax.vmap(
        lambda z: weierstrass_sigma(z, w1, w2, small_u_thresh, max_terms)
    )(z_array)


def log_weierstrass_sigma_batched(z_array, w1, w2, small_u_thresh=1e-12, max_terms=100):
    """
    Batched version for computing log σ at multiple points efficiently.
    
    Args:
        z_array: Array of points (shape: [n,])
        w1, w2: Period lattice generators
        
    Returns:
        Array of log σ(z) values (shape: [n,])
    """
    return jax.vmap(
        lambda z: log_weierstrass_sigma(z, w1, w2, small_u_thresh, max_terms)
    )(z_array)
