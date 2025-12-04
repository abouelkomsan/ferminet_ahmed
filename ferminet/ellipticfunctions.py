# # ================== Weierstrass sigma σ(z|w1,w2): stable & direct ==================
# import jax
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
# def _theta1_series(u, tau, max_terms=10, rtol=1e-16):
#     q = _xcispi(tau)
#     pref = 2.0 * jnp.sqrt(jnp.sqrt(q))
#     s = 0.0 + 0.0j; c = 0.0 + 0.0j
#     alt = 1.0 + 0.0j
#     qpow = 1.0 + 0.0j  # q^{n(n+1)} at n=0
#     for n in range(max_terms):
#         k = 2*n + 1
#         term = alt * qpow * jnp.sin(k * u)
#         s, c = _kahan_add(s, c, term)
#         # if jnp.abs(term) <= rtol * (jnp.abs(s) + 1.0):
#         #     print('Breaking at n = ', n)
#         #     break
#         alt = -alt
#         qpow = qpow * (q ** (2*n + 2))  # -> q^{(n+1)(n+2)}
#     return pref * s

# # θ1'(0|τ) = 2 q^{1/4} Σ (-1)^n q^{n(n+1)} (2n+1)
# def _theta1_d1_0(tau, max_terms=10, rtol=1e-16):
#     q = _xcispi(tau)
#     pref = 2.0 * jnp.sqrt(jnp.sqrt(q))
#     s = 0.0 + 0.0j; c = 0.0 + 0.0j
#     alt = 1.0 + 0.0j; qpow = 1.0 + 0.0j
#     for n in range(max_terms):
#         k = 2*n + 1
#         term = alt * qpow * k
#         s, c = _kahan_add(s, c, term)
#         # if jnp.abs(term) <= rtol * (jnp.abs(s) + 1.0):
#         #     print('Breaking at n = ', n)
#         #     break
#         alt = -alt
#         qpow = qpow * (q ** (2*n + 2))
#     return pref * s

# # θ1'''(0|τ) = -2 q^{1/4} Σ (-1)^n q^{n(n+1)} (2n+1)^3
# def _theta1_d3_0(tau, max_terms=10, rtol=1e-16):
#     q = _xcispi(tau)
#     pref = -2.0 * jnp.sqrt(jnp.sqrt(q))
#     s = 0.0 + 0.0j; c = 0.0 + 0.0j
#     alt = 1.0 + 0.0j; qpow = 1.0 + 0.0j
#     for n in range(max_terms):
#         k = 2*n + 1
#         term = alt * qpow * (k ** 3)
#         s, c = _kahan_add(s, c, term)
#         # if jnp.abs(term) <= rtol * (jnp.abs(s) + 1.0):
#         #     print('Breaking at n = ', n)
#         #     break
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
#     #if bool((jnp.real(z) == 0.0) & (jnp.imag(z) == 0.0)):
#        # return jnp.array(0.0 + 0.0j, dtype=jnp.complex128)

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
#     # if bool(jnp.abs(u_red) < small_u_thresh):
#     #     theta_ratio = u_red
#     # else:
#     #     theta_ratio = _theta1_series(u_red, tau) / t1p0
#     if jnp.abs(u_red) < small_u_thresh:
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
#     # if bool(jnp.abs(u_red) < small_u_thresh):
#     #     # log θ1(u) ≈ log θ1'(0) + log u  => ratio log = log u
#     #     log_theta_ratio = jnp.log(u_red)
#     # else:
#     #     log_theta_ratio = jnp.log(_theta1_series(u_red, tau)) - jnp.log(t1p0)
#     # if jnp.abs(u_red) < small_u_thresh:
#     #     # log θ1(u) ≈ log θ1'(0) + log u  => ratio log = log u
#     #     log_theta_ratio = jnp.log(u_red)
#     # else:
#     #     log_theta_ratio = jnp.log(_theta1_series(u_red, tau)) - jnp.log(t1p0)

#     log_theta_ratio = jax.lax.cond(
#     jnp.abs(u_red) < small_u_thresh,
#     lambda: jnp.log(u_red),  # True branch
#     lambda: jnp.log(_theta1_series(u_red, tau)) - jnp.log(t1p0)  # False branch
# )

#     # log σ at reduced point + exact correction back to z
#     log_base = jnp.log(2.0 * w1 / jnp.pi) + log_theta_ratio + c * (z_red * z_red)
#     log_corr = log_fac_theta + c * (z * z - z_red * z_red)

#     out = log_base + log_corr
#     return out


#########another implementation ########
# ================== Weierstrass sigma via ζ-integral (no theta/q-series) ==================
# ================== Weierstrass σ via ζ-integral @ reduced point + exact quasiperiods ==================
# ================== Weierstrass σ via ζ-integral @ reduced point + exact quasiperiods ==================

# ================== Weierstrass sigma σ(z|w1,w2): stable & direct ==================

# ================== Weierstrass sigma σ(z|w1,w2): stable & direct ==================
import jax
import jax.numpy as jnp

# def _c128(x):
#     x = jnp.asarray(x)
#     if x.dtype == jnp.complex64:   return x.astype(jnp.complex128)
#     if x.dtype == jnp.float32:     return x.astype(jnp.float64) + 0j
#     if jnp.issubdtype(x.dtype, jnp.floating): return x + 0j
#     return x

# def _xcispi(x):  # e^{i π x}
#     # x = _c128(x)
#     return jnp.exp(1j * jnp.pi * x)

# # θ1(u' + mπ + nπτ) = (-1)^{m+n} exp[-i (2n u' + n^2 π τ)] θ1(u')
# # def _reduce_u_and_z(u, tau, w1):
# #     pi = jnp.pi
# #     # choose integers n,m to center u' in the fundamental strip
# #     n = jnp.rint(jnp.imag(u) / (pi * jnp.imag(tau)))
# #     u1 = u - n * pi * tau
# #     m = jnp.rint(jnp.real(u1) / pi)
# #     u_red = u1 - m * pi
# #     z_red = (2.0 * w1 / pi) * u_red
# #     # exact θ1 multiplicative factor (MUST use u_red)
# #     log_fac_theta = 1j * pi * (m + n) - 1j * (2.0 * n * u_red + n * n * pi * tau)
# #     return u_red, z_red, log_fac_theta

# # ---- compensated (Kahan) summation helper ----
# def _kahan_add(acc, c, term):
#     y = term - c
#     t = acc + y
#     c_new = (t - acc) - y
#     return t, c_new

# # θ1(u|τ) = 2 q^{1/4} Σ_{n=0}∞ (-1)^n q^{n(n+1)} sin((2n+1)u)
# def _theta1_series(u, tau, max_terms=20, rtol=1e-16):
#     q = _xcispi(tau)
#     pref = 2.0 * jnp.sqrt(jnp.sqrt(q))
#     s = 0.0 + 0.0j; c = 0.0 + 0.0j
#     alt = 1.0 + 0.0j
#     qpow = 1.0 + 0.0j  # q^{n(n+1)} at n=0
#     for n in range(max_terms):
#         k = 2*n + 1
#         term = alt * qpow * jnp.sin(k * u)
#         s, c = _kahan_add(s, c, term)
#         # if jnp.abs(term) <= rtol * (jnp.abs(s) + 1.0):
#         #     print('Breaking at n = ', n)
#         #     break
#         alt = -alt
#         qpow = qpow * (q ** (2*n + 2))  # -> q^{(n+1)(n+2)}
#     return pref * s

# # θ1'(0|τ) = 2 q^{1/4} Σ (-1)^n q^{n(n+1)} (2n+1)
# def _theta1_d1_0(tau, max_terms=20, rtol=1e-16):
#     q = _xcispi(tau)
#     pref = 2.0 * jnp.sqrt(jnp.sqrt(q))
#     s = 0.0 + 0.0j; c = 0.0 + 0.0j
#     alt = 1.0 + 0.0j; qpow = 1.0 + 0.0j
#     for n in range(max_terms):
#         k = 2*n + 1
#         term = alt * qpow * k
#         s, c = _kahan_add(s, c, term)
#         # if jnp.abs(term) <= rtol * (jnp.abs(s) + 1.0):
#         #     print('Breaking at n = ', n)
#         #     break
#         alt = -alt
#         qpow = qpow * (q ** (2*n + 2))
#     return pref * s

# # θ1'''(0|τ) = -2 q^{1/4} Σ (-1)^n q^{n(n+1)} (2n+1)^3
# def _theta1_d3_0(tau, max_terms=20, rtol=1e-16):
#     q = _xcispi(tau)
#     pref = -2.0 * jnp.sqrt(jnp.sqrt(q))
#     s = 0.0 + 0.0j; c = 0.0 + 0.0j
#     alt = 1.0 + 0.0j; qpow = 1.0 + 0.0j
#     for n in range(max_terms):
#         k = 2*n + 1
#         term = alt * qpow * (k ** 3)
#         s, c = _kahan_add(s, c, term)
#         # if jnp.abs(term) <= rtol * (jnp.abs(s) + 1.0):
#         #     print('Breaking at n = ', n)
#         #     break
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

#     tau = w2 / w1  # ensure Im(tau) > 0 in your inputs
#     u   = jnp.pi * z / (2.0 * w1)

#     # reduce to central strip
#     u_red, z_red, log_fac_theta = _reduce_u_and_z(u, tau, w1)

#     # θ-derivatives at 0
#     t1p0   = _theta1_d1_0(tau)
#     t1ppp0 = _theta1_d3_0(tau)

#     # *** CORRECT gaussian constant (matches reference):  c = -(π^2/(24 w1^2)) * θ1'''(0)/θ1'(0)
#     c = - (jnp.pi * jnp.pi) / (24.0 * w1 * w1) * (t1ppp0 / t1p0)

#     # θ1(u_red)/θ1'(0) with small-u guard  (use lax.cond, no Python branch)
#     theta_ratio = jax.lax.cond(
#         jnp.abs(u_red) < small_u_thresh,
#         lambda a: a[0],  # just u_red
#         lambda a: _theta1_series(a[0], a[1]) / a[2],
#         operand=(u_red, tau, t1p0),
#     )

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

#     tau = w2 / w1  # ensure Im(tau) > 0 in your inputs
#     u   = jnp.pi * z / (2.0 * w1)

#     # reduce (u, z) into central strip & get exact θ1 multiplier (uses u_red!)
#     u_red, z_red, log_fac_theta = _reduce_u_and_z(u, tau, w1)

#     # θ1'(0), θ1'''(0) and Gaussian constant
#     t1p0   = _theta1_d1_0(tau)
#     t1ppp0 = _theta1_d3_0(tau)
#     c = - (jnp.pi * jnp.pi) / (24.0 * w1 * w1) * (t1ppp0 / t1p0)

#     # log[ θ1(u_red) / θ1'(0) ] with small-u guard (correct lax.cond signature)
#     log_theta_ratio = jax.lax.cond(
#         jnp.abs(u_red) < small_u_thresh,
#         lambda a: jnp.log(a[0]),  # log(u_red)
#         lambda a: jnp.log(_theta1_series(a[0], a[1])) - jnp.log(a[2]),
#         operand=(u_red, tau, t1p0),
#     )

#     # log σ at reduced point + exact correction back to z
#     log_base = jnp.log(2.0 * w1 / jnp.pi) + log_theta_ratio + c * (z_red * z_red)
#     log_corr = log_fac_theta + c * (z * z - z_red * z_red)

#     out = log_base + log_corr
#     return out

# def log_weierstrass_sigma(z, w1, w2, small_u_thresh=1e-12):
#     """
#     Principal-branch log σ(z | w1, w2), τ = w2/w1 with Im(τ) > 0.

#     Stable for large |z| via lattice reduction (evaluate at reduced z' and
#     restore with exact quasiperiod factor). Uses the Gaussian constant:
#         c = -(π^2 / (24 w1^2)) * θ1'''(0|τ) / θ1'(0|τ)

#     Returns:
#       complex: log σ(z). At z=0 returns ~ -inf + 0j (zero of σ).
#     """

#     tau = w2 / w1
#     u   = jnp.pi * z / (2.0 * w1)

#     # reduce (u, z) into central strip & get exact θ1 multiplier (uses u_red!)
#     u_red, z_red, log_fac_theta = _reduce_u_and_z(u, tau, w1)

#     # θ1'(0), θ1'''(0) and Gaussian constant
#     t1p0   = _theta1_d1_0(tau)
#     t1ppp0 = _theta1_d3_0(tau)
#     c = - (jnp.pi * jnp.pi) / (24.0 * w1 * w1) * (t1ppp0 / t1p0)

#     # ----- replace lax.cond with jnp.where -----

#     # full θ1 series
#     theta_series = _theta1_series(u_red, tau)          # θ1(u_red | τ)

#     # two branches
#     # small-u: θ1(u)/θ1'(0) ~ u  → log ~ log(u)
#     branch_small = jnp.log(u_red)

#     # generic: log[ θ1(u_red) / θ1'(0) ]
#     branch_generic = jnp.log(theta_series) - jnp.log(t1p0)

#     mask = (jnp.abs(u_red) < small_u_thresh)
#     log_theta_ratio = jnp.where(mask, branch_small, branch_generic)

#     # -------------------------------------------------

#     log_base = jnp.log(2.0 * w1 / jnp.pi) + log_theta_ratio + c * (z_red * z_red)
#     log_corr = log_fac_theta + c * (z * z - z_red * z_red)

#     out = log_base + log_corr
#     return out



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

# def LLL_log(z: jnp.ndarray, N_phi: jnp.ndarray, L1: jnp.ndarray, L2: jnp.ndarray,
#                    almost: jnp.ndarray, M: int = 11) -> jnp.ndarray:
#     """
#     Compute the log of σ̂(z; ω=L/2)*exp(-|z|^2/(2Nφ))*exp(-almost z^2/2)
#     in log space.
#     """
#     L1com = to_cplx_divsqrt2(L1); L2com = to_cplx_divsqrt2(L2)
#     w1, w2 = L1com / 2.0, L2com / 2.0

#     # Compute log(σ̂(z; ω))
#     log_sig = log_weierstrass_sigma(z, w1, w2)

#     # Compute log of the Gaussian term
#     log_gauss = -(z * jnp.conj(z)).real / (2.0 * N_phi)

#     # Compute log of the quadratic term
#     log_quad = -0.5 * almost * (z * z)

#     # Combine all terms in log space
#     log_result = log_sig + log_gauss + log_quad

#     return log_result

# def LLL_with_zeros_log(z, N_phi, L1, L2, almost, zeros, M: int = 100):
#     """
#     log(∏_a σ(z-a)) + [(Σ conj(a)) z - conj(z) (Σ a)]/(2 Nφ)
#       - n_z * |z|^2/(2 Nφ) - (n_z/2) * almost * z^2
#     where n_z = number of zeros
#     """
#     L1com = to_cplx_divsqrt2(L1); L2com = to_cplx_divsqrt2(L2)
#     w1, w2 = L1com / 2.0, L2com / 2.0

#     nz    = zeros.size

#     # sum over zeros of log σ(z-a)
#     log_sig = jnp.sum(jnp.array(
#         [log_weierstrass_sigma(z - a, w1, w2) for a in zeros]
#     ))

#     # gauge term (sums once over zeros)
#     A = jnp.sum(zeros)
#     log_gauge = (jnp.conj(A) * z - jnp.conj(z) * A) / (2.0 * N_phi)

#     # Gaussian and quadratic appear once per zero in your Julia loop
#     log_gauss = - nz * (z * jnp.conj(z)).real / (2.0 * N_phi)
#     log_quad  = - 0.5 * nz * almost * (z * z)
#     def safe_log(zlog, lo=-80.0, hi=80.0):
#       re = jnp.clip(jnp.real(zlog), a_min=lo, a_max=hi)
#       return re + 1j*jnp.imag(zlog)

#     log_result = log_sig + log_gauge + log_gauss + log_quad
#     log_result = safe_log(log_result)                 # keeps fp32 stable
#     return log_result.astype(jnp.complex64)


# def LLL_with_zeros_log(z, N_phi, L1, L2, almost, zeros, M: int = 100):
#     """
#     Map z back to the fundamental supercell, evaluate the LLL envelope there,
#     and add the magnetic phase for the corresponding lattice translation.

#     Returns:
#       complex64 log of the envelope at z with correct magnetic quasi-period.
#     """
#     # ---- real-space reduction r = t1 L1 + t2 L2, take n = floor(t), r' = r - n1 L1 - n2 L2
#     # Convert complex z -> real r using the (x+iy)/sqrt(2) convention.
#     sqrt2 = jnp.sqrt(jnp.array(2.0, dtype=jnp.float32))
#     r = jnp.array([jnp.real(z), jnp.imag(z)], dtype=jnp.float32) * sqrt2

#     lattice = jnp.stack([L1, L2], axis=1).astype(jnp.float32)   # shape (2,2) with columns L1,L2
#     Linv    = jnp.linalg.inv(lattice)
#     t       = Linv @ r                                           # fractional coords
#     n       = jnp.floor(t)                                       # put r' in [0,1)^2
#     n_int   = jax.lax.stop_gradient(n).astype(jnp.int32)

#     L_vec   = lattice @ n.astype(jnp.float32)                    # L = n1 L1 + n2 L2
#     L_vec   = jax.lax.stop_gradient(L_vec)
#     r_prime = r - L_vec                                          # reduced position in cell

#     # η = +1 if both n1,n2 are even, else −1
#     both_even = jnp.all((n_int & 1) == 0)
#     eta = jnp.where(both_even, jnp.array(1.0, dtype=jnp.float32),
#                                jnp.array(-1.0, dtype=jnp.float32))

#     # Magnetic phase: exp(i/2 * (r' × L)), with 2D cross as scalar
#     rprime_cross_L = r_prime[0] * L_vec[1] - r_prime[1] * L_vec[0]
#     theta = jnp.array(0.5, dtype=jnp.float32) * rprime_cross_L

#     # log(phase) = i * theta + log(eta);  log(±1) = 0 or iπ
#     log_eta = jnp.where(eta > 0.0, jnp.array(0.0, dtype=jnp.float32),
#                                    jnp.pi * jnp.array(1j, dtype=jnp.complex64))
#     log_phase = (1j * theta.astype(jnp.complex64)) + log_eta.astype(jnp.complex64)

#     # Reduced complex coordinate z' for evaluating the periodic part
#     z_prime = (r_prime[0] + 1j * r_prime[1]) / sqrt2

#     # ---- LLL envelope at reduced point (use z_prime everywhere below)
#     L1com = to_cplx_divsqrt2(L1).astype(jnp.complex64)
#     L2com = to_cplx_divsqrt2(L2).astype(jnp.complex64)
#     w1, w2 = L1com / 2.0, L2com / 2.0

#     nz = zeros.size
#     # sum_a log σ(z' - a)
#     log_sig = jnp.sum(jnp.array(
#         [log_weierstrass_sigma(z_prime - a, w1, w2) for a in zeros], dtype=jnp.complex64
#     ))

#     # gauge, Gaussian, quadratic terms (with z_prime)
#     A = jnp.sum(zeros).astype(jnp.complex64)
#     Nphi_f = jnp.asarray(N_phi, dtype=jnp.float32)

#     log_gauge = (jnp.conj(A) * z_prime - jnp.conj(z_prime) * A) / (2.0 * Nphi_f)
#     log_gauss = - nz * (z_prime * jnp.conj(z_prime)).real / (2.0 * Nphi_f)
#     log_quad  = - 0.5 * jnp.asarray(nz, jnp.float32) * jnp.asarray(almost, jnp.complex64) * (z_prime * z_prime)

#     def safe_log(zlog, lo=-80.0, hi=80.0):
#         re = jnp.clip(jnp.real(zlog), a_min=lo, a_max=hi)
#         return re + 1j * jnp.imag(zlog)

#     log_result = (log_sig
#                   + log_gauge.astype(jnp.complex64)
#                   + jnp.asarray(log_gauss, jnp.complex64)
#                   + jnp.asarray(log_quad,  jnp.complex64))

#     # add the exact magnetic phase for the (r' + L) decomposition
#     log_result = log_result + log_phase

#     # optional stability clamp on the real part
#     log_result = safe_log(log_result)
#     return log_result.astype(jnp.complex64)

###### implementation with shifting to supercell ########
# import jax
# import jax.numpy as jnp
# from jax import lax

# # # ------------------------- helpers (SG wrappers) -------------------------

# def floor_sg(x):
#   """Discrete floor with gradients stopped (both input and output)."""
#   x = lax.stop_gradient(x)
#   return lax.stop_gradient(jnp.floor(x))

# def round_sg(x):
#   """Discrete round with gradients stopped (both input and output)."""
#   x = lax.stop_gradient(x)
#   return lax.stop_gradient(jnp.rint(x))

# # # ------------------------- complex helpers (fp32/c64) --------------------

# def _xcispi(x):  # e^{i π x}
#     # force complex64 to stay GPU-friendly and consistent
#     x_c  = jnp.asarray(x, jnp.complex64)
#     pi_c = jnp.asarray(jnp.pi, jnp.complex64)
#     return jnp.exp(1j * pi_c * x_c)

# def to_cplx_divsqrt2(v2: jnp.ndarray) -> jnp.ndarray:
#   """(x,y) -> (x+iy)/sqrt(2)  (returns complex64)."""
#   return ((v2[0].astype(jnp.float32) + 1j * v2[1].astype(jnp.float32))
#           / jnp.sqrt(jnp.asarray(2.0, jnp.float32))).astype(jnp.complex64)

# # # ---- compensated (Kahan) summation helper ----
# # def _kahan_add(acc, c, term):
# #   acc = acc.astype(jnp.complex64); c = c.astype(jnp.complex64); term = term.astype(jnp.complex64)
# #   y = term - c
# #   t = acc + y
# #   c_new = (t - acc) - y
# #   return t, c_new

# # # ------------------------- theta reduction (uses round_sg) ----------------

# # # θ1(u' + mπ + nπτ) = (-1)^{m+n} exp[-i (2n u' + n^2 π τ)] θ1(u')
# def _reduce_u_and_z(u, tau, w1):
#   """Reduce u into central strip using SG-rounded m,n; return (u_red, z_red, log θ-shift)."""
#   pi = jnp.asarray(jnp.pi, jnp.float32)
#   u = u.astype(jnp.complex64); tau = tau.astype(jnp.complex64); w1 = w1.astype(jnp.complex64)

#   # choose integers n,m via STOP-GRADIENT round
#   n = round_sg(jnp.imag(u) / (pi * jnp.imag(tau)))          # real scalar
#   u1 = u - n.astype(jnp.complex64) * pi * tau
#   m = round_sg(jnp.real(u1) / pi)
#   u_red = u1 - m.astype(jnp.complex64) * pi

#   # map back to z via z = (2 w1 / π) u
#   z_red = (2.0 * w1 / pi.astype(jnp.complex64)) * u_red

#   # exact θ1 multiplicative factor (must use u_red)
#   log_fac_theta = (1j * pi.astype(jnp.complex64) * (m + n).astype(jnp.complex64)
#                    - 1j * (2.0 * n.astype(jnp.complex64) * u_red
#                            + (n.astype(jnp.complex64) ** 2) * pi.astype(jnp.complex64) * tau))
#   return u_red, z_red, log_fac_theta

# # ------------------------- theta series (c64) -----------------------------

# # θ1(u|τ) = 2 q^{1/4} Σ_{n=0}∞ (-1)^n q^{n(n+1)} sin((2n+1)u)
# def _theta1_series(u, tau, max_terms=12):
#   u = u.astype(jnp.complex64); tau = tau.astype(jnp.complex64)
#   q = _xcispi(tau)
#   pref = (2.0 * jnp.sqrt(jnp.sqrt(q))).astype(jnp.complex64)
#   s = jnp.array(0.0 + 0.0j, jnp.complex64)
#   c = jnp.array(0.0 + 0.0j, jnp.complex64)
#   alt  = jnp.array(1.0 + 0.0j, jnp.complex64)
#   qpow = jnp.array(1.0 + 0.0j, jnp.complex64)  # q^{n(n+1)} at n=0
#   for n in range(max_terms):
#     k = jnp.asarray(2*n + 1, jnp.float32)
#     term = alt * qpow * jnp.sin(k.astype(jnp.complex64) * u)
#     s, c = _kahan_add(s, c, term)
#     alt  = -alt
#     qpow = qpow * (q ** (2*n + 2))
#   return pref * s

# # θ1'(0|τ) = 2 q^{1/4} Σ (-1)^n q^{n(n+1)} (2n+1)
# def _theta1_d1_0(tau, max_terms=12):
#   tau = tau.astype(jnp.complex64)
#   q = _xcispi(tau)
#   pref = (2.0 * jnp.sqrt(jnp.sqrt(q))).astype(jnp.complex64)
#   s = jnp.array(0.0 + 0.0j, jnp.complex64)
#   c = jnp.array(0.0 + 0.0j, jnp.complex64)
#   alt  = jnp.array(1.0 + 0.0j, jnp.complex64)
#   qpow = jnp.array(1.0 + 0.0j, jnp.complex64)
#   for n in range(max_terms):
#     k = jnp.asarray(2*n + 1, jnp.float32)
#     term = alt * qpow * k.astype(jnp.complex64)
#     s, c = _kahan_add(s, c, term)
#     alt  = -alt
#     qpow = qpow * (q ** (2*n + 2))
#   return pref * s

# # θ1'''(0|τ) = -2 q^{1/4} Σ (-1)^n q^{n(n+1)} (2n+1)^3
# def _theta1_d3_0(tau, max_terms=12):
#   tau = tau.astype(jnp.complex64)
#   q = _xcispi(tau)
#   pref = (-2.0 * jnp.sqrt(jnp.sqrt(q))).astype(jnp.complex64)
#   s = jnp.array(0.0 + 0.0j, jnp.complex64)
#   c = jnp.array(0.0 + 0.0j, jnp.complex64)
#   alt  = jnp.array(1.0 + 0.0j, jnp.complex64)
#   qpow = jnp.array(1.0 + 0.0j, jnp.complex64)
#   for n in range(max_terms):
#     k = jnp.asarray(2*n + 1, jnp.float32)
#     term = alt * qpow * (k.astype(jnp.complex64) ** 3)
#     s, c = _kahan_add(s, c, term)
#     alt  = -alt
#     qpow = qpow * (q ** (2*n + 2))
#   return pref * s

# # ------------------------- σ and log σ (c64/FP32) ------------------------

# def weierstrass_sigma(z, w1, w2, small_u_thresh=1e-6):
#   """Direct σ(z | w1,w2) with central reduction and exact θ-shift (complex64)."""
#   z = z.astype(jnp.complex64); w1 = w1.astype(jnp.complex64); w2 = w2.astype(jnp.complex64)
#   pi = jnp.asarray(jnp.pi, jnp.float32).astype(jnp.complex64)
#   tau = w2 / w1
#   u   = pi * z / (2.0 * w1)
#   u_red, z_red, log_fac_theta = _reduce_u_and_z(u, tau, w1)

#   t1p0   = _theta1_d1_0(tau)
#   t1ppp0 = _theta1_d3_0(tau)
#   c = - (pi * pi) / (24.0 * w1 * w1) * (t1ppp0 / t1p0)

#   theta_ratio = jax.lax.cond(
#       (jnp.abs(u_red) < small_u_thresh),
#       lambda a: a[0],                                 # ~ u_red
#       lambda a: _theta1_series(a[0], a[1]) / a[2],
#       operand=(u_red, tau, t1p0),
#   )

#   sigma_base = (2.0 * w1 / pi) * theta_ratio * jnp.exp(c * (z_red * z_red))
#   corr = jnp.exp(log_fac_theta) * jnp.exp(c * (z * z - z_red * z_red))
#   return (sigma_base * corr).astype(jnp.complex64)

# def log_weierstrass_sigma(z, w1, w2, small_u_thresh=1e-6):
#   """Principal-branch log σ(z | w1,w2) with reduction and exact θ-shift (complex64)."""
#   z = z.astype(jnp.complex64); w1 = w1.astype(jnp.complex64); w2 = w2.astype(jnp.complex64)
#   pi = jnp.asarray(jnp.pi, jnp.float32).astype(jnp.complex64)
#   tau = w2 / w1
#   u   = pi * z / (2.0 * w1)

#   u_red, z_red, log_fac_theta = _reduce_u_and_z(u, tau, w1)
#   t1p0   = _theta1_d1_0(tau)
#   t1ppp0 = _theta1_d3_0(tau)
#   c = - (pi * pi) / (24.0 * w1 * w1) * (t1ppp0 / t1p0)

#   log_theta_ratio = jax.lax.cond(
#       (jnp.abs(u_red) < small_u_thresh),
#       lambda a: jnp.log(a[0]),
#       lambda a: jnp.log(_theta1_series(a[0], a[1])) - jnp.log(a[2]),
#       operand=(u_red, tau, t1p0),
#   )

#   log_base = jnp.log(2.0 * w1 / pi) + log_theta_ratio + c * (z_red * z_red)
#   log_corr = log_fac_theta + c * (z * z - z_red * z_red)
#   return (log_base + log_corr).astype(jnp.complex64)

# # ------------------------- lattice mapping + magnetic phase ---------------

# def _map_to_cell_and_phase(z, L1, L2):
#   """
#   Map complex z back to the fundamental cell spanned by L1, L2 (using floor_sg),
#   and return (z0, log_phase) where
#     z0 is the reduced complex coordinate, and
#     log_phase = log(η) + i * (r0 × L)/2 with η = +1 if (n1,n2) both even else -1.
#   All math in float32/complex64; gradients are stopped through (n1,n2).
#   """
#   # real-space r from z: z = (x + i y)/sqrt(2)
#   sqrt2 = jnp.sqrt(jnp.asarray(2.0, jnp.float32))
#   x = jnp.real(z).astype(jnp.float32) * sqrt2
#   y = jnp.imag(z).astype(jnp.float32) * sqrt2
#   r = jnp.stack([x, y], axis=0)  # (2,)

#   L1 = L1.astype(jnp.float32); L2 = L2.astype(jnp.float32)
#   lattice = jnp.stack([L1, L2], axis=1)               # (2,2)
#   Linv = jnp.linalg.inv(lattice.astype(jnp.float32))  # (2,2)

#   # fractional coords and STOP-GRADIENT integer windings
#   t = Linv @ r                                  # (2,)
#   n = jnp.stack([floor_sg(t[0]), floor_sg(t[1])], axis=0)  # int-ish, SG
#   n_int = n.astype(jnp.int32)

#   # L = n1 L1 + n2 L2, r0 = r - L
#   L_vec = lattice @ n.astype(jnp.float32)
#   r0 = r - L_vec

#   # Magnetic phase and η
#   r0_cross_L = r0[0]*L_vec[1] - r0[1]*L_vec[0]                 # scalar
#   theta = -1.0 * jnp.asarray(0.5, jnp.float32) * r0_cross_L           # observe the minus sign
#   both_even = jnp.all((n_int & 1) == 0)
#   eta = jnp.where(both_even, jnp.asarray(1.0, jnp.float32), jnp.asarray(-1.0, jnp.float32))
#   log_phase = jnp.log(eta.astype(jnp.complex64)) + 1j * theta.astype(jnp.complex64)

#   # back to complex z0
#   z0 = ((r0[0] + 1j * r0[1]) / sqrt2).astype(jnp.complex64)
#   return z0, log_phase

# # ------------------------- LLL with zeros (mapped + phase) ----------------

# def LLL_with_zeros_log(z, N_phi, L1, L2, almost, zeros, M: int = 100):
#   """
#   Returns complex64:
#     log ψ(z) = [ Σ_a log σ(z0 - a) ] + [(Σ conj(a)) z0 - conj(z0) (Σ a)]/(2 Nφ)
#                - n_z * |z0|^2/(2 Nφ) - (n_z/2) * almost * z0^2  +  log_phase
#   where z0 is z mapped back to the supercell and log_phase is the magnetic
#   quasi-periodicity factor from the winding n.
#   Discrete lattice reductions use stop_gradient to avoid Hessian/OOM.
#   """
#   # Map to cell and add magnetic phase
#   z0, log_phase = _map_to_cell_and_phase(z, L1, L2)

#   # lattice half-periods in complex convention
#   L1com = to_cplx_divsqrt2(L1); L2com = to_cplx_divsqrt2(L2)
#   w1, w2 = (L1com / 2.0).astype(jnp.complex64), (L2com / 2.0).astype(jnp.complex64)

#   zeros = jnp.asarray(zeros, jnp.complex64)
#   nz = zeros.size

#   # sum_a log σ(z0 - a)
#   log_sig = jnp.sum(jnp.array([log_weierstrass_sigma(z0 - a, w1, w2) for a in zeros], dtype=jnp.complex64))

#   # gauge / gaussian / quadratic terms (evaluated at z0)
#   Nphi_f32 = jnp.asarray(N_phi, jnp.float32)
#   Nphi_c64 = Nphi_f32.astype(jnp.complex64)
#   almost_c64 = jnp.asarray(almost, jnp.float32).astype(jnp.complex64)

#   A = jnp.sum(zeros).astype(jnp.complex64)
#   log_gauge = (jnp.conj(A) * z0 - jnp.conj(z0) * A) / (2.0 * Nphi_c64)

#   abs2_z0 = (z0 * jnp.conj(z0)).real.astype(jnp.float32)
#   log_gauss = - nz * abs2_z0.astype(jnp.float32) / (2.0 * Nphi_f32)
#   log_gauss = log_gauss.astype(jnp.complex64)

#   log_quad = - 0.5 * jnp.asarray(nz, jnp.float32) * almost_c64 * (z0 * z0)

#   # combine
#   log_result = (log_sig + log_gauge + log_gauss + log_quad + log_phase).astype(jnp.complex64)

#   # gentle clamp on real part (optional stability)
#   def _safe_log(zlog, lo=-80.0, hi=80.0):
#     re = jnp.clip(jnp.real(zlog), a_min=jnp.asarray(lo, jnp.float32), a_max=jnp.asarray(hi, jnp.float32))
#     return (re.astype(jnp.float32) + 1j * jnp.imag(zlog).astype(jnp.float32)).astype(jnp.complex64)

#   return _safe_log(log_result)


import jax
from jax import lax
import jax.numpy as jnp
from typing import Sequence, Mapping

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

def to_cplx_divsqrt2(v2: jnp.ndarray) -> jnp.ndarray:
  """(x,y) -> (x+iy)/sqrt(2) as complex64."""
  return ((v2[0].astype(jnp.float32) + 1j * v2[1].astype(jnp.float32))
          / jnp.sqrt(jnp.asarray(2.0, jnp.float32))).astype(jnp.complex64)

# ---------- reduce u,z with exact θ₁ shift (your old implementation) ----------

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


# def make_laughlin_log_psi_fn(
#     *,
#     L1: jnp.ndarray,
#     L2: jnp.ndarray,
#     q: int,
#     extra_kwargs: Dict[str, Any] = None,
# ):
#   """
#   Returns a function log_psi_L(pos) that computes the complex log of the
#   Laughlin wavefunction on a torus for flattened positions `pos`.

#   Args:
#     L1, L2: Real-space primitive vectors (shape (2,)).
#     q: Laughlin denominator (e.g. q=3 for ν=1/3).
#     extra_kwargs: Any extra parameters your ellipticfunctions-based
#       implementation needs (N_phi, tau, COM state index, etc.).

#   Returns:
#     log_psi_L(pos): (nelec*2,) -> complex scalar.
#   """
#   if extra_kwargs is None:
#     extra_kwargs = {}

#   def log_psi_L(pos_flat: jnp.ndarray) -> jnp.complexfloating:
#     nelec = pos_flat.shape[0] // 2
#     coords = pos_flat.reshape(nelec, 2)  # (N, 2)
#     # Example: map to complex coordinate z = (x + i y)/sqrt(2)
#     z = (coords[:, 0] + 1j * coords[:, 1]) / jnp.sqrt(2.0)

#     # --- THIS PART YOU REPLACE WITH YOUR ACTUAL IMPLEMENTATION ---
#     # Something like:
#     #   return ellipticfunctions.log_laughlin_torus(
#     #       z, L1=L1, L2=L2, q=q, **extra_kwargs
#     #   )
#     #
#     # For now, just a placeholder to show the interface:
#     return ellipticfunctions.log_laughlin_torus(z, L1, L2, q, **extra_kwargs)

#   return log_psi_L

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