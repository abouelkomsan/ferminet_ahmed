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


from ferminet import base_config
from ferminet.pbc import envelopes
from ferminet.utils import system

import numpy as np
import os
import datetime
import inspect
import jax
import jax.numpy as jnp
import logging
from jax.extend import backend as jbackend
import sys
from ferminet import train
from ferminet import ellipticfunctions



logging.basicConfig(
    level=logging.INFO,  # Adjust the logging level as needed
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stderr  # Send logs to stderr
)


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


matmul_precision = 'float32' # 'F64_F64_F64', 'float32'
logging.info(f"Setting jax_default_matmul_precision to {matmul_precision}")
jax.config.update("jax_default_matmul_precision", matmul_precision)
logging.info(f"Matmul precision set to: {jax.default_matmul_precision}")

def lattice_vecs(a1:np.ndarray, a2:np.ndarray,Tmatrix:np.ndarray) -> np.ndarray:
  "Return the basis T1,T2 of the super-cell built from the unit cell lattice vectors a1 and a2"
  T1 = Tmatrix[0,0]*a1 + Tmatrix[0,1]*a2
  T2 = Tmatrix[1,0]*a1 + Tmatrix[1,1]*a2
  return np.column_stack([T1, T2])

def lattice_vecs_centered(a1: np.ndarray, a2: np.ndarray, Tmatrix: np.ndarray) -> np.ndarray:
    """
    Return the basis T1,T2 of the supercell (columns), and shift them
    so the supercell spans [-L1/2, L1/2] × [-L2/2, L2/2].
    """
    # Build supercell lattice
    T1 = Tmatrix[0, 0] * a1 + Tmatrix[0, 1] * a2
    T2 = Tmatrix[1, 0] * a1 + Tmatrix[1, 1] * a2
    lattice = np.column_stack([T1, T2])

    # Shift origin to center: redefine the basis as centered
    # (effectively: origin at -L1/2 - L2/2, so cell spans symmetric range)
    L1c = lattice[:, 0] - 0.5 * (lattice[:, 0] + lattice[:, 1])
    L2c = lattice[:, 1] - 0.5 * (lattice[:, 0] + lattice[:, 1])

    return np.column_stack([L1c, L2c])

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

def e_over_hbar() -> float:
    """
    Returns e / hbar in units of 1/(meV·ns).
    """
    # Physical constants
    hbar = 1.054571817e-34  # J·s
    e = 1.602176634e-19     # C
    meV = 1e-3 * 1.602176634e-19  # J
    ns = 1e-9               # s

    # Compute e / hbar and convert to 1/(meV·ns)
    value = e / hbar  # in C·s⁻¹
    value_per_meV_ns = value * meV * ns  # convert to 1/(meV·ns)

    return value_per_meV_ns

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

def init_random_zeros(N_phi, L1, L2, key, scale=1.0):
    """
    Initialize N_phi random zeros (complex) within the unit cell spanned by L1,L2,
    and shift them so their mean is zero.

    Args:
        N_phi: int, number of zeros.
        L1, L2: jnp.ndarray, shape (2,), primitive lattice vectors (real space).
        key: jax.random.PRNGKey.
        scale: optional float, fraction of the cell size for random spread.

    Returns:
        zeros: jnp.ndarray, shape (N_phi,), dtype=complex64, Σzeros = 0.
    """
    # Random lattice coefficients in [0,1)
    key_u, key_v = jax.random.split(key)
    u = jax.random.uniform(key_u, (N_phi,), dtype=jnp.float32)
    v = jax.random.uniform(key_v, (N_phi,), dtype=jnp.float32)

    # Map to real-space vectors: a = u L1 + v L2
    a_vec = u[:, None] * L1[None, :] + v[:, None] * L2[None, :]
    # Convert to complex LLL convention (x+iy)/√2
    a_complex = (a_vec[:, 0] + 1j * a_vec[:, 1]) / jnp.sqrt(2.0)
    # Center to have zero mean
    a_complex -= jnp.mean(a_complex)
    # Optional scaling (e.g., 0.8 for slightly compressed spread)
    a_complex *= jnp.float32(scale)
    return a_complex.astype(jnp.complex64)


def make_zero_lattice(potential_lattice: np.ndarray,
                      Tmatrix: np.ndarray,
                      z_scale: float = 1.0):
    """
    Construct zeros on a lattice: one zero in each primitive cell inside the supercell.

    Args
    ----
    potential_lattice : (2, 2)
        Columns are primitive vectors a1, a2 (1-flux unit cell).
    Tmatrix : (2, 2)
        Integer tiling matrix, e.g. [[3,0],[0,3]] for a 3x3 supercell.
        We assume here it's diagonal (or at least that T[0,0], T[1,1] are the
        tile counts along a1, a2).
    z_scale : float
        Same z_scale you use in make_magnetic_envelope_2d. Used to convert to
        LLL complex coordinates z = z_scale * (x + i y)/sqrt(2).

    Returns
    -------
    zeros_cart : (N_phi, 2)
        Cartesian (x,y) positions of zeros in the supercell.
    zeros_complex : (N_phi,)
        Complex LLL coordinates of zeros: z = z_scale * (x + i y) / sqrt(2).
    """
    potential_lattice = np.asarray(potential_lattice, float)  # (2,2)
    Tmatrix = np.asarray(Tmatrix, int)

    a1 = potential_lattice[:, 0]   # (2,)
    a2 = potential_lattice[:, 1]   # (2,)

    # number of tiles along each primitive direction
    nx = int(Tmatrix[0, 0])
    ny = int(Tmatrix[1, 1])

    zeros_cart = []
    for ix in range(nx):
        for iy in range(ny):
            # origin of this primitive cell
            R = ix * a1 + iy * a2
            # put the zero at the cell center
            center = R + 0.5 * (a1 + a2)
            zeros_cart.append(center)

    zeros_cart = np.stack(zeros_cart, axis=0)  # (N_phi, 2)
    # N_phi = nx * ny = det(Tmatrix) for diagonal T

    # convert to LLL complex coordinates (same convention as in envelope)
    zeros_complex = z_scale * (zeros_cart[:, 0] + 1j * zeros_cart[:, 1]) / np.sqrt(2.0)

    return zeros_cart, zeros_complex

def get_config(momind):
  # Get default options.
  cfg = base_config.default()
  cfg.system.electrons = (9, 0)
  cfg.system.ndim = 2
  # A ghost atom at the origin defines one-electron coordinate system.
  # Element 'X' is a dummy nucleus with zero charge
  cfg.system.molecule = [system.Atom("X", (0., 0.,))]
  # Pretraining is not currently implemented for systems in PBC
  cfg.pretrain.method = None

  """ Defining the potential unit cell and the supercell """
  a = np.sqrt((4 * np.pi * 1**2) / np.sqrt(3))  # primitive 1-flux length
  a1 = a * np.array([np.sqrt(3) / 2, -0.5])
  a2 = a * np.array([0, 1.0])
  Tmatrix = np.array([[3,-3], [3, 6]])  
  lattice = lattice_vecs(a1, a2, Tmatrix)
  potential_lattice = lattice_vecs(a1, a2, np.array([[1,0], [0, 1]]))
  #kpoints = envelopes.make_kpoints(lattice, cfg.system.electrons)
  cfg.system.pbc_lattice = lattice
  """Defining KE, potential and interaction parameters"""
  meff = 1.0
  KE_prefactor = hbar2_over_m_eff(meff)
  print(KE_prefactor)
  pp_coffs = np .array([0.0, 0.0, 0.0])  
  pp_phases = np.array([0.0, 0.0, 0.0])
  #epsilon = 5.0
  intcoff = 30.0
  #print(epsilon)
  cfg.system.make_local_energy_fn = "ferminet.pbc.Hamiltonian_quantumHall.local_energy"
  cfg.system.make_local_energy_kwargs = {"lattice": lattice, "heg": True,"potential_type": "Coulomb","potential_kwargs": {"laplacian_method": "folx","interaction_energy_scale": intcoff},"kinetic_energy_kwargs": {"prefactor": KE_prefactor}, "periodic_lattice": potential_lattice,"periodic_potential_kwargs": {"coefficients": pp_coffs, "phases": pp_phases},"Bfield_lattice": potential_lattice,"Bfield_kwargs" : {"flux": -0.23,"threadedflux": np.array([0,0])}}
  cfg.network.network_type = "psiformer_magfield"
  cfg.network.complex = True
  cfg.network.psiformer.num_layers = 4
  cfg.network.psiformer.num_heads = 4
  cfg.network.psiformer.heads_dim = 64
  cfg.network.psiformer.mlp_hidden_dims  = (256,)
  cfg.network.determinants = 2
  cfg.batch_size = 1024
  cfg.optim.optimizer = "none"
  cfg.optim.iterations = 1000
  cfg.optim.lr.rate = 0.05
  #cfg.optim.lr.decay = 0.0
  #cfg.optim.lr.delay = 5000
  cfg.optim.kfac.norm_constraint = 1e-4
  #cfg.optim.lr.onecycle = True
  #cfg.optim.lr.onecycle_steps = 300000
  #cfg.optim.lr.rate_max = 30.0
  #cfg.optim.lr.onecycle_start = 1.0
  #cfg.optim.lr.onecycle_end = 0.0001 
  cfg.mcmc.enforce_symmetry_by_shift = "none"
  cfg.mcmc.symmetry_shift_kwargs = {"lattice": lattice,'move_width': 1}
  cfg.mcmc.project_to_supercell = True
  #cfg.optim.lr.delay = 50000
  #cfg.mcmc.move_width = 2.0
  #cfg.mcmc.init_width = 3.0
  cfg.mcmc.steps = 100
  cfg.mcmc.burn_in = 300
  cfg.initialization.donor_filename = "/data/ahmed95/torusquantumHall/ferminet_2025_12_26_11:17:45"
  #cfg.initialization.modifications = ['orbital-rnd']
  cfg.initialization.flatten_num_devices = False
  cfg.initialization.ignore_batch = False
  cfg.initialization.randomize = False
  cfg.initialization.reset_t = True
  cfg.targetmom.mom = None
  #key = jax.random.PRNGKey(64)
  #complex_zeros = init_random_zeros(9, lattice[:,0], lattice[:,1], key)
  zeros_cart, zeros_complex = make_zero_lattice(potential_lattice, Tmatrix, z_scale=1.0)
  zeros_complex = zeros_complex - zeros_complex.mean()
  cfg.network.psiformer_magfield.kwargs = {"lattice": lattice, "N_phi": np.array(27.0),"zeros": jnp.asarray(zeros_complex, jnp.complex64),"rescale": np.array(1.0),"N_holo": 27, "N_anti": 0,"bf_strength_init" : 0.01}
  cfg.targetmom.kwargs = {"abs_lattice": Tmatrix, "unit_cell_vectors": np.array([a1,a2]), "logsumtrick": True}
  #cfg.initialization.modifications = ['orbital-rnd']
  #cfg.log.save_path = 'ferminet_2025_09_08_15:31:46'
  cfg.log.save_frequency = 1
  cfg.network.make_feature_layer_fn = (
      "ferminet.pbc.feature_layer.make_pbc_feature_layer")
  cfg.network.make_feature_layer_kwargs = {
      "lattice": lattice,
      "include_r_ae": False,
  }
  cfg.network.jastrow = 'simple_ee'
  cfg.network.jastrow_kwargs = {"ndim": 2,"interaction_strength": intcoff}
  #kpoints = envelopes.make_kpoints_2d(lattice, cfg.system.electrons,9)
  cfg.network.make_envelope_fn = ( "ferminet.pbc.envelopes.make_LLL_envelope_2d_trainable_zeros_mixed5" )
  cfg.network.make_envelope_kwargs = {"lattice":lattice,"elliptic_log_sigma":ellipticfunctions._LLL_with_zeros_log_cached,"magfield_kwargs" :cfg.network.psiformer_magfield.kwargs}
  #cfg.network.make_envelope_fn = ( "ferminet.pbc.envelopes.make_magnetic_laughlin_envelope_2d" )
  #cfg.network.make_envelope_kwargs = {"lattice":lattice}
  cfg.targetmom.mom = momind
  cfg.targetmom.kwargs = {"abs_lattice": Tmatrix, "unit_cell_vectors": jnp.array([a1,a2]), "logsumtrick": False,"magnetic_length": 1.0}
  cfg.network.full_det = True
  # New functionality: Create a timestamped folder and save the function body
  timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H:%M:%S')
  cfg.log.save_path = f'/data/ahmed95/torusquantumHall/ferminet_18particles_int_1.0_batch_50_threehalfs_momind_{momind}'
  os.makedirs(cfg.log.save_path, exist_ok=True)

  # Get the body of the current function
  function_body = inspect.getsource(get_config)

  # Write the function body to a text file inside the folder
  file_path = os.path.join(cfg.log.save_path, 'get_config_body.txt')
  with open(file_path, 'w') as file:
    file.write(function_body)
  print(f"Folder '{cfg.log.save_path}' created and function body saved to '{file_path}'.")

  return cfg


# for momind in range(6):  # 0 to 5 inclusive
#     cfg = get_config(momind)
#     train.train(cfg)

train.train(get_config(None))