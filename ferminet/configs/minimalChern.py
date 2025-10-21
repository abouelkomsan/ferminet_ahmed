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
import logging

matmul_precision = 'float32' # 'F64_F64_F64', 'float32'
logging.info(f"Setting jax_default_matmul_precision to {matmul_precision}")
jax.config.update("jax_default_matmul_precision", matmul_precision)
logging.info(f"Matmul precision set to: {jax.default_matmul_precision}")

def lattice_vecs(a1:np.ndarray, a2:np.ndarray,Tmatrix:np.ndarray) -> np.ndarray:
  "Return the basis T1,T2 of the super-cell built from the unit cell lattice vectors a1 and a2"
  T1 = Tmatrix[0,0]*a1 + Tmatrix[0,1]*a2
  T2 = Tmatrix[1,0]*a1 + Tmatrix[1,1]*a2
  return np.column_stack([T1, T2])

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
  pp_coffs = np .array([0.0, 0.0, 0.0])  
  pp_phases = np.array([0.0, 0.0, 0.0])
  epsilon = 5.0
  intcoff = (coulomb_prefactor(epsilon))/KE_prefactor
  print(epsilon)
  cfg.system.make_local_energy_fn = "ferminet.pbc.Hamiltonian_minimalChern.local_energy"
  cfg.system.make_local_energy_kwargs = {"lattice": lattice, "heg": True,"potential_kwargs": {"laplacian_method": "folx","interaction_energy_scale": intcoff},"kinetic_energy_kwargs": {"prefactor": KE_prefactor}, "periodic_lattice": potential_lattice,"periodic_potential_kwargs": {"coefficients": pp_coffs, "phases": pp_phases},"Bfield_lattice": potential_lattice,"Bfield_kwargs" : {"flux": -0.23,"threadedflux": np.array([0,0])}}
  cfg.network.network_type = "psiformer"
  cfg.network.complex = True
  cfg.network.psiformer.num_layers = 4
  cfg.network.psiformer.num_heads = 6
  cfg.network.psiformer.heads_dim = 64
  cfg.network.psiformer.mlp_hidden_dims  = (256,)
  cfg.network.determinants = 4
  cfg.batch_size = 1024
  cfg.optim.iterations = 1000000
  #cfg.optim.lr.onecycle = True
  #cfg.optim.lr.onecycle_steps = 300000
  #cfg.optim.lr.rate_max = 30.0
  #cfg.optim.lr.onecycle_start = 1.0
  #cfg.optim.lr.onecycle_end = 0.0001 
  cfg.optim.lr.rate = 0.0001
  #cfg.optim.lr.decay = 0.0
  #cfg.optim.kfac.momentum = 0.2
  cfg.mcmc.enforce_symmetry_by_shift = "none"
  #cfg.mcmc.symmetry_shift_kwargs = {"lattice": lattice,'move_width': 1}
  #cfg.optim.lr.delay = 50000
  #cfg.mcmc.move_width = 2.0
  #cfg.mcmc.init_width = 3.0
  cfg.mcmc.steps = 50
  cfg.network.jastrow = 'none'
  cfg.initialization.donor_filename = "none"
  cfg.initialization.flatten_num_devices = False
  cfg.initialization.ignore_batch = False
  cfg.targetmom.mom = None
  #cfg.targetmom.kwargs = {"abs_lattice": Tmatrix, "unit_cell_vectors": np.array([a1,a2]), "logsumtrick": True}
  #cfg.initialization.modifications = ['orbital-rnd']
  #cfg.log.save_path = 'ferminet_2025_09_08_15:31:46'
  cfg.log.save_frequency = 40
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
    # New functionality: Create a timestamped folder and save the function body
  timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H:%M:%S')
  cfg.log.save_path = f'/work/submit/ahmed95/minimalChern_NN/ferminet_{timestamp}'
  os.makedirs(cfg.log.save_path, exist_ok=True)

  # Get the body of the current function
  function_body = inspect.getsource(get_config)

  # Write the function body to a text file inside the folder
  file_path = os.path.join(cfg.log.save_path, 'get_config_body.txt')
  with open(file_path, 'w') as file:
    file.write(function_body)
  print(f"Folder '{cfg.log.save_path}' created and function body saved to '{file_path}'.")

  return cfg


