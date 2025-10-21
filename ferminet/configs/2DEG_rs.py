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


def _sc_lattice_vecs(rs: float, nelec: int) -> np.ndarray:
    """Returns simple square lattice vectors with Wigner-Seitz radius rs in 2D."""
    area = np.pi * (rs**2) * nelec  # Area of the system in 2D
    length = area**(1 / 2)  # Length of the square's side
    return length * np.eye(2)  # 2D identity matrix scaled by length

def _sc_lattice_vecs_triang(rs: float, nelec: int) -> np.ndarray:
    """Returns triangular lattice vectors with Wigner-Seitz radius rs in 2D."""
    # Calculate the area of the system
    area = np.pi * (rs**2) * nelec  # Total area of the system in 2D
    
    # Calculate the lattice constant 'a' for a triangular lattice
    a = (2 * area / np.sqrt(3))**(1 / 2)
    
    # Define the lattice vectors for a triangular lattice
    lattice_vecs = np.array([
        [a, 0],                # First lattice vector
        [a / 2, a * np.sqrt(3) / 2]  # Second lattice vector
    ])
    
    return lattice_vecs

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
  cfg.system.electrons = (9, 0)
  cfg.system.ndim = 2
  # A ghost atom at the origin defines one-electron coordinate system.
  # Element 'X' is a dummy nucleus with zero charge
  cfg.system.molecule = [system.Atom("X", (0., 0.,))]
  # Pretraining is not currently implemented for systems in PBC
  cfg.pretrain.method = None

  """ Defining the potential unit cell and the supercell """

  lattice = _sc_lattice_vecs_triang(50,9)
    #kpoints = envelopes.make_kpoints(lattice, cfg.system.electrons)
  intcoff = 1.0
  cfg.system.make_local_energy_fn = "ferminet.pbc.Hamiltonian_2DEG.local_energy"
  cfg.system.make_local_energy_kwargs = {"lattice": lattice, "heg": True,"potential_kwargs": {"laplacian_method": "folx","interaction_energy_scale": intcoff}}
  cfg.network.network_type = "psiformer"
  cfg.network.complex = True
  cfg.network.psiformer.num_layers = 4
  cfg.network.psiformer.num_heads = 6
  cfg.network.psiformer.heads_dim = 64
  cfg.network.psiformer.mlp_hidden_dims  = (256,)
  cfg.network.determinants = 4
  cfg.batch_size = 1024
  cfg.optim.iterations = 100000
  cfg.optim.lr.rate = 0.05
  #cfg.optim.lr.decay = 0.1
  #cfg.optim.lr.delay = 1.0
  #cfg.mcmc.move_width = 2.0
  #cfg.mcmc.init_width = 3.0
  #cfg.mcmc.steps = 20
  cfg.initialization.donor_filename = "none"
  cfg.targetmom.mom = None

  #cfg.targetmom.kwargs = {"abs_lattice": Tmatrix, "unit_cell_vectors": np.array([a1,a2]), "logsumtrick": True}
  #cfg.initialization.modifications = ['orbital-rnd']
  #cfg.log.save_path = ''
  cfg.log.save_frequency = 10
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
  return cfg

