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
from ferminet import train

import numpy as np
import argparse
#set here number of devices
import os
import logging
import jax
from jax.extend import backend as jbackend
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

logging.basicConfig(
    level=logging.INFO,  # Adjust the logging level as needed
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stderr  # Send logs to stderr
)

_backend = jbackend.get_backend()
logging.info(f"Backend platform: {_backend.platform}")
logging.info(f"Platform version: {getattr(_backend, 'platform_version', 'n/a')}")
logging.info(f"Devices: {jax.devices()}")

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


def get_config(flux1,flux2,filename):
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
  rec = 2*np.pi*np.linalg.inv(lattice)
  threadedflux = flux1*rec[0,:] + flux2*rec[1,:] 
  """Defining KE, potential and interaction parameters"""
  meff = 1.0
  KE_prefactor = hbar2_over_m_eff(meff)
  print(KE_prefactor)
  pp_coffs = np.array([0.0, 0.0, 0.0])  
  pp_phases = np.array([0.0, 0.0, 0.0])
  intcoff = (coulomb_prefactor(5.0))/KE_prefactor
  print(intcoff)
  cfg.system.make_local_energy_fn = "ferminet.pbc.Hamiltonian_minimalChern.local_energy"
  cfg.system.make_local_energy_kwargs = {"lattice": lattice, "heg": True,"potential_kwargs": {"laplacian_method": "folx","interaction_energy_scale": intcoff},"kinetic_energy_kwargs": {"prefactor": KE_prefactor}, "periodic_lattice": potential_lattice,"periodic_potential_kwargs": {"coefficients": pp_coffs, "phases": pp_phases},"Bfield_lattice": potential_lattice,"Bfield_kwargs" : {"flux": -0.23,"threadedflux": threadedflux}}
  cfg.targetmom.mom = None
  cfg.network.network_type = "psiformer" 
  cfg.network.complex = True
  cfg.network.psiformer.num_layers = 4
  cfg.network.psiformer.num_heads = 6
  cfg.network.psiformer.heads_dim = 64
  cfg.network.psiformer.mlp_hidden_dims  = (256,)
  cfg.network.determinants = 4
  cfg.batch_size = 1024
  cfg.optim.iterations = 200000
  cfg.optim.lr.rate = 0.0001
  #cfg.optim.lr.decay = 0.0
  #cfg.optim.lr.decay = 1.5
  #cfg.optim.lr.delay = 1.0
  #cfg.mcmc.move_width = 2.0
  #cfg.mcmc.init_width = 3.0
  cfg.initialization.donor_filename = filename
  cfg.initialization.flatten_num_devices = False
  cfg.initialization.ignore_batch = False
  cfg.targetmom.kwargs = {"abs_lattice": Tmatrix, "unit_cell_vectors": np.array([a1,a2]), "logsumtrick": True}
  #cfg.initialization.modifications = ['orbital-rnd']
  #cfg.log.restore_path = 'ferminet_2025_08_24_11:33:55 copy'
  cfg.log.save_path = f"/data/ahmed95/NN_minimalChern/8particles_withflux/{flux2}"
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
  return cfg

################### Training script #####################
parser = argparse.ArgumentParser(description="Pass arguments to get_config.")
parser.add_argument("--flux1", type=float, required=True, help="First flux value (float).")
parser.add_argument("--flux2", type=float, required=True, help="Second flux value (float).")
parser.add_argument("--filename", type=str, required=True, help="Filename for initialization.")

args = parser.parse_args()

# Extract arguments
flux1 = args.flux1
flux2 = args.flux2
filename = args.filename

# Call get_config with the parsed arguments
cfg = get_config(flux1, flux2, filename)

# Start training
train.train(cfg)