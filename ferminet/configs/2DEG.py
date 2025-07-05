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

"""Unpolarised 14 electron simple cubic homogeneous electron gas."""

from ferminet import base_config
from ferminet.pbc import envelopes
from ferminet.utils import system

import numpy as np


def _sc_lattice_vecs(rs: float, nelec: int) -> np.ndarray:
  """Returns simple cubic lattice vectors with Wigner-Seitz radius rs."""
  area = np.pi * (rs**2) * nelec
  length = area**(1 / 2)
  return length * np.eye(2)


def get_config():
  # Get default options.
  cfg = base_config.default()
  cfg.system.electrons = (4, 0)
  cfg.system.ndim = 2
  # A ghost atom at the origin defines one-electron coordinate system.
  # Element 'X' is a dummy nucleus with zero charge
  cfg.system.molecule = [system.Atom("X", (0., 0.,))]
  # Pretraining is not currently implemented for systems in PBC
  cfg.pretrain.method = None

  lattice = _sc_lattice_vecs(1.0, sum(cfg.system.electrons))
  #kpoints = envelopes.make_kpoints(lattice, cfg.system.electrons)

  cfg.system.make_local_energy_fn = "ferminet.pbc.Hamiltonian_2DEG.local_energy"
  cfg.system.make_local_energy_kwargs = {"lattice": lattice, "heg": True,"potential_kwargs": {"laplacian_method": "folx"}}
  cfg.network.network_type = "psiformer"
  cfg.network.complex = False
  cfg.batch_size = 1000
  cfg.network.make_feature_layer_fn = (
      "ferminet.pbc.feature_layer.make_pbc_feature_layer")
  cfg.network.make_feature_layer_kwargs = {
      "lattice": lattice,
      "include_r_ae": False,
  }
  cfg.network.jastrow = 'NONE'
  cfg.network.make_envelope_fn = ( "ferminet.envelopes.make_null_envelope" )
  #cfg.network.make_envelope_kwargs = {"kpoints": kpoints}
  cfg.network.full_det = True
  return cfg
