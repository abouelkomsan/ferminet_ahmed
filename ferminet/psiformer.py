# Copyright 2023 DeepMind Technologies Limited.
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
# limitations under the License.

"""Attention-based networks for FermiNet."""

from typing import Mapping, Optional, Sequence, Tuple, Union

import attr
import chex
from ferminet import envelopes
from ferminet import jastrows
from ferminet import network_blocks
from ferminet import networks
from ferminet import ellipticfunctions
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
from ferminet import targetmom


@attr.s(auto_attribs=True, kw_only=True)
class PsiformerOptions(networks.BaseNetworkOptions):
  """Options controlling the Psiformer part of the network architecture.

  Attributes:
    num_layers: Number of self-attention layers.
    num_heads: Number of multihead self-attention heads.
    heads_dim: Embedding dimension for each self-attention head.
    mlp_hidden_dims: Tuple of sizes of hidden dimension of the MLP. Note that
      this does not include the final projection to the embedding dimension.
    use_layer_norm: If true, include a layer norm on both attention and MLP.
  """

  num_layers: int = 2
  num_heads: int = 4
  heads_dim: int = 64
  mlp_hidden_dims: Tuple[int, ...] = (256,)
  use_layer_norm: bool = False


def make_layer_norm() ->...:
  """Implementation of LayerNorm."""

  def init(param_shape: int) -> Mapping[str, jnp.ndarray]:
    params = {}
    params['scale'] = jnp.ones(param_shape)
    params['offset'] = jnp.zeros(param_shape)
    return params

  def apply(params: networks.ParamTree,
            inputs: jnp.ndarray,
            axis: int = -1) -> jnp.ndarray:
    mean = jnp.mean(inputs, axis=axis, keepdims=True)
    variance = jnp.var(inputs, axis=axis, keepdims=True)
    eps = 1e-5
    inv = params['scale'] * jax.lax.rsqrt(variance + eps)
    return inv * (inputs - mean) + params['offset']

  return init, apply


def make_multi_head_attention(num_heads: int, heads_dim: int) ->...:
  """FermiNet-style version of MultiHeadAttention."""

  # Linear layer plus reshape final dimensions to num_heads, heads_dim.
  def linear_projection(x: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    y = jnp.dot(x, weights)
    return y.reshape(*x.shape[:-1], num_heads, heads_dim)

  def init(key: chex.PRNGKey,
           q_d: int,
           kv_d: int,
           output_channels: Optional[int] = None) -> Mapping[str, jnp.ndarray]:

    # Dimension of attention projection.
    qkv_hiddens = num_heads * heads_dim
    if not output_channels:
      output_channels = qkv_hiddens

    key, *subkeys = jax.random.split(key, num=4)
    params = {}
    params['q_w'] = network_blocks.init_linear_layer(
        subkeys[0], in_dim=q_d, out_dim=qkv_hiddens, include_bias=False)['w']
    params['k_w'] = network_blocks.init_linear_layer(
        subkeys[1], in_dim=kv_d, out_dim=qkv_hiddens, include_bias=False)['w']
    params['v_w'] = network_blocks.init_linear_layer(
        subkeys[2], in_dim=kv_d, out_dim=qkv_hiddens, include_bias=False)['w']

    key, subkey = jax.random.split(key)
    params['attn_output'] = network_blocks.init_linear_layer(
        subkey, in_dim=qkv_hiddens, out_dim=output_channels,
        include_bias=False)['w']

    return params

  def apply(params: networks.ParamTree, query: jnp.ndarray, key: jnp.ndarray,
            value: jnp.ndarray) -> jnp.ndarray:
    """Computes MultiHeadAttention with keys, queries and values.

    Args:
      params: Parameters for attention embeddings.
      query: Shape [..., q_index_dim, q_d]
      key: Shape [..., kv_index_dim, kv_d]
      value: Shape [..., kv_index_dim, kv_d]

    Returns:
      A projection of attention-weighted value projections.
      Shape [..., q_index_dim, output_channels]
    """

    # Projections for q, k, v.
    # Output shape: [..., index_dim, num_heads, heads_dim].
    q = linear_projection(query, params['q_w'])
    k = linear_projection(key, params['k_w'])
    v = linear_projection(value, params['v_w'])

    attn_logits = jnp.einsum('...thd,...Thd->...htT', q, k)
    scale = 1. / np.sqrt(heads_dim)
    attn_logits *= scale

    attn_weights = jax.nn.softmax(attn_logits)

    attn = jnp.einsum('...htT,...Thd->...thd', attn_weights, v)

    # Concatenate attention matrix of all heads into a single vector.
    # Shape [..., q_index_dim, num_heads * heads_dim]
    attn = jnp.reshape(attn, (*query.shape[:-1], -1))

    # Apply a final projection to get the final embeddings.
    # Output shape: [..., q_index_dim, output_channels]
    return network_blocks.linear_layer(attn, params['attn_output'])

  return init, apply


def make_mlp() ->...:
  """Construct MLP, with final linear projection to embedding size."""

  def init(key: chex.PRNGKey, mlp_hidden_dims: Tuple[int, ...],
           embed_dim: int) -> Sequence[networks.Param]:
    params = []
    dims_one_in = [embed_dim, *mlp_hidden_dims]
    dims_one_out = [*mlp_hidden_dims, embed_dim]
    for i in range(len(dims_one_in)):
      key, subkey = jax.random.split(key)
      params.append(
          network_blocks.init_linear_layer(
              subkey,
              in_dim=dims_one_in[i],
              out_dim=dims_one_out[i],
              include_bias=True))
    return params

  def apply(params: Sequence[networks.Param],
            inputs: jnp.ndarray) -> jnp.ndarray:
    x = inputs
    for i in range(len(params)):
      x = jax.nn.gelu(network_blocks.linear_layer(x, **params[i]))
      #x = jax.nn.silu(network_blocks.linear_layer(x, **params[i]))
    return x

  return init, apply


def make_self_attention_block(num_layers: int,
                              num_heads: int,
                              heads_dim: int,
                              mlp_hidden_dims: Tuple[int, ...],
                              use_layer_norm: bool = False) ->...:
  """Create a QKV self-attention block."""
  attention_init, attention_apply = make_multi_head_attention(
      num_heads, heads_dim)
  if use_layer_norm:
    layer_norm_init, layer_norm_apply = make_layer_norm()
  mlp_init, mlp_apply = make_mlp()

  def init(key: chex.PRNGKey, qkv_d: int) -> networks.ParamTree:
    attn_dim = qkv_d
    params = {}
    attn_params = []
    ln_params = []
    mlp_params = []

    for _ in range(num_layers):
      key, attn_key, mlp_key = jax.random.split(key, 3)
      attn_params.append(
          attention_init(
              attn_key, q_d=qkv_d, kv_d=qkv_d, output_channels=attn_dim))
      if use_layer_norm:
        ln_params.append([layer_norm_init(attn_dim), layer_norm_init(attn_dim)])
      mlp_params.append(mlp_init(mlp_key, mlp_hidden_dims, attn_dim))

    params['attention'] = attn_params
    params['ln'] = ln_params
    params['mlp'] = mlp_params

    return params

  def apply(params: networks.ParamTree, qkv: jnp.ndarray) -> jnp.ndarray:
    x = qkv
    for layer in range(num_layers):
      attn_output = attention_apply(params['attention'][layer], x, x, x)

      # Residual + optional LayerNorm.
      x = x + attn_output
      if use_layer_norm:
        x = layer_norm_apply(params['ln'][layer][0], x)

      # MLP
      assert isinstance(params['mlp'][layer], (tuple, list))
      mlp_output = mlp_apply(params['mlp'][layer], x)

      # Residual + optional LayerNorm.
      x = x + mlp_output
      if use_layer_norm:
        x = layer_norm_apply(params['ln'][layer][1], x)

    return x

  return init, apply


def make_psiformer_layers(
    nspins: Tuple[int, ...],
    natoms: int,
    options: PsiformerOptions,
) -> Tuple[networks.InitLayersFn, networks.ApplyLayersFn]:
  """Creates the permutation-equivariant layers for the Psiformer.

  Args:
    nspins: Tuple with number of spin up and spin down electrons.
    natoms: number of atoms.
    options: network options.

  Returns:
    Tuple of init, apply functions.
  """
  del nspins, natoms  # Unused.

  # Attention network.
  attn_dim = options.num_heads * options.heads_dim
  self_attn_init, self_attn_apply = make_self_attention_block(
      num_layers=options.num_layers,
      num_heads=options.num_heads,
      heads_dim=options.heads_dim,
      mlp_hidden_dims=options.mlp_hidden_dims,
      use_layer_norm=options.use_layer_norm,
  )

  def init(key: chex.PRNGKey) -> Tuple[int, networks.ParamTree]:
    """Returns tuple of output dimension from the final layer and parameters."""
    params = {}
    key, subkey = jax.random.split(key)
    feature_dims, params['input'] = options.feature_layer.init()
    one_electron_feature_dim, _ = feature_dims
    # Concatenate spin of each electron with other one-electron features.
    feature_dim = one_electron_feature_dim + 1

    # Map to Attention dim.
    key, subkey = jax.random.split(key)
    params['embed'] = network_blocks.init_linear_layer(
        subkey, in_dim=feature_dim, out_dim=attn_dim, include_bias=False
    )['w']

    # Attention block params.
    key, subkey = jax.random.split(key)
    params.update(self_attn_init(key, attn_dim))

    return attn_dim, params

  def apply(
      params,
      *,
      ae: jnp.ndarray,
      r_ae: jnp.ndarray,
      ee: jnp.ndarray,
      r_ee: jnp.ndarray,
      spins: jnp.ndarray,
      charges: jnp.ndarray,
  ) -> jnp.ndarray:
    """Applies the Psiformer interaction layers to a walker configuration.

    Args:
      params: parameters for the interaction and permuation-equivariant layers.
      ae: electron-nuclear vectors.
      r_ae: electron-nuclear distances.
      ee: electron-electron vectors.
      r_ee: electron-electron distances.
      spins: spin of each electron.
      charges: nuclear charges.

    Returns:
      Array of shape (nelectron, output_dim), where the output dimension,
      output_dim, is given by init, and is suitable for projection into orbital
      space.
    """
    del charges  # Unused.

    # Only one-electron features are used by the Psiformer.
    ae_features, _ = options.feature_layer.apply(
        ae=ae, r_ae=r_ae, ee=ee, r_ee=r_ee, **params['input']
    )

    # For the Psiformer, the spin feature is required for correct permutation
    # equivariance.
    ae_features = jnp.concatenate((ae_features, spins[..., None]), axis=-1)

    features = ae_features  # Just 1-electron stream for now.

    # Embed into attention dimension.
    x = jnp.dot(features, params['embed'])

    return self_attn_apply(params, x)

  return init, apply


def make_fermi_net(
    nspins: Tuple[int, ...],
    charges: jnp.ndarray,
    *,
    ndim: int = 3,
    determinants: int = 16,
    states: int = 0,
    envelope: Optional[envelopes.Envelope] = None,
    feature_layer: Optional[networks.FeatureLayer] = None,
    jastrow: Union[str, jastrows.JastrowType] = jastrows.JastrowType.SIMPLE_EE,
    jastrow_kwargs: dict = {},
    complex_output: bool = False,
    bias_orbitals: bool = False,
    rescale_inputs: bool = False,
    # Psiformer-specific kwargs below.
    num_layers: int,
    num_heads: int,
    heads_dim: int,
    mlp_hidden_dims: Tuple[int, ...],
    use_layer_norm: bool,
    pbc_lattice: jnp.ndarray,
) -> networks.Network:
  """Psiformer with stacked Self Attention layers.

  Includes standard envelope and determinants.

  Args:
    nspins: Tuple of the number of spin-up and spin-down electrons.
    charges: (natom) array of atom nuclear charges.
    ndim: Dimension of the system. Change only with caution.
    determinants: Number of determinants.
    states: Number of outputs, one per excited (or ground) state. Ignored if 0.
    envelope: Envelope to use to impose orbitals go to zero at infinity.
    feature_layer: Input feature construction.
    jastrow: Type of Jastrow factor if used, or 'simple_ee' if 'default'.
    complex_output: If true, the wavefunction output is complex-valued.
    bias_orbitals: If true, include a bias in the final linear layer to shape
      the outputs into orbitals.
    rescale_inputs: If true, rescale the inputs so they grow as log(|r|).
    num_layers: Number of stacked self-attention layers.
    num_heads: Number of self-attention heads.
    heads_dim: Embedding dimension per-head for self-attention.
    mlp_hidden_dims: Tuple of hidden dimensions of the MLP.
    use_layer_norm: If true, use layer_norm on both attention and MLP.

  Returns:
    Network object containing init, apply, orbitals, options, where init and
    apply are callables which initialise the network parameters and apply the
    network respectively, orbitals is a callable which applies the network up to
    the orbitals, and options specifies the settings used in the network.
  """

  if not envelope:
    envelope = envelopes.make_isotropic_envelope()

  if not feature_layer:
    natoms = charges.shape[0]
    feature_layer = networks.make_ferminet_features(
        natoms, nspins, ndim=ndim, rescale_inputs=rescale_inputs
    )

  if isinstance(jastrow, str):
    if jastrow.upper() == 'DEFAULT':
      jastrow = jastrows.JastrowType.SIMPLE_EE
    else:
      jastrow = jastrows.JastrowType[jastrow.upper()]

  options = PsiformerOptions(
      ndim=ndim,
      determinants=determinants,
      states=states,
      envelope=envelope,
      feature_layer=feature_layer,
      jastrow=jastrow,
      jastrow_kwargs=jastrow_kwargs,
      complex_output=complex_output,
      bias_orbitals=bias_orbitals,
      full_det=True,  # Required for Psiformer.
      rescale_inputs=rescale_inputs,
      num_layers=num_layers,
      num_heads=num_heads,
      heads_dim=heads_dim,
      mlp_hidden_dims=mlp_hidden_dims,
      use_layer_norm=use_layer_norm,
      pbc_lattice = pbc_lattice,
  )  # pytype: disable=wrong-keyword-args

  psiformer_layers = make_psiformer_layers(nspins, charges.shape[0], options)

  orbitals_init, orbitals_apply = networks.make_orbitals(
      nspins=nspins,
      charges=charges,
      options=options,
      equivariant_layers=psiformer_layers,
  )

  def network_init(key: chex.PRNGKey) -> networks.ParamTree:
    return orbitals_init(key)

  def network_apply(
      params,
      pos: jnp.ndarray,
      spins: jnp.ndarray,
      atoms: jnp.ndarray,
      charges: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Forward evaluation of the Psiformer.

    Args:
      params: network parameter tree.
      pos: The electron positions, a 3N dimensional vector.
      spins: The electron spins, an N dimensional vector.
      atoms: Array with positions of atoms.
      charges: Array with nuclear charges.

    Returns:
      Output of antisymmetric neural network in log space, i.e. a tuple of sign
      of and log absolute value of the network evaluated at x.
    """
    orbitals = orbitals_apply(params, pos, spins, atoms, charges)
    if options.states:
      batch_logdet_matmul = jax.vmap(network_blocks.logdet_matmul, in_axes=0)
      orbitals = [
          jnp.reshape(orbital, (options.states, -1) + orbital.shape[1:])
          for orbital in orbitals
      ]
      result = batch_logdet_matmul(orbitals)
    else:
      result = network_blocks.logdet_matmul(orbitals)
    if 'state_scale' in params:
      # only used at inference time for excited states
      result = result[0], result[1] + params['state_scale']
    return result

  return networks.Network(
      options=options,
      init=network_init,
      apply=network_apply,
      orbitals=orbitals_apply,
  )


def make_fermi_net_with_zero_projection(
    nspins: Tuple[int, ...],
    charges: jnp.ndarray,
    *,
    ndim: int = 3,
    determinants: int = 16,
    states: int = 0,
    envelope: Optional[envelopes.Envelope] = None,
    feature_layer: Optional[networks.FeatureLayer] = None,
    jastrow: Union[str, jastrows.JastrowType] = jastrows.JastrowType.SIMPLE_EE,
    jastrow_kwargs: dict = {},
    complex_output: bool = False,
    bias_orbitals: bool = False,
    rescale_inputs: bool = False,
    # Psiformer-specific kwargs:
    num_layers: int,
    num_heads: int,
    heads_dim: int,
    mlp_hidden_dims: Tuple[int, ...],
    use_layer_norm: bool,
    pbc_lattice: jnp.ndarray,
    # New: number of holomorphic zeros per orbital used by the LLL envelope.
    N_holo: int,
) -> networks.Network:
  """Psiformer with stacked Self Attention layers and NN-projected zeros.

  Differences from make_fermi_net:
    * Uses make_orbitals_with_zero_projection instead of the standard
      make_orbitals.
    * The PRE_DETERMINANT envelope (typically LLL) is still used, but its zeros
      are *not* free parameters: they are supplied at apply-time from a linear
      projection of the final equivariant features.
  """

  if envelope is None:
    raise ValueError(
        "make_fermi_net_with_zero_projection expects an explicit PRE_DETERMINANT "
        "envelope (e.g. make_LLL_envelope_2d_trainable_zeros_mixed4)."
    )

  if envelope.apply_type != envelopes.EnvelopeType.PRE_DETERMINANT:
    raise ValueError(
        "Envelope for make_fermi_net_with_zero_projection must be PRE_DETERMINANT."
    )

  if feature_layer is None:
    natoms = charges.shape[0]
    feature_layer = networks.make_ferminet_features(
        natoms, nspins, ndim=ndim, rescale_inputs=rescale_inputs
    )

  if isinstance(jastrow, str):
    if jastrow.upper() == 'DEFAULT':
      jastrow = jastrows.JastrowType.SIMPLE_EE
    else:
      jastrow = jastrows.JastrowType[jastrow.upper()]

  options = PsiformerOptions(
      ndim=ndim,
      determinants=determinants,
      states=states,
      envelope=envelope,
      feature_layer=feature_layer,
      jastrow=jastrow,
      jastrow_kwargs=jastrow_kwargs,
      complex_output=complex_output,
      bias_orbitals=bias_orbitals,
      full_det=True,  # Required for Psiformer.
      rescale_inputs=rescale_inputs,
      num_layers=num_layers,
      num_heads=num_heads,
      heads_dim=heads_dim,
      mlp_hidden_dims=mlp_hidden_dims,
      use_layer_norm=use_layer_norm,
      pbc_lattice=pbc_lattice,
  )  # pytype: disable=wrong-keyword-args

  psiformer_layers = make_psiformer_layers(nspins, charges.shape[0], options)

  # Use the modified orbital builder that ties zeros to the final layer.
  orbitals_init, orbitals_apply = networks.make_orbitals_with_zero_projection(
      nspins=nspins,
      charges=charges,
      options=options,
      equivariant_layers=psiformer_layers,
      N_holo=N_holo,
  )

  def network_init(key: chex.PRNGKey) -> networks.ParamTree:
    return orbitals_init(key)

  def network_apply(
      params,
      pos: jnp.ndarray,
      spins: jnp.ndarray,
      atoms: jnp.ndarray,
      charges: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Forward evaluation of the Psiformer with NN-projected zeros."""
    orbitals = orbitals_apply(params, pos, spins, atoms, charges)
    if options.states:
      batch_logdet_matmul = jax.vmap(network_blocks.logdet_matmul, in_axes=0)
      orbitals = [
          jnp.reshape(orbital, (options.states, -1) + orbital.shape[1:])
          for orbital in orbitals
      ]
      result = batch_logdet_matmul(orbitals)
    else:
      result = network_blocks.logdet_matmul(orbitals)
    if 'state_scale' in params:
      result = result[0], result[1] + params['state_scale']
    return result

  return networks.Network(
      options=options,
      init=network_init,
      apply=network_apply,
      orbitals=orbitals_apply,
  )


def make_fermi_net_with_zero_projection_pooled(
    nspins: Tuple[int, ...],
    charges: jnp.ndarray,
    *,
    ndim: int = 3,
    determinants: int = 16,
    states: int = 0,
    envelope: Optional[envelopes.Envelope] = None,
    feature_layer: Optional[networks.FeatureLayer] = None,
    jastrow: Union[str, jastrows.JastrowType] = jastrows.JastrowType.SIMPLE_EE,
    jastrow_kwargs: dict = {},
    complex_output: bool = False,
    bias_orbitals: bool = False,
    rescale_inputs: bool = False,
    # Psiformer-specific kwargs:
    num_layers: int,
    num_heads: int,
    heads_dim: int,
    mlp_hidden_dims: Tuple[int, ...],
    use_layer_norm: bool,
    pbc_lattice: jnp.ndarray,
    # New: number of holomorphic zeros per orbital used by the LLL envelope.
    N_holo: int,
) -> networks.Network:
  """Psiformer with stacked Self Attention layers and NN-projected zeros.

  Differences from make_fermi_net:
    * Uses make_orbitals_with_zero_projection instead of the standard
      make_orbitals.
    * The PRE_DETERMINANT envelope (typically LLL) is still used, but its zeros
      are *not* free parameters: they are supplied at apply-time from a linear
      projection of the final equivariant features.
  """

  if envelope is None:
    raise ValueError(
        "make_fermi_net_with_zero_projection expects an explicit PRE_DETERMINANT "
        "envelope (e.g. make_LLL_envelope_2d_trainable_zeros_mixed4)."
    )

  if envelope.apply_type != envelopes.EnvelopeType.PRE_DETERMINANT:
    raise ValueError(
        "Envelope for make_fermi_net_with_zero_projection must be PRE_DETERMINANT."
    )

  if feature_layer is None:
    natoms = charges.shape[0]
    feature_layer = networks.make_ferminet_features(
        natoms, nspins, ndim=ndim, rescale_inputs=rescale_inputs
    )

  if isinstance(jastrow, str):
    if jastrow.upper() == 'DEFAULT':
      jastrow = jastrows.JastrowType.SIMPLE_EE
    else:
      jastrow = jastrows.JastrowType[jastrow.upper()]

  options = PsiformerOptions(
      ndim=ndim,
      determinants=determinants,
      states=states,
      envelope=envelope,
      feature_layer=feature_layer,
      jastrow=jastrow,
      jastrow_kwargs=jastrow_kwargs,
      complex_output=complex_output,
      bias_orbitals=bias_orbitals,
      full_det=True,  # Required for Psiformer.
      rescale_inputs=rescale_inputs,
      num_layers=num_layers,
      num_heads=num_heads,
      heads_dim=heads_dim,
      mlp_hidden_dims=mlp_hidden_dims,
      use_layer_norm=use_layer_norm,
      pbc_lattice=pbc_lattice,
  )  # pytype: disable=wrong-keyword-args

  psiformer_layers = make_psiformer_layers(nspins, charges.shape[0], options)

  # Use the modified orbital builder that ties zeros to the final layer.
  orbitals_init, orbitals_apply = networks.make_orbitals_with_zero_projection_pooled(
      nspins=nspins,
      charges=charges,
      options=options,
      equivariant_layers=psiformer_layers,
      N_holo=N_holo,
  )

  def network_init(key: chex.PRNGKey) -> networks.ParamTree:
    return orbitals_init(key)

  def network_apply(
      params,
      pos: jnp.ndarray,
      spins: jnp.ndarray,
      atoms: jnp.ndarray,
      charges: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Forward evaluation of the Psiformer with NN-projected zeros."""
    orbitals = orbitals_apply(params, pos, spins, atoms, charges)
    if options.states:
      batch_logdet_matmul = jax.vmap(network_blocks.logdet_matmul, in_axes=0)
      orbitals = [
          jnp.reshape(orbital, (options.states, -1) + orbital.shape[1:])
          for orbital in orbitals
      ]
      result = batch_logdet_matmul(orbitals)
    else:
      result = network_blocks.logdet_matmul(orbitals)
    if 'state_scale' in params:
      result = result[0], result[1] + params['state_scale']
    return result

  return networks.Network(
      options=options,
      init=network_init,
      apply=network_apply,
      orbitals=orbitals_apply,
  )

def make_fermi_net_with_zero_projection_momind_0(
    nspins: Tuple[int, ...],
    charges: jnp.ndarray,
    *,
    ndim: int = 3,
    determinants: int = 16,
    states: int = 0,
    envelope: Optional[envelopes.Envelope] = None,
    feature_layer: Optional[networks.FeatureLayer] = None,
    jastrow: Union[str, jastrows.JastrowType] = jastrows.JastrowType.SIMPLE_EE,
    jastrow_kwargs: dict = {},
    complex_output: bool = False,
    bias_orbitals: bool = False,
    rescale_inputs: bool = False,
    # Psiformer-specific kwargs:
    num_layers: int,
    num_heads: int,
    heads_dim: int,
    mlp_hidden_dims: Tuple[int, ...],
    use_layer_norm: bool,
    pbc_lattice: jnp.ndarray,
    # vortexformer-specific:
    N_holo: int,
    # magnetic Bloch / momentum:
    momind: int,
    mom_kwargs: dict,
) -> networks.Network:
  """Vortexformer (Psiformer + NN-projected zeros) with magnetic-momentum
  projection using the many-body magnetic translation

    Single-particle:  T(R) f(r) = exp(i r×R / 2) f(r+R),   (ℓ_B = 1)
    Many-body COM:    T_MB(R) Ψ({r_i}) = exp(i/2 Σ_i r_i×R) Ψ({r_i+R})

  and a Bloch projector

    Ψ_k({r_i}) = Σ_R exp[-i k_eff·R] T_MB(R) Ψ({r_i}),

  where R runs over magnetic unit-cell translations inside the supercell
  (COM translations), and

    k_eff = k + Δk,

  with k = m g1 + n g2 (discrete MBZ point) and Δk determined by the
  boundary twists η_{L1}, η_{L2} via

    T(L1) ψ = (η_{L1})^{Nφ} ψ,   T(L2) ψ = (η_{L2})^{Nφ} ψ,

  ⇒ Δk = δ1 g1 + δ2 g2,
     δ1 = Nφ θ_{L1}/(2π),  δ2 = Nφ θ_{L2}/(2π),  η_{Li} = e^{i θ_{Li}}.
  """

  if envelope is None:
    raise ValueError(
        "make_fermi_net_with_zero_projection_momind_0 expects an explicit "
        "PRE_DETERMINANT envelope."
    )

  if envelope.apply_type != envelopes.EnvelopeType.PRE_DETERMINANT:
    raise ValueError(
        "Envelope for make_fermi_net_with_zero_projection_momind_0 must "
        "be PRE_DETERMINANT."
    )

  if mom_kwargs is None:
    raise ValueError("mom_kwargs (with abs_lattice, unit_cell_vectors) must be provided.")

  if feature_layer is None:
    natoms = charges.shape[0]
    feature_layer = networks.make_ferminet_features(
        natoms, nspins, ndim=ndim, rescale_inputs=rescale_inputs
    )

  if isinstance(jastrow, str):
    if jastrow.upper() == 'DEFAULT':
      jastrow = jastrows.JastrowType.SIMPLE_EE
    else:
      jastrow = jastrows.JastrowType[jastrow.upper()]

  options = PsiformerOptions(
      ndim=ndim,
      determinants=determinants,
      states=states,
      envelope=envelope,
      feature_layer=feature_layer,
      jastrow=jastrow,
      jastrow_kwargs=jastrow_kwargs,
      complex_output=complex_output,
      bias_orbitals=bias_orbitals,
      full_det=True,
      rescale_inputs=rescale_inputs,
      num_layers=num_layers,
      num_heads=num_heads,
      heads_dim=heads_dim,
      mlp_hidden_dims=mlp_hidden_dims,
      use_layer_norm=use_layer_norm,
      pbc_lattice=pbc_lattice,
  )  # pytype: disable=wrong-keyword-args

  psiformer_layers = make_psiformer_layers(nspins, charges.shape[0], options)

  orbitals_init, orbitals_apply = networks.make_orbitals_with_zero_projection(
      nspins=nspins,
      charges=charges,
      options=options,
      equivariant_layers=psiformer_layers,
      N_holo=N_holo,
  )

  def network_init(key: chex.PRNGKey) -> networks.ParamTree:
    return orbitals_init(key)

  # ---------------------------------------------------------------------------
  # Magnetic lattice / momentum data (via targetmom.*)
  # ---------------------------------------------------------------------------

  abs_lattice = mom_kwargs['abs_lattice']              # 2x2 Tmatrix of supercell
  unit_cell_vectors = mom_kwargs['unit_cell_vectors']  # (2,2): a1,a2 magnetic cell

  # Supercell lattice vectors T1,T2 (columns)
  lattice = targetmom.lattice_vecs(
      unit_cell_vectors[0], unit_cell_vectors[1], abs_lattice
  )  # shape (2,2)
  T1 = lattice[:, 0]
  T2 = lattice[:, 1]

  # Reciprocal of supercell, T_i·G_j = 2π δ_ij
  reciprocal = targetmom.reciprocal_vecs(
      unit_cell_vectors[0],
      unit_cell_vectors[1],
      abs_lattice,
  )  # shape (2,2), columns G1,G2

  # g1,g2 following targetmom.g1g2 convention
  g1, g2, _, _, _ = targetmom.g1g2(
      abs_lattice, reciprocal[:, 0], reciprocal[:, 1]
  )

  # Integer k-labels in k-space (2D integer lattice inside MBZ)
  klabels = targetmom.kpoints(abs_lattice)   # (N_k, 2)
  klabel = klabels[momind]                   # chosen (k1,k2) integers

  # Bare physical k-vector: k = k1 g1 + k2 g2 (targetmom.mn)
  kvec = targetmom.mn(klabel, g1, g2)        # shape (2,)

  # Magnetic cell vectors a1,a2 (1 flux quantum per cell)
  a1 = unit_cell_vectors[0]                 # (2,)
  a2 = unit_cell_vectors[1]                 # (2,)

  # Compute Nφ = area_super / area_cell
  area_cell = jnp.abs(a1[0] * a2[1] - a1[1] * a2[0])   # |a1 x a2|
  area_super = jnp.abs(T1[0] * T2[1] - T1[1] * T2[0])  # |T1 x T2|

  area_cell_f = float(area_cell)
  area_super_f = float(area_super)
  Nphi = int(round(area_super_f / area_cell_f))
  if Nphi <= 0:
    raise ValueError(f"Invalid Nphi={Nphi} from areas: cell={area_cell_f}, super={area_super_f}")

  # ---------------------------------------------------------------------------
  # Boundary twists and k-shift Δk
  # ---------------------------------------------------------------------------
  # Expect mom_kwargs to possibly contain eta_L1, eta_L2 (complex phases).
  # If not provided, default to no twist (eta = 1).
  # eta_L_common = mom_kwargs.get('eta_L', 1.0 + 0.0j)
  # eta_L1 = mom_kwargs.get('eta_L1', eta_L_common)
  # eta_L2 = mom_kwargs.get('eta_L2', eta_L_common)
  Ne = sum(nspins)

  theta_L1 = 0.0# Ne  # in (-π, π]
  theta_L2 = 0.0#* Ne

  # δ1 = Nφ θ_{L1} / (2π),  δ2 = Nφ θ_{L2} / (2π)
  delta1 = Nphi * theta_L1 / (2.0 * np.pi)
  delta2 = Nphi * theta_L2 / (2.0 * np.pi)

  # Δk = δ1 g1 + δ2 g2
  kshift = delta1 * g1 + delta2 * g2          # (2,)
  kvec_eff = kvec + kshift                    # shifted momentum

  # All translations R inside the supercell and their integer (m,n) labels
  translations, xy_pairs = targetmom.find_all_translations_in_supercell(
      lattice, unit_cell_vectors
  )  # translations: (N_T,2), xy_pairs: (N_T,2) with integers (m,n)

  num_translations = translations.shape[0]
  r_int = xy_pairs[:, 0].astype(jnp.int32)
  s_int = xy_pairs[:, 1].astype(jnp.int32)

  # π r s factor
  pi_rs = jnp.pi * (r_int + s_int).astype(jnp.float32)  # (N_T,)
  # Number of electrons

  # ---------------------------------------------------------------------------
  # Helper: phases for T_MB(R) and Bloch factor e^{-i k_eff·R}
  # ---------------------------------------------------------------------------

  def compute_phases_for_translations(pos_xy: jnp.ndarray) -> jnp.ndarray:
    """Return phases[R] for all translations R, given electron positions pos_xy.

    Many-body COM translation (ℓ_B=1):
      T_MB(R) Ψ({r_i}) = exp(i/2 Σ_i r_i×R) Ψ({r_i+R})

    Bloch weight with shifted momentum:
      e^{-i k_eff·R}

    Total phase exponent (in the 'i * ...' of exp) is:

      phase(R; {r_i}) = -k_eff·R + (1/2) Σ_i (r_i×R)
    """
    # r: (1,Ne,2), R: (N_T,1,2)
    r = pos_xy[None, :, :]           # (1,Ne,2)
    R = translations[:, None, :]     # (N_T,1,2)

    # r_i × R = r_x R_y - r_y R_x
    cross = r[..., 0] * R[..., 1] - r[..., 1] * R[..., 0]   # (N_T,Ne)

    # Σ_i r_i×R
    sum_cross = jnp.sum(cross, axis=1)                      # (N_T,)

    # (1/2) Σ_i r_i×R
    mag_phase = 0.5 * sum_cross                             # (N_T,)

    # Bloch factor: -k_eff·R
    bloch_phase = -jnp.dot(translations, kvec_eff)          # (N_T,)

    total_phase = bloch_phase + mag_phase  + pi_rs      # (N_T,)
    return total_phase

  # ---------------------------------------------------------------------------
  # Apply: magnetic Bloch vortexformer
  # ---------------------------------------------------------------------------

  def network_apply(
      params,
      pos: jnp.ndarray,
      spins: jnp.ndarray,
      atoms: jnp.ndarray,
      charges: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Evaluate the magnetic-momentum-projected vortexformer Ψ_k at positions `pos`."""

    assert options.states == 0, "Momentum-projected version currently assumes states=0."

    full_ndim = options.ndim

    # Interpret pos in the same way orbitals_apply expects.
    if pos.ndim == 1:
      if pos.shape[0] != Ne * full_ndim:
        raise ValueError(
            f"Flattened pos has length {pos.shape[0]}, expected {Ne * full_ndim}."
        )
      pos_full = pos.reshape(Ne, full_ndim)   # (Ne, ndim)
      original_shape = 'flat'
    elif pos.ndim == 2:
      if pos.shape != (Ne, full_ndim):
        raise ValueError(
            f"pos has shape {pos.shape}, expected ({Ne}, {full_ndim})."
        )
      pos_full = pos                          # (Ne, ndim)
      original_shape = 'matrix'
    else:
      raise ValueError("pos must be 1D ((Ne*ndim,)) or 2D ((Ne,ndim)).")

    # xy coordinates (where magnetic structure lives)
    pos_xy = pos_full[:, :2]                  # (Ne,2)

    # Total phases per translation for this configuration:
    #   -k_eff·R  + (1/2 Σ_i r_i×R)
    phases_per_translation = compute_phases_for_translations(pos_xy)  # (N_T,)

    # For each translation R, build COM-shifted coordinates:
    #   r_i → r_i + R
    total_shifts_xy = translations            # (N_T,2)

    def translated_config(i):
      base = pos_full                       # (Ne, full_ndim)
      shift_xy = total_shifts_xy[i]         # (2,)
      new_xy = pos_xy + shift_xy[None, :]   # (Ne,2)
      if full_ndim == 2:
        out = new_xy
      else:
        higher = base[:, 2:]               # (Ne, full_ndim-2)
        out = jnp.concatenate([new_xy, higher], axis=1)
      return out

    translated_full = jax.vmap(translated_config)(
        jnp.arange(num_translations)
    )  # (N_T,Ne,full_ndim)

    def eval_translation(i):
      if original_shape == 'flat':
        t_pos = translated_full[i].reshape(Ne * full_ndim)
      else:
        t_pos = translated_full[i]

      orbitals = orbitals_apply(params, t_pos, spins, atoms, charges)
      phase_i, logdet_i = network_blocks.logdet_matmul(orbitals)

      # Multiply by COM + Bloch phase:
      phase_i = phase_i + phases_per_translation[i]
      return phase_i, logdet_i

    phases_i, logdets_i = jax.vmap(eval_translation)(
        jnp.arange(num_translations)
    )  # each shape (N_T,)

    # Complex log-sum-exp over translations
    max_logdet = jnp.max(logdets_i)
    shifted_exps = jnp.exp(
        logdets_i - max_logdet + 1j * phases_i
    )
    total_determinant = jnp.sum(shifted_exps) / num_translations

    overall_log = max_logdet + jnp.log(jnp.abs(total_determinant))
    overall_phase = jnp.angle(total_determinant)

    if 'state_scale' in params:
      overall_log = overall_log + params['state_scale']

    return overall_phase, overall_log

  return networks.Network(
      options=options,
      init=network_init,
      apply=network_apply,
      orbitals=orbitals_apply,
  )

def make_fermi_net_with_zero_projection_momind_COMK(
    nspins: Tuple[int, ...],
    charges: jnp.ndarray,
    *,
    ndim: int = 3,
    determinants: int = 16,
    states: int = 0,
    envelope: Optional[envelopes.Envelope] = None,
    feature_layer: Optional[networks.FeatureLayer] = None,
    jastrow: Union[str, jastrows.JastrowType] = jastrows.JastrowType.SIMPLE_EE,
    jastrow_kwargs: dict = {},
    complex_output: bool = False,
    bias_orbitals: bool = False,
    rescale_inputs: bool = False,
    # Psiformer-specific kwargs:
    num_layers: int,
    num_heads: int,
    heads_dim: int,
    mlp_hidden_dims: Tuple[int, ...],
    use_layer_norm: bool,
    pbc_lattice: jnp.ndarray,
    # vortexformer-specific:
    N_holo: int,
    # target COM momentum index:
    momind: int,
    mom_kwargs: dict,
) -> networks.Network:
  """Vortexformer with COM-momentum projection via:

       |Ψ_K> = T_COM(δ_K) |Ψ_0>,

     where |Ψ_0> is the k=(0,0) COM-projected seed from your original
     projector, and

       δ_K = (ℓ_B^2 / N_e) ( -K_y, K_x ),  K = k1 g1 + k2 g2.

     The seed projector is *exactly* the same structure as your
     make_fermi_net_with_zero_projection_momind_0, but hard-wired
     to the klabel = (0,0) COM point. Then we apply T_COM(δ_K) on top,
     in the correct operator order:

       Ψ_K(r) = e^{ i/2 Σ_i r_i×δ_K } Ψ_0(r + δ_K).
  """

  if envelope is None:
    raise ValueError(
        "make_fermi_net_with_zero_projection_momind_COMK expects an explicit "
        "PRE_DETERMINANT envelope."
    )

  if envelope.apply_type != envelopes.EnvelopeType.PRE_DETERMINANT:
    raise ValueError(
        "Envelope for make_fermi_net_with_zero_projection_momind_COMK must "
        "be PRE_DETERMINANT."
    )

  if mom_kwargs is None:
    raise ValueError("mom_kwargs (with abs_lattice, unit_cell_vectors) must be provided.")

  if feature_layer is None:
    natoms = charges.shape[0]
    feature_layer = networks.make_ferminet_features(
        natoms, nspins, ndim=ndim, rescale_inputs=rescale_inputs
    )

  if isinstance(jastrow, str):
    if jastrow.upper() == 'DEFAULT':
      jastrow = jastrows.JastrowType.SIMPLE_EE
    else:
      jastrow = jastrows.JastrowType[jastrow.upper()]

  options = PsiformerOptions(
      ndim=ndim,
      determinants=determinants,
      states=states,
      envelope=envelope,
      feature_layer=feature_layer,
      jastrow=jastrow,
      jastrow_kwargs=jastrow_kwargs,
      complex_output=complex_output,
      bias_orbitals=bias_orbitals,
      full_det=True,
      rescale_inputs=rescale_inputs,
      num_layers=num_layers,
      num_heads=num_heads,
      heads_dim=heads_dim,
      mlp_hidden_dims=mlp_hidden_dims,
      use_layer_norm=use_layer_norm,
      pbc_lattice=pbc_lattice,
  )  # pytype: disable=wrong-keyword-args

  psiformer_layers = make_psiformer_layers(nspins, charges.shape[0], options)

  orbitals_init, orbitals_apply = networks.make_orbitals_with_zero_projection(
      nspins=nspins,
      charges=charges,
      options=options,
      equivariant_layers=psiformer_layers,
      N_holo=N_holo,
  )

  def network_init(key: chex.PRNGKey) -> networks.ParamTree:
    return orbitals_init(key)

  # ---------------------------------------------------------------------------
  # Magnetic lattice / momentum (g1,g2, klabels, etc.)
  # ---------------------------------------------------------------------------

  abs_lattice = mom_kwargs['abs_lattice']              # 2x2 Tmatrix of supercell
  unit_cell_vectors = mom_kwargs['unit_cell_vectors']  # (2,2): a1,a2 magnetic cell

  lattice = targetmom.lattice_vecs(
      unit_cell_vectors[0], unit_cell_vectors[1], abs_lattice
  )  # shape (2,2)
  T1 = lattice[:, 0]
  T2 = lattice[:, 1]

  reciprocal = targetmom.reciprocal_vecs(
      unit_cell_vectors[0],
      unit_cell_vectors[1],
      abs_lattice,
  )  # shape (2,2), columns G1,G2

  g1, g2, _, _, _ = targetmom.g1g2(
      abs_lattice, reciprocal[:, 0], reciprocal[:, 1]
  )

  # All COM k-labels
  klabels = targetmom.kpoints(abs_lattice)   # (N_k, 2)

  # Target COM momentum label and vector
  klabel_target = klabels[momind]            # (k1,k2)
  Kvec = targetmom.mn(klabel_target, g1, g2) # (2,)

  # Find k=(0,0) label to use as SEED projector
  klabels_np = np.array(klabels)
  zero_mask = (klabels_np[:, 0] == 0) & (klabels_np[:, 1] == 0)
  idx_zero = int(np.where(zero_mask)[0][0])
  klabel_zero = klabels[idx_zero]

  # For the seed, we take effectively k=0 (any twists you want can be
  # encoded via pi_rs or by adding a constant kshift_zero here if needed).


  # Magnetic cell vectors and ℓ_B
  a1 = unit_cell_vectors[0]
  a2 = unit_cell_vectors[1]
  area_cell = jnp.abs(a1[0] * a2[1] - a1[1] * a2[0])
  area_super = jnp.abs(T1[0] * T2[1] - T1[1] * T2[0])

  ellB2 = area_cell / (2.0 * jnp.pi)   # |a1×a2| = 2πℓ_B^2
  Ne = sum(nspins)
  area_cell_f = float(area_cell)
  area_super_f = float(area_super)
  Nphi = int(round(area_super_f / area_cell_f))
  if Nphi <= 0:
    raise ValueError(f"Invalid Nphi={Nphi} from areas: cell={area_cell_f}, super={area_super_f}")

  kvec_zero = targetmom.mn(klabel_zero, g1, g2)  # (2,)
  theta_L1 = 0.0 #* Ne
  theta_L2 = 0.0 #* Ne

  # δ1 = Nφ θ_{L1} / (2π),  δ2 = Nφ θ_{L2} / (2π)
  delta1 = Nphi * theta_L1 / (2.0 * np.pi)
  delta2 = Nphi * theta_L2 / (2.0 * np.pi)

  # Δk = δ1 g1 + δ2 g2
  kshift = delta1 * g1 + delta2 * g2          # (2,)
  kvec_eff_zero = kshift      # treat seed as k=0
  #kvec_eff_zero = jnp.zeros_like(kvec_zero)      # treat seed as k=0
  # Total electrons
  print(Ne)
  # All magnetic-cell translations R = r a1 + s a2
  translations, xy_pairs = targetmom.find_all_translations_in_supercell(
      lattice, unit_cell_vectors
  )  # translations: (N_T,2), xy_pairs: (N_T,2) with integers (r,s)

  num_translations = translations.shape[0]
  r_int = xy_pairs[:, 0].astype(jnp.int32)
  s_int = xy_pairs[:, 1].astype(jnp.int32)

  # If your seed eigenvalue includes a pure (-1)^{r+s} or (-1)^{rs} etc,
  # you can put that here as an R-dependent phase. For now I leave it off,
  # assuming your current implementation already captures it in the
  # underlying ansatz; if not, you can uncomment / adjust this:
  #
  # pi_rs = jnp.pi * (r_int + s_int).astype(jnp.float32)   # for (-1)^{r+s}
  # or
  pi_rs = jnp.pi * (r_int + s_int).astype(jnp.float32)   # for (-1)^{rs}

  # ---------------------------------------------------------------------------
  # 1) SEED PROJECTOR: k = (0,0) Bloch projection
  # ---------------------------------------------------------------------------

  def compute_seed_phases(pos_xy: jnp.ndarray) -> jnp.ndarray:
    """Phases[R] for the k=(0,0) COM-projected seed, evaluated at pos_xy.

    pos_xy: (Ne,2) positions at which we want Ψ_0(pos_xy).
    """

    # r: (1,Ne,2), R: (N_T,1,2)
    r = pos_xy[None, :, :]          # (1,Ne,2)
    R = translations[:, None, :]    # (N_T,1,2)

    # r_i × R = r_x R_y - r_y R_x   (your convention)
    cross = r[..., 0] * R[..., 1] - r[..., 1] * R[..., 0]   # (N_T,Ne)
    sum_cross = jnp.sum(cross, axis=1)                      # (N_T,)

    # COM magnetic phase: (1/2) Σ_i r_i×R (ℓ_B=1 units)
    mag_phase = 0.5 * sum_cross                             # (N_T,)

    # Bloch part for seed (k=(0,0)): -k_eff_zero·R = 0
    bloch_phase = -jnp.dot(translations, kvec_eff_zero)     # (N_T,) == 0

    # If you want explicit (-1)^{r+s} or (-1)^{rs}, add pi_rs here:
    # total_phase = bloch_phase + mag_phase + pi_rs
    total_phase = bloch_phase + mag_phase + pi_rs

    return total_phase  # (N_T,)

  def seed_projector(
      params,
      pos_full: jnp.ndarray,
      spins: jnp.ndarray,
      atoms: jnp.ndarray,
      charges: jnp.ndarray,
      *,
      original_shape: str,
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Evaluate the k=(0,0) COM-projected seed Ψ_0 at positions pos_full.

    This is the same translation sum as in your original momentum projector,
    but with k fixed to (0,0). We treat pos_full as the 'x' argument in
      Ψ_0(x) = Σ_R e^{i φ_seed(R;x)} Ψ_base(x+R).
    """

    Ne_local = sum(nspins)
    full_ndim = options.ndim

    pos_xy = pos_full[:, :2]  # (Ne,2)

    # Seed phases for this configuration x = pos_full
    phases_per_translation = compute_seed_phases(pos_xy)  # (N_T,)

    def translated_config(i):
      base = pos_full                   # (Ne,full_ndim)
      shift_xy = translations[i]        # (2,)
      new_xy = pos_xy + shift_xy[None, :]
      if full_ndim == 2:
        out = new_xy
      else:
        higher = base[:, 2:]
        out = jnp.concatenate([new_xy, higher], axis=1)
      return out

    translated_full = jax.vmap(translated_config)(
        jnp.arange(num_translations)
    )  # (N_T,Ne,full_ndim)

    def eval_translation(i):
      if original_shape == 'flat':
        t_pos = translated_full[i].reshape(Ne_local * full_ndim)
      else:
        t_pos = translated_full[i]

      orbitals = orbitals_apply(params, t_pos, spins, atoms, charges)
      phase_i, logdet_i = network_blocks.logdet_matmul(orbitals)
      phase_i = phase_i + phases_per_translation[i]
      return phase_i, logdet_i

    phases_i, logdets_i = jax.vmap(eval_translation)(
        jnp.arange(num_translations)
    )

    max_logdet = jnp.max(logdets_i)
    shifted_exps = jnp.exp(logdets_i - max_logdet + 1j * phases_i)
    total = jnp.sum(shifted_exps) / num_translations

    phase0 = jnp.angle(total)
    log0 = max_logdet + jnp.log(jnp.abs(total))

    return phase0, log0

  # ---------------------------------------------------------------------------
  # 2) COM shift δ_K = (ℓ_B^2 / N_e) z×K = (ℓ_B^2 / N_e) (-K_y, K_x)
  # ---------------------------------------------------------------------------

  delta_xy = (1.0 / float(Ne)) * jnp.array(
      [-Kvec[1], Kvec[0]], dtype=Kvec.dtype
  )  # (2,)

  # ---------------------------------------------------------------------------
  # Apply: |Ψ_K> = T_COM(δ_K) |Ψ_0>
  # ---------------------------------------------------------------------------

  def network_apply(
      params,
      pos: jnp.ndarray,
      spins: jnp.ndarray,
      atoms: jnp.ndarray,
      charges: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:

    assert options.states == 0, "COM-K version currently assumes states=0."

    full_ndim = options.ndim
    Ne_local = sum(nspins)

    # Interpret pos in the same way orbitals_apply expects.
    if pos.ndim == 1:
      if pos.shape[0] != Ne_local * full_ndim:
        raise ValueError(
            f"Flattened pos has length {pos.shape[0]}, expected {Ne_local * full_ndim}."
        )
      pos_full = pos.reshape(Ne_local, full_ndim)
      original_shape = 'flat'
    elif pos.ndim == 2:
      if pos.shape != (Ne_local, full_ndim):
        raise ValueError(
            f"pos has shape {pos.shape}, expected ({Ne_local}, {full_ndim})."
        )
      pos_full = pos
      original_shape = 'matrix'
    else:
      raise ValueError("pos must be 1D ((Ne*ndim,)) or 2D ((Ne,ndim)).")

    # Original xy coordinates r (for COM phase of δ_K)
    pos_xy = pos_full[:, :2]

    # COM phase from T_COM(δ_K): exp(i/2 Σ_i r_i×δ_K)
    #   r_i × δ_K = r_x δ_y - r_y δ_x
    cross_delta = pos_xy[:, 0] * delta_xy[1] - pos_xy[:, 1] * delta_xy[0]
    phi_delta = 0.5 * jnp.sum(cross_delta)    # scalar

    # Shift the configuration by δ_K in xy-plane: this is the argument of Ψ_0
    pos_xy_shift = pos_xy + delta_xy[None, :]
    if full_ndim == 2:
      pos_shift_full = pos_xy_shift
    else:
      higher = pos_full[:, 2:]
      pos_shift_full = jnp.concatenate([pos_xy_shift, higher], axis=1)

    # Now evaluate the *seed* |Ψ_0> at x = pos_shift_full
    phase0, log0 = seed_projector(
        params, pos_shift_full, spins, atoms, charges,
        original_shape=original_shape
    )

    # Finally multiply by the COM phase (order: T_COM(δ_K) after P_seed)
    phase_tot = phase0 + phi_delta
    log_tot = log0

    if 'state_scale' in params:
      log_tot = log_tot + params['state_scale']

    return phase_tot, log_tot

  return networks.Network(
      options=options,
      init=network_init,
      apply=network_apply,
      orbitals=orbitals_apply,
  )

def make_fermi_net_with_zero_projection_momind(
    nspins: Tuple[int, ...],
    charges: jnp.ndarray,
    *,
    ndim: int = 3,
    determinants: int = 16,
    states: int = 0,
    envelope: Optional[envelopes.Envelope] = None,
    feature_layer: Optional[networks.FeatureLayer] = None,
    jastrow: Union[str, jastrows.JastrowType] = jastrows.JastrowType.SIMPLE_EE,
    jastrow_kwargs: dict = {},
    complex_output: bool = False,
    bias_orbitals: bool = False,
    rescale_inputs: bool = False,
    # Psiformer-specific kwargs:
    num_layers: int,
    num_heads: int,
    heads_dim: int,
    mlp_hidden_dims: Tuple[int, ...],
    use_layer_norm: bool,
    pbc_lattice: jnp.ndarray,
    # vortexformer-specific:
    N_holo: int,
    # magnetic Bloch / momentum:
    momind: int,
    mom_kwargs: dict,
) -> networks.Network:
  """Vortexformer (Psiformer + NN-projected zeros) projected to a magnetic
  Bloch state using

    Ψ_k({r_i}) = Σ_a exp{
        i [ Σ_i ( k·(r_i - a)/2 + (z·(r_i×a))/(2ℓ_B^2) ) + π r s ]
      } Ψ({r_i + a + ℓ_B^2 z×k}),

  where a = r a1 + s a2 runs over magnetic unit-cell translations inside the
  supercell, and all translations are center-of-mass (COM) translations.
  """

  if envelope is None:
    raise ValueError(
        "make_fermi_net_with_zero_projection_magnetic_bloch expects an explicit "
        "PRE_DETERMINANT envelope."
    )

  if envelope.apply_type != envelopes.EnvelopeType.PRE_DETERMINANT:
    raise ValueError(
        "Envelope for make_fermi_net_with_zero_projection_magnetic_bloch must "
        "be PRE_DETERMINANT."
    )

  if feature_layer is None:
    natoms = charges.shape[0]
    feature_layer = networks.make_ferminet_features(
        natoms, nspins, ndim=ndim, rescale_inputs=rescale_inputs
    )

  if isinstance(jastrow, str):
    if jastrow.upper() == 'DEFAULT':
      jastrow = jastrows.JastrowType.SIMPLE_EE
    else:
      jastrow = jastrows.JastrowType[jastrow.upper()]

  options = PsiformerOptions(
      ndim=ndim,
      determinants=determinants,
      states=states,
      envelope=envelope,
      feature_layer=feature_layer,
      jastrow=jastrow,
      jastrow_kwargs=jastrow_kwargs,
      complex_output=complex_output,
      bias_orbitals=bias_orbitals,
      full_det=True,
      rescale_inputs=rescale_inputs,
      num_layers=num_layers,
      num_heads=num_heads,
      heads_dim=heads_dim,
      mlp_hidden_dims=mlp_hidden_dims,
      use_layer_norm=use_layer_norm,
      pbc_lattice=pbc_lattice,
  )  # pytype: disable=wrong-keyword-args

  psiformer_layers = make_psiformer_layers(nspins, charges.shape[0], options)

  orbitals_init, orbitals_apply = networks.make_orbitals_with_zero_projection(
      nspins=nspins,
      charges=charges,
      options=options,
      equivariant_layers=psiformer_layers,
      N_holo=N_holo,
  )

  def network_init(key: chex.PRNGKey) -> networks.ParamTree:
    return orbitals_init(key)

  # ---------------------------------------------------------------------------
  # Magnetic lattice / momentum data (now all via targetmom.*)
  # ---------------------------------------------------------------------------

  abs_lattice = mom_kwargs['abs_lattice']              # 2x2 Tmatrix of supercell
  unit_cell_vectors = mom_kwargs['unit_cell_vectors']  # (2,2): a1,a2 magnetic cell

  # Supercell lattice vectors T1,T2 (columns)
  lattice = targetmom.lattice_vecs(
      unit_cell_vectors[0], unit_cell_vectors[1], abs_lattice
  )  # shape (2,2)
  T1 = lattice[:, 0]
  T2 = lattice[:, 1]
  # Reciprocal of supercell, T_i·G_j = 2π δ_ij
  reciprocal = targetmom.reciprocal_vecs(
      unit_cell_vectors[0],
      unit_cell_vectors[1],
      abs_lattice,
  )  # shape (2,2), columns G1,G2

  # g1,g2 following your targetmom.g1g2 convention
  g1, g2, _, _, _ = targetmom.g1g2(
      abs_lattice, reciprocal[:, 0], reciprocal[:, 1]
  )

  # Integer k-labels in k-space (2D integer lattice inside MBZ)
  klabels = targetmom.kpoints(abs_lattice)   # (N_k, 2)
  klabel = klabels[momind]                  # chosen (m,n)

  # Physical k-vector: k = m g1 + n g2 (your targetmom.mn)
  kvec = targetmom.mn(klabel, g1, g2)       # shape (2,)

 
  # Magnetic cell vectors a1,a2 (1 flux quantum per cell)
  a1 = unit_cell_vectors[0]                 # (2,)
  a2 = unit_cell_vectors[1]                 # (2,)

  # Magnetic length ℓ_B from |a1×a2| = 2π ℓ_B^2
  area = jnp.abs(a1[0] * a2[1] - a1[1] * a2[0])
  ellB2 = area / (2.0 * jnp.pi)
  area_cell = jnp.abs(a1[0] * a2[1] - a1[1] * a2[0])   # |a1 x a2|
  area_super = jnp.abs(T1[0] * T2[1] - T1[1] * T2[0])  # |T1 x T2|

  area_cell_f = float(area_cell)
  area_super_f = float(area_super)
  Nphi = int(round(area_super_f / area_cell_f))
  theta_L1 = jnp.pi  # in (-π, π]
  theta_L2 = jnp.pi

  # δ1 = Nφ θ_{L1} / (2π),  δ2 = Nφ θ_{L2} / (2π)
  delta1 = Nphi * theta_L1 / (2.0 * np.pi)
  delta2 = Nphi * theta_L2 / (2.0 * np.pi)

  # Δk = δ1 g1 + δ2 g2
  kshift = delta1 * g1 + delta2 * g2          # (2,)
  kvec_eff = kvec + kshift   
  # Guiding-center shift ℓ_B^2 z×k = ℓ_B^2 (-k_y, k_x)
  shift_k = (ellB2 * jnp.array([-kvec[1], kvec[0]], dtype=kvec.dtype))  # (2,)

  # All translations a inside the supercell and their integer (r,s)
  translations, xy_pairs = targetmom.find_all_translations_in_supercell(
      lattice, unit_cell_vectors
  )  # translations: (N_T,2), xy_pairs: (N_T,2) with integers (r,s)

  num_translations = translations.shape[0]
  r_int = xy_pairs[:, 0].astype(jnp.int32)
  s_int = xy_pairs[:, 1].astype(jnp.int32)

  # π r s factor
  pi_rs = jnp.pi * (r_int * s_int).astype(jnp.float32)  # (N_T,)

  # ---------------------------------------------------------------------------
  # Helper: phase Σ_i [ k·(r_i - a)/2 + (z·(r_i×a))/(2ℓ_B^2) ] + π r s
  # ---------------------------------------------------------------------------

  def compute_phases_for_translations(pos_xy: jnp.ndarray) -> jnp.ndarray:
    """Return phases[a] for all translations a, given electron positions pos_xy.

    phases[a] = Σ_i [ k·(r_i - a)/2 + (z·(r_i×a))/(2ℓ_B^2) ] + π r s.
    pos_xy: (Ne,2)
    """
    # r: (1,Ne,2), a: (N_T,1,2)
    r = pos_xy[None, :, :]           # (1,Ne,2)
    a = translations[:, None, :]     # (N_T,1,2)
  
    # r_i - a
    r_minus_a = (r -  a)/pos_xy.shape[0]                # (N_T,Ne,2)
    #r_minus_a =  - a  
    print(pos_xy.shape[0])
    # k·(r_i - a) for each i and each a
    dot_term = jnp.sum(r_minus_a * kvec_eff[None, None, :], axis=-1)  # (N_T,Ne)

    # z·(r_i×a) = r_x a_y - r_y a_x
    cross = r[..., 0] * a[..., 1] - r[..., 1] * a[..., 0]         # (N_T,Ne)

    # per-electron contribution: [k·(r_i - a) + (cross/ℓ_B^2)] / 2
    per_e_phase = 0.5*(dot_term + cross)               # (N_T,Ne)

    # sum over electrons
    sum_over_e = jnp.sum(per_e_phase, axis=1)                    # (N_T,)

    # add π r s
    total_phase = sum_over_e + pos_xy.shape[0]*pi_rs                             # (N_T,)
    return total_phase

  # ---------------------------------------------------------------------------
  # Apply: magnetic Bloch vortexformer
  # ---------------------------------------------------------------------------

  def network_apply(
      params,
      pos: jnp.ndarray,
      spins: jnp.ndarray,
      atoms: jnp.ndarray,
      charges: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Evaluate the magnetic Bloch-projected vortexformer Ψ_k at positions `pos`."""

    assert options.states == 0, "Bloch-projected version currently assumes states=0."

    Ne = sum(nspins)
    full_ndim = options.ndim

    # Interpret pos in the same way orbitals_apply expects.
    if pos.ndim == 1:
      if pos.shape[0] != Ne * full_ndim:
        raise ValueError(
            f"Flattened pos has length {pos.shape[0]}, expected {Ne * full_ndim}."
        )
      pos_full = pos.reshape(Ne, full_ndim)   # (Ne, ndim)
      original_shape = 'flat'
    elif pos.ndim == 2:
      if pos.shape != (Ne, full_ndim):
        raise ValueError(
            f"pos has shape {pos.shape}, expected ({Ne}, {full_ndim})."
        )
      pos_full = pos                          # (Ne, ndim)
      original_shape = 'matrix'
    else:
      raise ValueError("pos must be 1D ((Ne*ndim,)) or 2D ((Ne,ndim)).")

    # xy coordinates (where magnetic structure lives)
    pos_xy = pos_full[:, :2]                  # (Ne,2)

    # Phases per translation for this configuration (φ_nk formula)
    phases_per_translation = compute_phases_for_translations(pos_xy)  # (N_T,)

    # For each translation a, build COM-shifted coordinates:
    # r_i → r_i + a + ℓ_B^2 z×k
    total_shifts_xy = translations + (shift_k[None, :])/pos_xy.shape[0]                # (N_T,2)

    def translated_config(i):
      base = pos_full                   # (Ne, full_ndim)
      shift_xy = total_shifts_xy[i]     # (2,)
      new_xy = pos_xy + shift_xy[None, :]  # (Ne,2)
      if full_ndim == 2:
        out = new_xy
      else:
        higher = base[:, 2:]           # (Ne, full_ndim-2)
        out = jnp.concatenate([new_xy, higher], axis=1)
      return out

    translated_full = jax.vmap(translated_config)(
        jnp.arange(num_translations)
    )  # (N_T,Ne,full_ndim)

    def eval_translation(i):
      if original_shape == 'flat':
        t_pos = translated_full[i].reshape(Ne * full_ndim)
      else:
        t_pos = translated_full[i]

      orbitals = orbitals_apply(params, t_pos, spins, atoms, charges)
      phase_i, logdet_i = network_blocks.logdet_matmul(orbitals)

      # Add Bloch phase from your φ_nk formula
      phase_i = phase_i + phases_per_translation[i]
      return phase_i, logdet_i

    phases_i, logdets_i = jax.vmap(eval_translation)(
        jnp.arange(num_translations)
    )  # each shape (N_T,)

    # Complex log-sum-exp over translations
    max_logdet = jnp.max(logdets_i)
    shifted_exps = jnp.exp(
        logdets_i - max_logdet + 1j * phases_i
    )
    total_determinant = jnp.sum(shifted_exps) / num_translations

    overall_log = max_logdet + jnp.log(jnp.abs(total_determinant))
    overall_phase = jnp.angle(total_determinant)

    if 'state_scale' in params:
      overall_log = overall_log + params['state_scale']

    return overall_phase, overall_log

  return networks.Network(
      options=options,
      init=network_init,
      apply=network_apply,
      orbitals=orbitals_apply,
  )

def make_fermi_net_with_zero_projection_COM_Kx(
    nspins: Tuple[int, ...],
    charges: jnp.ndarray,
    *,
    ndim: int = 3,
    determinants: int = 16,
    states: int = 0,
    envelope: Optional[envelopes.Envelope] = None,
    feature_layer: Optional[networks.FeatureLayer] = None,
    jastrow: Union[str, jastrows.JastrowType] = jastrows.JastrowType.SIMPLE_EE,
    jastrow_kwargs: dict = {},
    complex_output: bool = False,
    bias_orbitals: bool = False,
    rescale_inputs: bool = False,
    # Psiformer-specific kwargs:
    num_layers: int = 2,
    num_heads: int = 4,
    heads_dim: int = 64,
    mlp_hidden_dims: Tuple[int, ...] = (256,),
    use_layer_norm: bool = False,
    pbc_lattice: jnp.ndarray = None,
    # vortexformer-specific:
    N_holo: int = 0,
    # COM momentum K_x label:
    momind: int = 0,
    mom_kwargs: dict = None,
) -> networks.Network:
  """Vortexformer projected to a COM-momentum sector along L1.

  Uses COM translations by L1/N_phi, where

    T_COM(L1/N_phi)^{N_phi} = T_COM(L1),
    T_COM(L1) ψ = (-1)^{N_phi * N_e} ψ,

  and projects onto an eigenstate of T_COM(L1/N_phi) with eigenvalue

    λ_x(Kx) = exp[i (π N_e + 2π Kx / N_phi)].

  The projector is

    Ψ_{Kx}({r_i}) ∝ Σ_{m=0}^{N_phi-1}
        exp[-i m (π N_e + 2π Kx / N_phi)] T_COM(m L1/N_phi) Ψ({r_i}).
  """

  if envelope is None:
    raise ValueError(
        "make_fermi_net_with_zero_projection_COM_Kx expects an explicit "
        "PRE_DETERMINANT envelope."
    )
  if envelope.apply_type != envelopes.EnvelopeType.PRE_DETERMINANT:
    raise ValueError(
        "Envelope for make_fermi_net_with_zero_projection_COM_Kx must "
        "be PRE_DETERMINANT."
    )
  if mom_kwargs is None:
    raise ValueError("mom_kwargs (with abs_lattice, unit_cell_vectors) must be provided.")

  if feature_layer is None:
    natoms = charges.shape[0]
    feature_layer = networks.make_ferminet_features(
        natoms, nspins, ndim=ndim, rescale_inputs=rescale_inputs
    )

  if isinstance(jastrow, str):
    jastrow = (
        jastrows.JastrowType.SIMPLE_EE
        if jastrow.upper() == "DEFAULT"
        else jastrows.JastrowType[jastrow.upper()]
    )

  options = PsiformerOptions(
      ndim=ndim,
      determinants=determinants,
      states=states,
      envelope=envelope,
      feature_layer=feature_layer,
      jastrow=jastrow,
      jastrow_kwargs=jastrow_kwargs,
      complex_output=complex_output,
      bias_orbitals=bias_orbitals,
      full_det=True,
      rescale_inputs=rescale_inputs,
      num_layers=num_layers,
      num_heads=num_heads,
      heads_dim=heads_dim,
      mlp_hidden_dims=mlp_hidden_dims,
      use_layer_norm=use_layer_norm,
      pbc_lattice=pbc_lattice,
  )

  psiformer_layers = make_psiformer_layers(nspins, charges.shape[0], options)

  orbitals_init, orbitals_apply = networks.make_orbitals_with_zero_projection(
      nspins=nspins,
      charges=charges,
      options=options,
      equivariant_layers=psiformer_layers,
      N_holo=N_holo,
  )

  def network_init(key: chex.PRNGKey) -> networks.ParamTree:
    return orbitals_init(key)

  # ---------------------------------------------------------------------------
  # Magnetic geometry and COM translation data
  # ---------------------------------------------------------------------------

  abs_lattice = mom_kwargs["abs_lattice"]              # 2x2 supercell T-matrix
  unit_cell_vectors = mom_kwargs["unit_cell_vectors"]  # (2,2): magnetic a1,a2

  # Supercell lattice vectors T1,T2 (columns)
  lattice = targetmom.lattice_vecs(
      unit_cell_vectors[0], unit_cell_vectors[1], abs_lattice
  )  # shape (2,2)
  T1 = lattice[:, 0]
  T2 = lattice[:, 1]

  # Magnetic cell vectors a1,a2 (1 flux quantum per cell)
  a1 = unit_cell_vectors[0]  # (2,)
  a2 = unit_cell_vectors[1]  # (2,)

  # Areas
  area_cell = jnp.abs(a1[0] * a2[1] - a1[1] * a2[0])   # |a1 x a2|
  area_super = jnp.abs(T1[0] * T2[1] - T1[1] * T2[0])  # |T1 x T2|

  # Magnetic length ℓ_B^2 from |a1×a2| = 2π ℓ_B^2
  ellB2 = area_cell / (2.0 * jnp.pi)

  # Number of flux quanta N_φ = area_super / area_cell
  area_cell_f = float(area_cell)
  area_super_f = float(area_super)
  Nphi = int(round(area_super_f / area_cell_f))
  if Nphi <= 0:
    raise ValueError(f"Invalid Nphi={Nphi} from areas: cell={area_cell_f}, super={area_super_f}")

  Ne = sum(nspins)

  Kx = int(momind)
  if not (0 <= Kx < Nphi):
    raise ValueError(f"K_x index momind={momind} out of range for Nphi={Nphi}.")

  # Fundamental COM translation step along L1: b = L1 / Nphi
  L1 = T1
  step = L1 / float(Nphi)  # (2,)

  # Translation vectors a_m = m * step, m=0..Nphi-1
  m_vals = jnp.arange(Nphi, dtype=jnp.float32)        # (Nphi,)
  translations = m_vals[:, None] * step[None, :]      # (Nphi,2)

  # Phase for each power m of T_COM(L1/Nphi):
  # eigenvalue λ_x(Kx) = exp[i(π Ne + 2π Kx / Nphi)]
  # projector weight: exp[-i m (π Ne + 2π Kx / Nphi)]
  kx_phases = (
      -jnp.pi * Ne * m_vals
      - 2.0 * jnp.pi * Kx * m_vals / float(Nphi)
  )  # (Nphi,)

  num_translations = translations.shape[0]

  # ---------------------------------------------------------------------------
  # Helper: COM magnetic translation phase for all a_m
  #   φ_m(X) = (a_m × Σ_i r_i)/(2ℓ_B^2)  -  m (π Ne + 2π Kx / Nφ)
  # ---------------------------------------------------------------------------

  def compute_phases_for_translations(pos_xy: jnp.ndarray) -> jnp.ndarray:
    """Return phases[m] for COM translations a_m, given electron positions.

    pos_xy: (Ne,2) electron positions in supercell coordinates.
    a_m × R = a_x R_y - a_y R_x.
    """
    R_sum = jnp.sum(pos_xy, axis=0)  # (2,)
    cross = translations[:, 0] * R_sum[1] - translations[:, 1] * R_sum[0]  # (Nphi,)
    com_phase = -cross / (2.0 * ellB2)  # (Nphi,)
    total_phase = com_phase + kx_phases
    return total_phase  # (Nphi,)

  # ---------------------------------------------------------------------------
  # Apply: COM-momentum–projected vortexformer (K_x sector)
  # ---------------------------------------------------------------------------

  def network_apply(
      params,
      pos: jnp.ndarray,
      spins: jnp.ndarray,
      atoms: jnp.ndarray,
      charges: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:

    assert options.states == 0, "COM K_x–projected version currently assumes states=0."

    full_ndim = options.ndim
    Ne_local = sum(nspins)

    # Interpret pos: flat or (Ne,ndim)
    if pos.ndim == 1:
      if pos.shape[0] != Ne_local * full_ndim:
        raise ValueError(
            f"Flattened pos has length {pos.shape[0]}, "
            f"expected {Ne_local * full_ndim}."
        )
      pos_full = pos.reshape(Ne_local, full_ndim)
      original_shape = "flat"
    elif pos.ndim == 2:
      if pos.shape != (Ne_local, full_ndim):
        raise ValueError(
            f"pos has shape {pos.shape}, expected ({Ne_local}, {full_ndim})."
        )
      pos_full = pos
      original_shape = "matrix"
    else:
      raise ValueError("pos must be 1D ((Ne*ndim,)) or 2D ((Ne,ndim)).")

    pos_xy = pos_full[:, :2]  # (Ne,2)

    phases_per_translation = compute_phases_for_translations(pos_xy)  # (Nphi,)

    # r_i → r_i + a_m
    def translated_config(i):
      base = pos_full
      shift_xy = translations[i]
      new_xy = pos_xy + shift_xy[None, :]
      if full_ndim == 2:
        out = new_xy
      else:
        higher = base[:, 2:]
        out = jnp.concatenate([new_xy, higher], axis=1)
      return out

    translated_full = jax.vmap(translated_config)(
        jnp.arange(num_translations)
    )  # (Nphi,Ne,full_ndim)

    def eval_translation(i):
      if original_shape == "flat":
        t_pos = translated_full[i].reshape(Ne_local * full_ndim)
      else:
        t_pos = translated_full[i]

      orbitals = orbitals_apply(params, t_pos, spins, atoms, charges)
      phase_i, logdet_i = network_blocks.logdet_matmul(orbitals)
      phase_i = phase_i + phases_per_translation[i]
      return phase_i, logdet_i

    phases_i, logdets_i = jax.vmap(eval_translation)(
        jnp.arange(num_translations)
    )  # (Nphi,)

    max_logdet = jnp.max(logdets_i)
    shifted_exps = jnp.exp(logdets_i - max_logdet + 1j * phases_i)
    total = jnp.sum(shifted_exps) / num_translations

    overall_log = max_logdet + jnp.log(jnp.abs(total))
    overall_phase = jnp.angle(total)

    if "state_scale" in params:
      overall_log = overall_log + params["state_scale"]

    return overall_phase, overall_log

  return networks.Network(
      options=options,
      init=network_init,
      apply=network_apply,
      orbitals=orbitals_apply,
  )

# def make_fermi_net_with_zero_projection_COM_Kx(
#     nspins: Tuple[int, ...],
#     charges: jnp.ndarray,
#     *,
#     ndim: int = 3,
#     determinants: int = 16,
#     states: int = 0,
#     envelope: Optional[envelopes.Envelope] = None,
#     feature_layer: Optional[networks.FeatureLayer] = None,
#     jastrow: Union[str, jastrows.JastrowType] = jastrows.JastrowType.SIMPLE_EE,
#     jastrow_kwargs: dict = {},
#     complex_output: bool = False,
#     bias_orbitals: bool = False,
#     rescale_inputs: bool = False,
#     # Psiformer-specific kwargs:
#     num_layers: int = 2,
#     num_heads: int = 4,
#     heads_dim: int = 64,
#     mlp_hidden_dims: Tuple[int, ...] = (256,),
#     use_layer_norm: bool = False,
#     pbc_lattice: jnp.ndarray = None,
#     # vortexformer-specific:
#     N_holo: int = 0,
#     # COM momentum K_x label:
#     momind: int = 0,
#     mom_kwargs: dict = None,
# ) -> networks.Network:
#   """Vortexformer projected to a COM-momentum sector along L1.

#   Uses COM translations by L1/N_phi, where

#     T_COM(L1/N_phi)^{N_phi} = T_COM(L1),
#     T_COM(L1) ψ = (-1)^{N_phi * N_e} ψ,

#   and projects onto an eigenstate of T_COM(L1/N_phi) with eigenvalue

#     λ_x(Kx) = exp[i (π N_e + 2π Kx / N_phi)].

#   The projector is

#     Ψ_{Kx}({r_i}) ∝ Σ_{m=0}^{N_phi-1}
#         exp[-i m (π N_e + 2π Kx / N_phi)] T_COM(m L1/N_phi) Ψ({r_i}).
#   """

#   if envelope is None:
#     raise ValueError(
#         "make_fermi_net_with_zero_projection_COM_Kx expects an explicit "
#         "PRE_DETERMINANT envelope."
#     )
#   if envelope.apply_type != envelopes.EnvelopeType.PRE_DETERMINANT:
#     raise ValueError(
#         "Envelope for make_fermi_net_with_zero_projection_COM_Kx must "
#         "be PRE_DETERMINANT."
#     )
#   if mom_kwargs is None:
#     raise ValueError("mom_kwargs (with abs_lattice, unit_cell_vectors) must be provided.")

#   if feature_layer is None:
#     natoms = charges.shape[0]
#     feature_layer = networks.make_ferminet_features(
#         natoms, nspins, ndim=ndim, rescale_inputs=rescale_inputs
#     )

#   if isinstance(jastrow, str):
#     jastrow = (
#         jastrows.JastrowType.SIMPLE_EE
#         if jastrow.upper() == "DEFAULT"
#         else jastrows.JastrowType[jastrow.upper()]
#     )

#   options = PsiformerOptions(
#       ndim=ndim,
#       determinants=determinants,
#       states=states,
#       envelope=envelope,
#       feature_layer=feature_layer,
#       jastrow=jastrow,
#       jastrow_kwargs=jastrow_kwargs,
#       complex_output=complex_output,
#       bias_orbitals=bias_orbitals,
#       full_det=True,
#       rescale_inputs=rescale_inputs,
#       num_layers=num_layers,
#       num_heads=num_heads,
#       heads_dim=heads_dim,
#       mlp_hidden_dims=mlp_hidden_dims,
#       use_layer_norm=use_layer_norm,
#       pbc_lattice=pbc_lattice,
#   )

#   psiformer_layers = make_psiformer_layers(nspins, charges.shape[0], options)

#   orbitals_init, orbitals_apply = networks.make_orbitals_with_zero_projection(
#       nspins=nspins,
#       charges=charges,
#       options=options,
#       equivariant_layers=psiformer_layers,
#       N_holo=N_holo,
#   )

#   def network_init(key: chex.PRNGKey) -> networks.ParamTree:
#     return orbitals_init(key)

#   # ---------------------------------------------------------------------------
#   # Magnetic geometry and COM translation data (outside apply)
#   # ---------------------------------------------------------------------------

#   abs_lattice = jnp.asarray(mom_kwargs["abs_lattice"], dtype=jnp.float32)              # (2,2)
#   unit_cell_vectors = jnp.asarray(mom_kwargs["unit_cell_vectors"], dtype=jnp.float32)  # (2,2)

#   # Supercell lattice vectors T1,T2 (columns)
#   lattice = targetmom.lattice_vecs(
#       unit_cell_vectors[0], unit_cell_vectors[1], abs_lattice
#   )  # (2,2)
#   lattice = jnp.asarray(lattice, dtype=jnp.float32)
#   T1 = lattice[:, 0]
#   T2 = lattice[:, 1]

#   # Magnetic cell vectors a1,a2 (1 flux quantum per cell)
#   a1 = unit_cell_vectors[0]
#   a2 = unit_cell_vectors[1]

#   # Areas
#   area_cell = jnp.abs(a1[0] * a2[1] - a1[1] * a2[0])   # |a1 x a2|
#   area_super = jnp.abs(T1[0] * T2[1] - T1[1] * T2[0])  # |T1 x T2|

#   # Magnetic length ℓ_B^2 from |a1×a2| = 2π ℓ_B^2
#   ellB2 = area_cell / (2.0 * jnp.pi)

#   # Number of flux quanta N_φ = area_super / area_cell (Python int, static)
#   area_cell_f = float(jax.device_get(area_cell))
#   area_super_f = float(jax.device_get(area_super))
#   Nphi = int(round(area_super_f / area_cell_f))
#   if Nphi <= 0:
#     raise ValueError(f"Invalid Nphi={Nphi} from areas: cell={area_cell_f}, super={area_super_f}")

#   Ne = sum(nspins)
#   Kx = int(momind)
#   if not (0 <= Kx < Nphi):
#     raise ValueError(f"K_x index momind={momind} out of range for Nphi={Nphi}.")

#   # Fundamental COM translation step along L1: b = L1 / Nphi
#   L1 = T1
#   step = L1 / jnp.asarray(float(Nphi), dtype=jnp.float32)  # (2,)

#   # Translation vectors a_m = m * step, m=0..Nphi-1  (all JAX arrays)
#   m_vals = jnp.arange(Nphi, dtype=jnp.float32)        # (Nphi,)
#   translations = m_vals[:, None] * step[None, :]      # (Nphi,2)

#   # Phase for each power m of T_COM(L1/Nphi):
#   #   eigenvalue λ_x(Kx) = exp[i(π Ne + 2π Kx / Nphi)]
#   #   projector weight: exp[-i m (π Ne + 2π Kx / Nphi)]
#   Ne_f = jnp.asarray(Ne, dtype=jnp.float32)
#   Kx_f = jnp.asarray(Kx, dtype=jnp.float32)
#   Nphi_f = jnp.asarray(float(Nphi), dtype=jnp.float32)

#   projector_phases = (
#       -jnp.pi * Ne_f * m_vals
#       - 2.0 * jnp.pi * Kx_f * m_vals / Nphi_f
#   )  # (Nphi,)

#   # ---------------------------------------------------------------------------
#   # Helper: COM magnetic translation phase for all a_m
#   #   φ_m^COM(X) = (a_m × Σ_i r_i)/(2ℓ_B^2)
#   # ---------------------------------------------------------------------------

#   def compute_com_phases(pos_xy: jnp.ndarray) -> jnp.ndarray:
#     """Return com_phase[m] = (a_m × Σ_i r_i)/(2ℓ_B^2) for all a_m."""
#     R_sum = jnp.sum(pos_xy, axis=0)  # (2,)
#     cross = translations[:, 0] * R_sum[1] - translations[:, 1] * R_sum[0]  # (Nphi,)
#     com_phase = -cross / (2.0 * ellB2)  # (Nphi,)
#     return com_phase  # (Nphi,)

#   # ---------------------------------------------------------------------------
#   # Apply: COM-momentum–projected vortexformer (K_x sector)
#   # ---------------------------------------------------------------------------

#   def network_apply(
#       params,
#       pos: jnp.ndarray,
#       spins: jnp.ndarray,
#       atoms: jnp.ndarray,
#       charges: jnp.ndarray,
#   ) -> Tuple[jnp.ndarray, jnp.ndarray]:
#     """Evaluate the COM K_x–projected vortexformer Ψ_{Kx} at positions `pos`."""

#     assert options.states == 0, "COM K_x–projected version currently assumes states=0."

#     full_ndim = options.ndim
#     Ne_local = sum(nspins)

#     # Interpret pos: flat or (Ne,ndim)
#     if pos.ndim == 1:
#       if pos.shape[0] != Ne_local * full_ndim:
#         raise ValueError(
#             f"Flattened pos has length {pos.shape[0]}, "
#             f"expected {Ne_local * full_ndim}."
#         )
#       pos_full = pos.reshape(Ne_local, full_ndim)
#       original_shape = "flat"
#     elif pos.ndim == 2:
#       if pos.shape != (Ne_local, full_ndim):
#         raise ValueError(
#             f"pos has shape {pos.shape}, expected ({Ne_local}, {full_ndim})."
#         )
#       pos_full = pos
#       original_shape = "matrix"
#     else:
#       raise ValueError("pos must be 1D ((Ne*ndim,)) or 2D ((Ne,ndim)).")

#     pos_full = jnp.asarray(pos_full)
#     pos_xy = pos_full[:, :2]  # (Ne,2)

#     # Per-translation total phase: COM phase + projector phase
#     com_phases = compute_com_phases(pos_xy)          # (Nphi,)
#     total_phases = com_phases + projector_phases     # (Nphi,)

#     float_dtype = pos_full.dtype
#     complex_dtype = jnp.complex64 if float_dtype == jnp.float32 else jnp.complex128

#     # Streaming complex log-sum-exp
#     max_logdet = jnp.array(-jnp.inf, dtype=float_dtype)
#     sum_norm = jnp.zeros((), dtype=complex_dtype)

#     # Unrolled loop over translations (folx-friendly)
#     for m in range(Nphi):
#       shift_xy = translations[m]      # (2,)
#       phase_i = total_phases[m]       # scalar

#       # Shift COM by shift_xy
#       new_xy = pos_xy + shift_xy[None, :]  # (Ne,2)
#       if full_ndim == 2:
#         t_pos_full = new_xy
#       else:
#         higher = pos_full[:, 2:]  # (Ne, full_ndim-2)
#         t_pos_full = jnp.concatenate([new_xy, higher], axis=-1)

#       if original_shape == "flat":
#         t_pos = t_pos_full.reshape(Ne_local * full_ndim)
#       else:
#         t_pos = t_pos_full

#       # Evaluate orbitals and determinant
#       orbitals = orbitals_apply(params, t_pos, spins, atoms, charges)
#       phase_det, logdet = network_blocks.logdet_matmul(orbitals)
#       total_phase = phase_det + phase_i  # scalar

#       # Streaming complex log-sum-exp update
#       cond = logdet > max_logdet

#       term1 = sum_norm + jnp.exp(logdet - max_logdet + 1j * total_phase)
#       term2 = sum_norm * jnp.exp(max_logdet - logdet) + jnp.exp(1j * total_phase)

#       sum_norm = jnp.where(cond, term2, term1)
#       max_logdet = jnp.where(cond, logdet, max_logdet)

#     # Recover the total determinant:
#     total = sum_norm * jnp.exp(max_logdet)
#     total = total / Nphi_f

#     abs_total = jnp.abs(total)
#     eps = jnp.asarray(1e-38, dtype=float_dtype)
#     abs_total = jnp.where(abs_total == 0.0, eps, abs_total)

#     overall_log = jnp.log(abs_total)
#     overall_phase = jnp.angle(total)

#     if "state_scale" in params:
#       overall_log = overall_log + params["state_scale"]

#     return overall_phase, overall_log

#   return networks.Network(
#       options=options,
#       init=network_init,
#       apply=network_apply,
#       orbitals=orbitals_apply,
#   )
  
def make_fermi_net_with_zero_projection_COM_Ky(
    nspins: Tuple[int, ...],
    charges: jnp.ndarray,
    *,
    ndim: int = 3,
    determinants: int = 16,
    states: int = 0,
    envelope: Optional[envelopes.Envelope] = None,
    feature_layer: Optional[networks.FeatureLayer] = None,
    jastrow: Union[str, jastrows.JastrowType] = jastrows.JastrowType.SIMPLE_EE,
    jastrow_kwargs: dict = {},
    complex_output: bool = False,
    bias_orbitals: bool = False,
    rescale_inputs: bool = False,
    # Psiformer-specific kwargs:
    num_layers: int = 2,
    num_heads: int = 4,
    heads_dim: int = 64,
    mlp_hidden_dims: Tuple[int, ...] = (256,),
    use_layer_norm: bool = False,
    pbc_lattice: jnp.ndarray = None,
    # vortexformer-specific:
    N_holo: int = 0,
    # COM momentum K_y:
    momind: int = 0,
    mom_kwargs: dict = None,
) -> networks.Network:
  """Vortexformer projected to a COM-momentum sector along L2.

  Uses COM translations by L2/N_phi, where

    T_COM(L2/N_phi)^{N_phi} = T_COM(L2),
    T_COM(L2) ψ = (-1)^{N_phi * N_e} ψ,

  and projects onto an eigenstate of T_COM(L2/N_phi) with eigenvalue

    λ_y(Ky) = exp[i (π N_e + 2π Ky / N_phi)].

  The projector is

    Ψ_{Ky}({r_i}) ∝ Σ_{m=0}^{N_phi-1}
        exp[-i m (π N_e + 2π Ky / N_phi)] T_COM(m L2/N_phi) Ψ({r_i}).
  """

  if envelope is None:
    raise ValueError(
        "make_fermi_net_with_zero_projection_COM_Ky expects an explicit "
        "PRE_DETERMINANT envelope."
    )
  if envelope.apply_type != envelopes.EnvelopeType.PRE_DETERMINANT:
    raise ValueError(
        "Envelope for make_fermi_net_with_zero_projection_COM_Ky must "
        "be PRE_DETERMINANT."
    )
  if mom_kwargs is None:
    raise ValueError("mom_kwargs (with abs_lattice, unit_cell_vectors) must be provided.")

  if feature_layer is None:
    natoms = charges.shape[0]
    feature_layer = networks.make_ferminet_features(
        natoms, nspins, ndim=ndim, rescale_inputs=rescale_inputs
    )

  if isinstance(jastrow, str):
    jastrow = (
        jastrows.JastrowType.SIMPLE_EE
        if jastrow.upper() == "DEFAULT"
        else jastrows.JastrowType[jastrow.upper()]
    )

  options = PsiformerOptions(
      ndim=ndim,
      determinants=determinants,
      states=states,
      envelope=envelope,
      feature_layer=feature_layer,
      jastrow=jastrow,
      jastrow_kwargs=jastrow_kwargs,
      complex_output=complex_output,
      bias_orbitals=bias_orbitals,
      full_det=True,
      rescale_inputs=rescale_inputs,
      num_layers=num_layers,
      num_heads=num_heads,
      heads_dim=heads_dim,
      mlp_hidden_dims=mlp_hidden_dims,
      use_layer_norm=use_layer_norm,
      pbc_lattice=pbc_lattice,
  )

  psiformer_layers = make_psiformer_layers(nspins, charges.shape[0], options)

  orbitals_init, orbitals_apply = networks.make_orbitals_with_zero_projection(
      nspins=nspins,
      charges=charges,
      options=options,
      equivariant_layers=psiformer_layers,
      N_holo=N_holo,
  )

  def network_init(key: chex.PRNGKey) -> networks.ParamTree:
    return orbitals_init(key)

  # ---------------------------------------------------------------------------
  # Magnetic geometry and COM translation data
  # ---------------------------------------------------------------------------

  abs_lattice = mom_kwargs["abs_lattice"]              # 2x2 supercell T-matrix
  unit_cell_vectors = mom_kwargs["unit_cell_vectors"]  # (2,2): magnetic a1,a2

  lattice = targetmom.lattice_vecs(
      unit_cell_vectors[0], unit_cell_vectors[1], abs_lattice
  )  # shape (2,2)
  T1 = lattice[:, 0]
  T2 = lattice[:, 1]

  a1 = unit_cell_vectors[0]
  a2 = unit_cell_vectors[1]

  area_cell = jnp.abs(a1[0] * a2[1] - a1[1] * a2[0])
  area_super = jnp.abs(T1[0] * T2[1] - T1[1] * T2[0])

  ellB2 = area_cell / (2.0 * jnp.pi)

  area_cell_f = float(area_cell)
  area_super_f = float(area_super)
  Nphi = int(round(area_super_f / area_cell_f))
  if Nphi <= 0:
    raise ValueError(f"Invalid Nphi={Nphi} from areas: cell={area_cell_f}, super={area_super_f}")

  Ne = sum(nspins)

  Ky = int(momind)
  if not (0 <= Ky < Nphi):
    raise ValueError(f"K_y index momind={momind} out of range for Nphi={Nphi}.")

  # Fundamental COM translation step along L2: b = L2 / Nphi
  L2 = T2
  step = L2 / float(Nphi)

  m_vals = jnp.arange(Nphi, dtype=jnp.float32)
  translations = m_vals[:, None] * step[None, :]  # (Nphi,2)

  # Eigenvalue per step: exp[i(π Ne + 2π Ky / Nphi)]
  ky_phases = (
      -jnp.pi * Ne * m_vals
      - 2.0 * jnp.pi * Ky * m_vals / float(Nphi)
  )  # (Nphi,)

  num_translations = translations.shape[0]

  def compute_phases_for_translations(pos_xy: jnp.ndarray) -> jnp.ndarray:
    R_sum = jnp.sum(pos_xy, axis=0)  # (2,)
    cross = translations[:, 0] * R_sum[1] - translations[:, 1] * R_sum[0]
    com_phase = -cross / (2.0 * ellB2)
    total_phase = com_phase + ky_phases
    return total_phase

  def network_apply(
      params,
      pos: jnp.ndarray,
      spins: jnp.ndarray,
      atoms: jnp.ndarray,
      charges: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:

    assert options.states == 0, "COM K_y–projected version currently assumes states=0."

    full_ndim = options.ndim
    Ne_local = sum(nspins)

    if pos.ndim == 1:
      if pos.shape[0] != Ne_local * full_ndim:
        raise ValueError(
            f"Flattened pos has length {pos.shape[0]}, "
            f"expected {Ne_local * full_ndim}."
        )
      pos_full = pos.reshape(Ne_local, full_ndim)
      original_shape = "flat"
    elif pos.ndim == 2:
      if pos.shape != (Ne_local, full_ndim):
        raise ValueError(
            f"pos has shape {pos.shape}, expected ({Ne_local}, {full_ndim})."
        )
      pos_full = pos
      original_shape = "matrix"
    else:
      raise ValueError("pos must be 1D ((Ne*ndim,)) or 2D ((Ne,ndim)).")

    pos_xy = pos_full[:, :2]

    phases_per_translation = compute_phases_for_translations(pos_xy)

    def translated_config(i):
      base = pos_full
      shift_xy = translations[i]
      new_xy = pos_xy + shift_xy[None, :]
      if full_ndim == 2:
        out = new_xy
      else:
        higher = base[:, 2:]
        out = jnp.concatenate([new_xy, higher], axis=1)
      return out

    translated_full = jax.vmap(translated_config)(
        jnp.arange(num_translations)
    )

    def eval_translation(i):
      if original_shape == "flat":
        t_pos = translated_full[i].reshape(Ne_local * full_ndim)
      else:
        t_pos = translated_full[i]
      orbitals = orbitals_apply(params, t_pos, spins, atoms, charges)
      phase_i, logdet_i = network_blocks.logdet_matmul(orbitals)
      phase_i = phase_i + phases_per_translation[i]
      return phase_i, logdet_i

    phases_i, logdets_i = jax.vmap(eval_translation)(
        jnp.arange(num_translations)
    )

    max_logdet = jnp.max(logdets_i)
    shifted_exps = jnp.exp(logdets_i - max_logdet + 1j * phases_i)
    total = jnp.sum(shifted_exps) / num_translations

    overall_log = max_logdet + jnp.log(jnp.abs(total))
    overall_phase = jnp.angle(total)

    if "state_scale" in params:
      overall_log = overall_log + params["state_scale"]

    return overall_phase, overall_log

  return networks.Network(
      options=options,
      init=network_init,
      apply=network_apply,
      orbitals=orbitals_apply,
  )

# def _egcd_int(a: int, b: int):
#   """(g,x,y) with g=gcd(a,b)>=0 and x*a + y*b = g."""
#   a0, b0 = int(a), int(b)
#   x0, x1, y0, y1 = 1, 0, 0, 1
#   while b0 != 0:
#     q = a0 // b0
#     a0, b0 = b0, a0 - q * b0
#     x0, x1 = x1, x0 - q * x1
#     y0, y1 = y1, y0 - q * y1
#   if a0 < 0:
#     a0, x0, y0 = -a0, -x0, -y0
#   return int(a0), int(x0), int(y0)

# def column_hnf_2x2(A_in: np.ndarray):
#   """
#   Column Hermite normal form for full-rank 2x2 integer A:
#     H = A * V
#   with V unimodular (det=+1) and H in upper-triangular column-HNF:
#     H = [[h11, h12],
#          [  0, h22]],
#   with h11>0, h22>0, and 0 <= h12 < h11.

#   Returns (H, V) as int64 numpy arrays.
#   """
#   A = np.array(A_in, dtype=np.int64).copy()
#   if A.shape != (2, 2):
#     raise ValueError(f"A must be (2,2), got {A.shape}")
#   detA = int(round(np.linalg.det(A)))
#   if detA == 0:
#     raise ValueError(f"A must be full-rank (det≠0). Got:\n{A}")

#   a, b = int(A[0, 0]), int(A[0, 1])
#   c, d = int(A[1, 0]), int(A[1, 1])

#   # Use row2=(c,d) to build a unimodular V with:
#   #   v1 = ( d/g, -c/g )  -> row2·v1 = 0
#   #   v2 = ( s, t )       -> row2·v2 = g  (Bezout)
#   g, s, t = _egcd_int(c, d)  # s*c + t*d = g
#   if g == 0:
#     # Should not happen for full rank 2x2
#     raise ValueError("Unexpected gcd(c,d)=0 in HNF.")
#   v1x, v1y = d // g, -c // g
#   v2x, v2y = s, t

#   V = np.array([[v1x, v2x],
#                 [v1y, v2y]], dtype=np.int64)
#   # det(V)= (d/g)*t - (-c/g)*s = (d t + c s)/g = g/g = 1
#   H = (A @ V).astype(np.int64)

#   # Make diagonals positive
#   if H[0, 0] < 0:
#     V[:, 0] *= -1
#     H[:, 0] *= -1
#   if H[1, 1] < 0:
#     V[:, 1] *= -1
#     H[:, 1] *= -1

#   h11 = int(H[0, 0])
#   if h11 <= 0:
#     raise ValueError(f"Bad HNF: h11={h11}, H=\n{H}")

#   # Reduce h12 into [0, h11)
#   h12 = int(H[0, 1])
#   q = int(np.floor(h12 / h11))  # floor division (works for negatives)
#   if q != 0:
#     # col2 <- col2 - q col1  i.e. V[:,1] <- V[:,1] - q V[:,0]
#     V[:, 1] -= q * V[:, 0]
#     H[:, 1] -= q * H[:, 0]
#   # Now enforce 0 <= h12 < h11
#   h12 = int(H[0, 1])
#   h12_mod = h12 % h11
#   if h12_mod != h12:
#     # col2 <- col2 + t col1 with t=(h12_mod - h12)/h11
#     tfix = (h12_mod - h12) // h11
#     V[:, 1] += tfix * V[:, 0]
#     H[:, 1] += tfix * H[:, 0]

#   # Final checks
#   if int(H[1, 0]) != 0:
#     raise ValueError(f"HNF failed to zero H[1,0]:\nH=\n{H}\nA=\n{A}\nV=\n{V}")
#   h11 = int(H[0, 0])
#   h22 = int(H[1, 1])
#   if h11 <= 0 or h22 <= 0:
#     raise ValueError(f"Bad HNF diagonals: H=\n{H}")
#   if not (0 <= int(H[0, 1]) < h11):
#     raise ValueError(f"Bad HNF remainder: H=\n{H}")

#   return H, V


# # ---------------------------------------------------------------------------
# # Main network constructor: project using a1,a2 directly (no b1,b2 generators)
# # ---------------------------------------------------------------------------

# def make_fermi_net_with_zero_projection_COM_k1k2(
#     nspins: Tuple[int, ...],
#     charges: jnp.ndarray,
#     *,
#     ndim: int = 3,
#     determinants: int = 16,
#     states: int = 0,
#     envelope: Optional["envelopes.Envelope"] = None,
#     feature_layer: Optional["networks.FeatureLayer"] = None,
#     jastrow: Union[str, "jastrows.JastrowType"] = "DEFAULT",
#     jastrow_kwargs: dict = {},
#     complex_output: bool = False,
#     bias_orbitals: bool = False,
#     rescale_inputs: bool = False,
#     # Psiformer-specific kwargs:
#     num_layers: int = 2,
#     num_heads: int = 4,
#     heads_dim: int = 64,
#     mlp_hidden_dims: Tuple[int, ...] = (256,),
#     use_layer_norm: bool = False,
#     pbc_lattice: jnp.ndarray = None,
#     # vortexformer-specific:
#     N_holo: int = 0,
#     # momentum labels (in a1,a2 basis, modulo M):
#     momind1: int = 0,
#     momind2: int = 0,
#     mom_kwargs: dict = None,
# ) -> "networks.Network":
#   """
#   Project to a simultaneous eigenstate of COM translations generated by T_COM(a1), T_COM(a2),
#   where a1,a2 enclose ONE flux quantum (so T_COM(a1) and T_COM(a2) commute).

#   Torus vectors are:
#     L1 = m1 a1 + m2 a2
#     L2 = n1 a1 + n2 a2
#   with M = [[m1,m2],[n1,n2]] = abs_lattice.

#   We solve the constraints on eigenphases (alpha1,alpha2):
#     m1 alpha1 + m2 alpha2 = Theta_L1 (mod 2π)
#     n1 alpha1 + n2 alpha2 = Theta_L2 (mod 2π)

#   by choosing an integer momentum label k=(k1,k2) in Z^2 / (M Z^2), and setting:
#     alpha = M^{-1} ( Theta + 2π k ).

#   The projector sums over translations Δ(u,v)=u a1 + v a2 for (u,v) in Z^2/(M^T Z^2),
#   using weights:
#     exp[ iπ (Ne mod 2) (uv mod 2) ] * exp[-i (u alpha1 + v alpha2) ]
#   and the COM magnetic translation phase:
#     exp[ -i (Δ × Σ r_i) / (2 ℓ_B^2) ].
#   """
#   if envelope is None:
#     raise ValueError("Expected explicit PRE_DETERMINANT envelope.")
#   if envelope.apply_type != envelopes.EnvelopeType.PRE_DETERMINANT:
#     raise ValueError("Envelope must be PRE_DETERMINANT.")
#   if mom_kwargs is None:
#     raise ValueError("mom_kwargs (with abs_lattice, unit_cell_vectors) must be provided.")

#   if feature_layer is None:
#     natoms = charges.shape[0]
#     feature_layer = networks.make_ferminet_features(
#         natoms, nspins, ndim=ndim, rescale_inputs=rescale_inputs
#     )

#   if isinstance(jastrow, str):
#     jastrow = (
#         jastrows.JastrowType.SIMPLE_EE
#         if jastrow.upper() == "DEFAULT"
#         else jastrows.JastrowType[jastrow.upper()]
#     )

#   options = PsiformerOptions(
#       ndim=ndim,
#       determinants=determinants,
#       states=states,
#       envelope=envelope,
#       feature_layer=feature_layer,
#       jastrow=jastrow,
#       jastrow_kwargs=jastrow_kwargs,
#       complex_output=complex_output,
#       bias_orbitals=bias_orbitals,
#       full_det=True,
#       rescale_inputs=rescale_inputs,
#       num_layers=num_layers,
#       num_heads=num_heads,
#       heads_dim=heads_dim,
#       mlp_hidden_dims=mlp_hidden_dims,
#       use_layer_norm=use_layer_norm,
#       pbc_lattice=pbc_lattice,
#   )

#   psiformer_layers = make_psiformer_layers(nspins, charges.shape[0], options)

#   orbitals_init, orbitals_apply = networks.make_orbitals_with_zero_projection(
#       nspins=nspins,
#       charges=charges,
#       options=options,
#       equivariant_layers=psiformer_layers,
#       N_holo=N_holo,
#   )

#   def network_init(key: chex.PRNGKey) -> networks.ParamTree:
#     return orbitals_init(key)

#   # ---------------------------------------------------------------------------
#   # Geometry
#   # ---------------------------------------------------------------------------
#   abs_lattice = mom_kwargs["abs_lattice"]              # M (2x2) integer-ish
#   unit_cell_vectors = mom_kwargs["unit_cell_vectors"]  # (2,2): a1,a2

#   lattice = targetmom.lattice_vecs(unit_cell_vectors[0], unit_cell_vectors[1], abs_lattice)
#   T1 = lattice[:, 0]  # = L1
#   T2 = lattice[:, 1]  # = L2

#   a1 = unit_cell_vectors[0]
#   a2 = unit_cell_vectors[1]

#   area_cell = jnp.abs(a1[0] * a2[1] - a1[1] * a2[0])
#   area_super = jnp.abs(T1[0] * T2[1] - T1[1] * T2[0])
#   ellB2 = area_cell / (2.0 * jnp.pi)

#   Nphi = int(round(float(area_super) / float(area_cell)))
#   if Nphi <= 0:
#     raise ValueError(f"Invalid Nphi={Nphi}")

#   Ne = sum(nspins)

#   # ---------------------------------------------------------------------------
#   # Integer matrix M
#   # ---------------------------------------------------------------------------
#   M_int = np.array(np.rint(np.array(abs_lattice)).astype(np.int64))
#   if M_int.shape != (2, 2):
#     raise ValueError(f"abs_lattice must be (2,2), got {M_int.shape}")

#   detM = int(round(np.linalg.det(M_int)))
#   if abs(detM) != Nphi:
#     raise ValueError(f"|det(M)|={abs(detM)} != Nphi={Nphi}. Check M vs geometry.")

#   # ---------------------------------------------------------------------------
#   # Momentum labeling k in Z^2 / (M Z^2) using column-HNF of M
#   #   Hm = M * Vm = [[h11, h12],[0,h22]]
#   # reps: k1=0..h11-1, k2=0..h22-1  (total Nphi)
#   # ---------------------------------------------------------------------------
#   Hm_int, Vm_int = column_hnf_2x2(M_int)
#   hm11 = int(Hm_int[0, 0])
#   hm22 = int(Hm_int[1, 1])
#   print(hm11)
#   print(hm22)
#   if hm11 * hm22 != abs(detM):
#     raise ValueError("HNF momentum invariants inconsistent with det(M).")

#   k1 = int(momind1)
#   k2 = int(momind2)
#   if not (0 <= k1 < hm11):
#     raise ValueError(f"k1={k1} out of range hm11={hm11}")
#   if not (0 <= k2 < hm22):
#     raise ValueError(f"k2={k2} out of range hm22={hm22}")

#   # This kvec is a representative of Z^2 / (M Z^2)
#   kvec = np.array([k1, k2], dtype=np.int64)

#   # ---------------------------------------------------------------------------
#   # Boundary phases for L1 and L2 (COM)
#   # Using your rule: for single particle
#   #   t(L) ψ = (eta_{p,q})^{Nphi} ψ, eta_{p,q}=(-1)^{pq+p+q}
#   # For L1=(1,0) => eta=-1 -> phase=(-1)^Nphi.
#   # For COM -> raise to Ne: (-1)^{Nphi*Ne}.
#   # Same for L2=(0,1).
#   # Allow overrides from mom_kwargs.
#   # ---------------------------------------------------------------------------
#   default_Theta_L1 = jnp.pi * float((int(Nphi) * int(Ne)) & 1)
#   default_Theta_L2 = jnp.pi * float((int(Nphi) * int(Ne)) & 1)
#   Theta_L1 = jnp.asarray(mom_kwargs.get("ThetaL1_override",
#                     mom_kwargs.get("Theta1_override", default_Theta_L1)), dtype=jnp.float32)
#   Theta_L2 = jnp.asarray(mom_kwargs.get("ThetaL2_override",
#                     mom_kwargs.get("Theta2_override", default_Theta_L2)), dtype=jnp.float32)

#   # Solve M alpha = Theta + 2π k   (mod 2π). Pick the representative:
#   # alpha = M^{-1} (Theta + 2π k)
#   M_inv = np.linalg.inv(M_int.astype(np.float64))  # safe at init time
#   theta_vec = np.array([float(Theta_L1), float(Theta_L2)], dtype=np.float64)
#   alpha_vec = M_inv @ (theta_vec + 2.0 * np.pi * kvec.astype(np.float64))
#   alpha1 = jnp.asarray(alpha_vec[0], dtype=jnp.float32)
#   alpha2 = jnp.asarray(alpha_vec[1], dtype=jnp.float32)

#   # ---------------------------------------------------------------------------
#   # Enumerate translations (u,v) in Z^2 / (M^T Z^2) using column-HNF of A=M^T
#   #   Ha = A * Va = [[ha11, ha12],[0,ha22]]
#   # reps: u=0..ha11-1, v=0..ha22-1  (total Nphi)
#   # ---------------------------------------------------------------------------
#   A_int = M_int.T
#   Ha_int, Va_int = column_hnf_2x2(A_int)
#   ha11 = int(Ha_int[0, 0])
#   ha22 = int(Ha_int[1, 1])
#   if ha11 * ha22 != abs(detM):
#     raise ValueError("HNF translation invariants inconsistent with det(M).")

#   u = jnp.arange(ha11, dtype=jnp.int32)
#   v = jnp.arange(ha22, dtype=jnp.int32)
#   uu, vv = jnp.meshgrid(u, v, indexing="ij")  # (ha11,ha22)
#   uu_f = uu.reshape(-1).astype(jnp.float32)
#   vv_f = vv.reshape(-1).astype(jnp.float32)

#   # Real-space translation vectors Δ = u a1 + v a2  (NO new generators)
#   translations = uu_f[:, None] * a1[None, :] + vv_f[:, None] * a2[None, :]  # (Nphi,2)
#   num_translations = translations.shape[0]
#   if int(num_translations) != Nphi:
#     raise ValueError(f"Expected Nphi translations, got {num_translations} (Nphi={Nphi}).")

#   # Cocycle phase for your "single-shot" T_COM(Δ):
#   # exp[i π Ne u v] = (-1)^{Ne u v}
#   uv_par = ((uu.reshape(-1) * vv.reshape(-1)) & 1).astype(jnp.float32)
#   cocycle = jnp.pi * float(Ne & 1) * uv_par

#   # Fourier weights for eigenvalues of T_COM(a1), T_COM(a2):
#   # weight(u,v) = exp[-i(u alpha1 + v alpha2)] * exp[i π Ne u v]
#   weight_phases = cocycle - (uu_f * alpha1 + vv_f * alpha2)

#   # ---------------------------------------------------------------------------
#   # Position-dependent COM magnetic translation phase
#   # ---------------------------------------------------------------------------
#   def compute_phases_for_translations(pos_xy: jnp.ndarray) -> jnp.ndarray:
#     R_sum = jnp.sum(pos_xy, axis=0)  # (2,)
#     cross = translations[:, 0] * R_sum[1] - translations[:, 1] * R_sum[0]
#     com_phase = -cross / (2.0 * ellB2)
#     return com_phase + weight_phases

#   # ---------------------------------------------------------------------------
#   # Apply
#   # ---------------------------------------------------------------------------
#   def network_apply(params, pos, spins, atoms, charges):
#     assert options.states == 0, "This projector assumes states=0."
#     full_ndim = options.ndim
#     Ne_local = sum(nspins)

#     if pos.ndim == 1:
#       if pos.shape[0] != Ne_local * full_ndim:
#         raise ValueError("Bad flattened pos length.")
#       pos_full = pos.reshape(Ne_local, full_ndim)
#       original_shape = "flat"
#     elif pos.ndim == 2:
#       if pos.shape != (Ne_local, full_ndim):
#         raise ValueError("Bad pos shape.")
#       pos_full = pos
#       original_shape = "matrix"
#     else:
#       raise ValueError("pos must be 1D or 2D.")

#     pos_xy = pos_full[:, :2]
#     phases_per_translation = compute_phases_for_translations(pos_xy)

#     def translated_config(i):
#       shift_xy = translations[i]
#       new_xy = pos_xy + shift_xy[None, :]
#       if full_ndim == 2:
#         return new_xy
#       higher = pos_full[:, 2:]
#       return jnp.concatenate([new_xy, higher], axis=1)

#     translated_full = jax.vmap(translated_config)(jnp.arange(num_translations))

#     def eval_translation(i):
#       t_pos = (translated_full[i].reshape(Ne_local * full_ndim)
#                if original_shape == "flat" else translated_full[i])
#       orbitals = orbitals_apply(params, t_pos, spins, atoms, charges)
#       phase_i, logdet_i = network_blocks.logdet_matmul(orbitals)
#       phase_i = phase_i + phases_per_translation[i]
#       return phase_i, logdet_i

#     phases_i, logdets_i = jax.vmap(eval_translation)(jnp.arange(num_translations))

#     max_logdet = jnp.max(logdets_i)
#     shifted = jnp.exp(logdets_i - max_logdet + 1j * phases_i)
#     total = jnp.sum(shifted) / num_translations

#     overall_log = max_logdet + jnp.log(jnp.abs(total))
#     overall_phase = jnp.angle(total)

#     if "state_scale" in params:
#       overall_log = overall_log + params["state_scale"]

#     return overall_phase, overall_log

#   return networks.Network(
#       options=options,
#       init=network_init,
#       apply=network_apply,
#       orbitals=orbitals_apply,
#   )


# def make_fermi_net_with_zero_projection_COM_k1k2(
#     nspins: Tuple[int, ...],
#     charges: jnp.ndarray,
#     *,
#     ndim: int = 3,
#     determinants: int = 16,
#     states: int = 0,
#     envelope: Optional["envelopes.Envelope"] = None,
#     feature_layer: Optional["networks.FeatureLayer"] = None,
#     jastrow: Union[str, "jastrows.JastrowType"] = "DEFAULT",
#     jastrow_kwargs: dict = {},
#     complex_output: bool = False,
#     bias_orbitals: bool = False,
#     rescale_inputs: bool = False,
#     # Psiformer-specific kwargs:
#     num_layers: int = 2,
#     num_heads: int = 4,
#     heads_dim: int = 64,
#     mlp_hidden_dims: Tuple[int, ...] = (256,),
#     use_layer_norm: bool = False,
#     pbc_lattice: jnp.ndarray = None,
#     # vortexformer-specific:
#     N_holo: int = 0,
#     # momentum labels: integers (m,n) for k = (m+phi1) g1 + (n+phi2) g2
#     momind1: int = 0,
#     momind2: int = 0,
#     mom_kwargs: dict = None,
# ) -> "networks.Network":
#   """
#   Option A: include the FULL single-particle parity
#       eta_{u,v} = (-1)^(u v + u + v)
#   directly in the projector weights for COM.

#   Weight for each translation Δ = u a1 + v a2 is:
#       exp[-i k·Δ] * (eta_{u,v})^(Ne) * exp[-i (Δ×Σr)/(2ℓ_B^2)]

#   i.e. weight phase:
#       + π Ne (u v + u + v)  -  k·Δ   (mod 2π),
#   plus the configuration-dependent magnetic COM phase.
#   """

#   if envelope is None:
#     raise ValueError("Expected explicit PRE_DETERMINANT envelope.")
#   if envelope.apply_type != envelopes.EnvelopeType.PRE_DETERMINANT:
#     raise ValueError("Envelope must be PRE_DETERMINANT.")
#   if mom_kwargs is None:
#     raise ValueError("mom_kwargs (with abs_lattice, unit_cell_vectors) must be provided.")

#   if feature_layer is None:
#     natoms = charges.shape[0]
#     feature_layer = networks.make_ferminet_features(
#         natoms, nspins, ndim=ndim, rescale_inputs=rescale_inputs
#     )

#   if isinstance(jastrow, str):
#     jastrow = (
#         jastrows.JastrowType.SIMPLE_EE
#         if jastrow.upper() == "DEFAULT"
#         else jastrows.JastrowType[jastrow.upper()]
#     )

#   options = PsiformerOptions(
#       ndim=ndim,
#       determinants=determinants,
#       states=states,
#       envelope=envelope,
#       feature_layer=feature_layer,
#       jastrow=jastrow,
#       jastrow_kwargs=jastrow_kwargs,
#       complex_output=complex_output,
#       bias_orbitals=bias_orbitals,
#       full_det=True,
#       rescale_inputs=rescale_inputs,
#       num_layers=num_layers,
#       num_heads=num_heads,
#       heads_dim=heads_dim,
#       mlp_hidden_dims=mlp_hidden_dims,
#       use_layer_norm=use_layer_norm,
#       pbc_lattice=pbc_lattice,
#   )

#   psiformer_layers = make_psiformer_layers(nspins, charges.shape[0], options)

#   orbitals_init, orbitals_apply = networks.make_orbitals_with_zero_projection(
#       nspins=nspins,
#       charges=charges,
#       options=options,
#       equivariant_layers=psiformer_layers,
#       N_holo=N_holo,
#   )

#   def network_init(key: chex.PRNGKey) -> networks.ParamTree:
#     return orbitals_init(key)

#   # ---------------------------------------------------------------------------
#   # Geometry
#   # ---------------------------------------------------------------------------
#   abs_lattice = mom_kwargs["abs_lattice"]              # 2x2 integer-ish matrix M
#   unit_cell_vectors = mom_kwargs["unit_cell_vectors"]  # (2,2): a1,a2

#   lattice = targetmom.lattice_vecs(unit_cell_vectors[0], unit_cell_vectors[1], abs_lattice)
#   L1 = lattice[:, 0]
#   L2 = lattice[:, 1]

#   a1 = unit_cell_vectors[0]
#   a2 = unit_cell_vectors[1]

#   area_cell = jnp.abs(a1[0] * a2[1] - a1[1] * a2[0])   # |a1×a2|
#   area_super = jnp.abs(L1[0] * L2[1] - L1[1] * L2[0])  # |L1×L2|
#   ellB2 = area_cell / (2.0 * jnp.pi)

#   Nphi = int(round(float(area_super) / float(area_cell)))
#   if Nphi <= 0:
#     raise ValueError(f"Invalid Nphi={Nphi}")

#   Ne = sum(nspins)

#   # ---------------------------------------------------------------------------
#   # Reciprocal vectors: (g1,g2) for k = (m+phi1) g1 + (n+phi2) g2
#   # ---------------------------------------------------------------------------
#   A_cell_oriented = a1[0] * a2[1] - a1[1] * a2[0]
#   if float(A_cell_oriented) == 0.0:
#     raise ValueError("Degenerate unit cell: a1×a2 = 0.")

#   reciprocal = targetmom.reciprocal_vecs(
#       unit_cell_vectors[0], unit_cell_vectors[1], np.array([[1, 0], [0, 1]])
#   )
#   g1, g2, _, _, _ = targetmom.g1g2(
#       jnp.asarray(abs_lattice, dtype=jnp.float32),
#       reciprocal[:, 0],
#       reciprocal[:, 1],
#   )
#   g1 = jnp.asarray(g1, dtype=jnp.float32)
#   g2 = jnp.asarray(g2, dtype=jnp.float32)

#   # ---------------------------------------------------------------------------
#   # Boundary twists -> (phi1,phi2) offsets in k
#   # ---------------------------------------------------------------------------
#   default_Theta_L1 = jnp.pi * float((int(Nphi) * int(Ne)) & 1)
#   default_Theta_L2 = jnp.pi * float((int(Nphi) * int(Ne)) & 1)

#   Theta_L1 = jnp.asarray(mom_kwargs.get("ThetaL1_override", default_Theta_L1), dtype=jnp.float32)
#   Theta_L2 = jnp.asarray(mom_kwargs.get("ThetaL2_override", default_Theta_L2), dtype=jnp.float32)

#   phi1_default = jnp.mod(Theta_L1 / (2.0 * jnp.pi), 1.0).astype(jnp.float32)
#   phi2_default = jnp.mod(Theta_L2 / (2.0 * jnp.pi), 1.0).astype(jnp.float32)

#   phi1 = jnp.asarray(mom_kwargs.get("phi1_override", phi1_default), dtype=jnp.float32)
#   phi2 = jnp.asarray(mom_kwargs.get("phi2_override", phi2_default), dtype=jnp.float32)

#   # ---------------------------------------------------------------------------
#   # Reduce (momind1,momind2) to your chosen BZ representative using projector(...)
#   # (recommended to turn back on for non-diagonal M)
#   # ---------------------------------------------------------------------------
#   k_in = jnp.array([int(momind1), int(momind2)], dtype=jnp.int32)
#   if bool(mom_kwargs.get("use_projector", True)):
#     k_rep, _, _ = targetmom.projector(k_in, jnp.asarray(abs_lattice, dtype=jnp.int32))
#     k_rep = k_rep.astype(jnp.float32)
#   else:
#     k_rep = k_in.astype(jnp.float32)

#   # k vector in real 2D: k = (m+phi1) g1 + (n+phi2) g2
#   k_vec = targetmom.mn_with_flux(k_rep, g1, g2, phi1, phi2).astype(jnp.float32)  # (2,)

#   # ---------------------------------------------------------------------------
#   # Enumerate translations Δ = u a1 + v a2 in the supercell (Nphi of them)
#   # ---------------------------------------------------------------------------
#   M_int = np.array(np.rint(np.array(abs_lattice)).astype(np.int64))
#   max_entry = int(np.max(np.abs(M_int)))
#   searchlimit = int(mom_kwargs.get("searchlimit_override", max(10, 4 * max_entry + 2)))

#   translations, uv_pairs = targetmom.find_all_translations_in_supercell(
#       jnp.asarray(lattice, dtype=jnp.float32),
#       jnp.asarray(unit_cell_vectors, dtype=jnp.float32),
#       searchlimit=searchlimit,
#       tolerance=1e-3,
#   )

#   if translations.shape[0] != Nphi:
#     raise ValueError(
#         f"find_all_translations_in_supercell returned {translations.shape[0]} translations, "
#         f"expected Nphi={Nphi}. Increase searchlimit (currently {searchlimit})."
#     )

#   translations = jnp.asarray(translations, dtype=jnp.float32)  # (Nphi,2)
#   uv_pairs = jnp.asarray(uv_pairs, dtype=jnp.int32)            # (Nphi,2) = (u,v)

#   u_int = uv_pairs[:, 0]
#   v_int = uv_pairs[:, 1]

#   # ---------------------------------------------------------------------------
#   # Option A weights:
#   #   eta_{u,v} = (-1)^(u v + u + v)
#   #   (eta_{u,v})^Ne = exp[i π Ne (u v + u + v)]
#   #   Fourier: exp[-i k·Δ]
#   #
#   # IMPORTANT: reduce (uv+u+v) mod 2 so the phase is only 0 or π (mod 2π),
#   # and likewise only Ne mod 2 matters. This prevents huge phases and keeps
#   # exact parity logic.
#   # ---------------------------------------------------------------------------
#   parity_uv_u_v = jnp.mod(u_int * v_int + u_int + v_int, 2).astype(jnp.float32)  # 0 or 1
#   eta_phase = jnp.pi * float(Ne & 1) * parity_uv_u_v                             # 0 or π
#   kdotDelta = jnp.einsum("i,ni->n", k_vec, translations)                         # (Nphi,)
#   weight_phases =  - kdotDelta                                          # (Nphi,)

#   num_translations = translations.shape[0]

#   # ---------------------------------------------------------------------------
#   # Position-dependent magnetic COM phase: exp[-i (Δ × Σ r_i)/(2ℓ_B^2)]
#   # Total phase used in the sum: com_phase + weight_phases
#   # ---------------------------------------------------------------------------
#   def compute_phases_for_translations(pos_xy: jnp.ndarray) -> jnp.ndarray:
#     R_sum = jnp.sum(pos_xy, axis=0)  # (2,)
#     cross = translations[:, 0] * R_sum[1] - translations[:, 1] * R_sum[0]  # (Nphi,)
#     com_phase = -cross / (2.0 * ellB2)
#     return com_phase + weight_phases

#   # ---------------------------------------------------------------------------
#   # Apply
#   # ---------------------------------------------------------------------------
#   def network_apply(params, pos, spins, atoms, charges):
#     assert options.states == 0, "This projector assumes states=0."

#     full_ndim = options.ndim
#     Ne_local = sum(nspins)

#     # Interpret pos: flat or (Ne,ndim)
#     if pos.ndim == 1:
#       if pos.shape[0] != Ne_local * full_ndim:
#         raise ValueError("Bad flattened pos length.")
#       pos_full = pos.reshape(Ne_local, full_ndim)
#       original_shape = "flat"
#     elif pos.ndim == 2:
#       if pos.shape != (Ne_local, full_ndim):
#         raise ValueError("Bad pos shape.")
#       pos_full = pos
#       original_shape = "matrix"
#     else:
#       raise ValueError("pos must be 1D or 2D.")

#     pos_xy = pos_full[:, :2]
#     phases_per_translation = compute_phases_for_translations(pos_xy)  # (Nphi,)

#     # Build translated configs r_i -> r_i + Δ
#     def translated_config(i):
#       shift_xy = translations[i]
#       new_xy = pos_xy + shift_xy[None, :]
#       if full_ndim == 2:
#         return new_xy
#       higher = pos_full[:, 2:]
#       return jnp.concatenate([new_xy, higher], axis=1)

#     translated_full = jax.vmap(translated_config)(jnp.arange(num_translations))  # (Nphi,Ne,ndim)

#     # Evaluate all translations (vmap) and do log-sum-exp with complex phases
#     def eval_translation(i):
#       t_pos = (translated_full[i].reshape(Ne_local * full_ndim)
#                if original_shape == "flat" else translated_full[i])
#       orbitals = orbitals_apply(params, t_pos, spins, atoms, charges)
#       phase_i, logdet_i = network_blocks.logdet_matmul(orbitals)
#       phase_i = phase_i + phases_per_translation[i]
#       return phase_i, logdet_i

#     phases_i, logdets_i = jax.vmap(eval_translation)(jnp.arange(num_translations))

#     max_logdet = jnp.max(logdets_i)
#     shifted = jnp.exp(logdets_i - max_logdet + 1j * phases_i)
#     total = jnp.sum(shifted) / num_translations

#     overall_log = max_logdet + jnp.log(jnp.abs(total))
#     overall_phase = jnp.angle(total)

#     if "state_scale" in params:
#       overall_log = overall_log + params["state_scale"]

#     return overall_phase, overall_log

#   return networks.Network(
#       options=options,
#       init=network_init,
#       apply=network_apply,
#       orbitals=orbitals_apply,
#   )

def make_fermi_net_with_zero_projection_COM_k1k2(
    nspins: Tuple[int, ...],
    charges: jnp.ndarray,
    *,
    ndim: int = 3,
    determinants: int = 16,
    states: int = 0,
    envelope: Optional[envelopes.Envelope] = None,
    feature_layer: Optional[networks.FeatureLayer] = None,
    jastrow: Union[str, jastrows.JastrowType] = jastrows.JastrowType.SIMPLE_EE,
    jastrow_kwargs: dict = {},
    complex_output: bool = False,
    bias_orbitals: bool = False,
    rescale_inputs: bool = False,
    # Psiformer-specific kwargs:
    num_layers: int = 2,
    num_heads: int = 4,
    heads_dim: int = 64,
    mlp_hidden_dims: Tuple[int, ...] = (256,),
    use_layer_norm: bool = False,
    pbc_lattice: jnp.ndarray = None,
    # vortexformer-specific:
    N_holo: int = 0,
    # momentum labels: integer class k in Z^2 / (M Z^2)
    momind1: int = 0,
    momind2: int = 0,
    mom_kwargs: dict = None,
) -> "networks.Network":
  r"""
  Corrected "Option A" COM projector (your parity kept explicit).

  We want COM eigenstates under translations by R = u a1 + v a2:
      \tilde T_COM(R) Ψ = (η_{u,v})^{Ne} e^{ i (u α1 + v α2) } Ψ
  with
      η_{u,v} = (-1)^{uv + u + v}.

  The projector is
      Ψ_{α} ∝ Σ_{(u,v) in Z^2/(M^T Z^2)}
          (η_{u,v})^{Ne} e^{-i(u α1 + v α2)}
          e^{-i (R × Σ r_i)/(2 ℓ_B^2)}  Ψ({r_i + R})

  The phases α=(α1,α2) are NOT chosen by a naive (phi1,phi2) shift.
  They must satisfy the torus BC constraints along L1,L2:
      L1 = m1 a1 + m2 a2,  L2 = n1 a1 + n2 a2,  M=[[m1,m2],[n1,n2]]

  Your NN/sigma basis fixes
      \tilde T_COM(L1) Ψ = e^{i Θ_L1} Ψ,   \tilde T_COM(L2) Ψ = e^{i Θ_L2} Ψ.

  But also, by the Bloch form:
      \tilde T_COM(L1) gives (η_{m1,m2})^{Ne} e^{ i(m1 α1 + m2 α2) }
      \tilde T_COM(L2) gives (η_{n1,n2})^{Ne} e^{ i(n1 α1 + n2 α2) }

  Therefore α must solve
      M α = Θ' + 2π k   (mod 2π),
  where
      Θ'_L1 = Θ_L1 - Ne * arg(η_{m1,m2}),  Θ'_L2 = Θ_L2 - Ne * arg(η_{n1,n2}),
      k is an integer representative of Z^2/(M Z^2).

  This fixes the α used in the Fourier weights, and then the state WILL be an
  eigenstate of COM translations generated by a1,a2 (up to your η-parity).
  """

  if envelope is None:
    raise ValueError("Expected explicit PRE_DETERMINANT envelope.")
  if envelope.apply_type != envelopes.EnvelopeType.PRE_DETERMINANT:
    raise ValueError("Envelope must be PRE_DETERMINANT.")
  if mom_kwargs is None:
    raise ValueError("mom_kwargs (with abs_lattice, unit_cell_vectors) must be provided.")

  if feature_layer is None:
    natoms = charges.shape[0]
    feature_layer = networks.make_ferminet_features(
        natoms, nspins, ndim=ndim, rescale_inputs=rescale_inputs
    )

  if isinstance(jastrow, str):
    jastrow = (
        jastrows.JastrowType.SIMPLE_EE
        if jastrow.upper() == "DEFAULT"
        else jastrows.JastrowType[jastrow.upper()]
    )

  options = PsiformerOptions(
      ndim=ndim,
      determinants=determinants,
      states=states,
      envelope=envelope,
      feature_layer=feature_layer,
      jastrow=jastrow,
      jastrow_kwargs=jastrow_kwargs,
      complex_output=complex_output,
      bias_orbitals=bias_orbitals,
      full_det=True,
      rescale_inputs=rescale_inputs,
      num_layers=num_layers,
      num_heads=num_heads,
      heads_dim=heads_dim,
      mlp_hidden_dims=mlp_hidden_dims,
      use_layer_norm=use_layer_norm,
      pbc_lattice=pbc_lattice,
  )

  psiformer_layers = make_psiformer_layers(nspins, charges.shape[0], options)

  orbitals_init, orbitals_apply = networks.make_orbitals_with_zero_projection(
      nspins=nspins,
      charges=charges,
      options=options,
      equivariant_layers=psiformer_layers,
      N_holo=N_holo,
  )

  def network_init(key: chex.PRNGKey) -> networks.ParamTree:
    return orbitals_init(key)

  # ---------------------------------------------------------------------------
  # Geometry
  # ---------------------------------------------------------------------------
  abs_lattice = mom_kwargs["abs_lattice"]              # M (2x2)
  unit_cell_vectors = mom_kwargs["unit_cell_vectors"]  # (2,2): a1,a2

  lattice = targetmom.lattice_vecs(unit_cell_vectors[0], unit_cell_vectors[1], abs_lattice)
  L1 = lattice[:, 0]
  L2 = lattice[:, 1]

  a1 = unit_cell_vectors[0]
  a2 = unit_cell_vectors[1]

  area_cell = jnp.abs(a1[0] * a2[1] - a1[1] * a2[0])
  area_super = jnp.abs(L1[0] * L2[1] - L1[1] * L2[0])
  ellB2 = area_cell / (2.0 * jnp.pi)

  Nphi = int(round(float(area_super) / float(area_cell)))
  if Nphi <= 0:
    raise ValueError(f"Invalid Nphi={Nphi}")

  Ne = int(sum(nspins))
  Ne_par = Ne & 1

  # ---------------------------------------------------------------------------
  # Integer M and consistency
  # ---------------------------------------------------------------------------
  M_int = np.array(np.rint(np.array(abs_lattice)).astype(np.int64))
  if M_int.shape != (2, 2):
    raise ValueError(f"abs_lattice must be (2,2), got {M_int.shape}")

  detM = int(round(np.linalg.det(M_int)))
  if abs(detM) != Nphi:
    raise ValueError(f"|det(M)|={abs(detM)} != Nphi={Nphi}. Check M vs geometry.")

  m1, m2 = int(M_int[0, 0]), int(M_int[0, 1])
  n1, n2 = int(M_int[1, 0]), int(M_int[1, 1])

  # ---------------------------------------------------------------------------
  # Torus BC phases along L1,L2 (COM) from your NN/sigma convention
  # ---------------------------------------------------------------------------
  default_Theta_L1 = jnp.pi * float((int(Nphi) * int(Ne)) & 1)
  default_Theta_L2 = jnp.pi * float((int(Nphi) * int(Ne)) & 1)

  Theta_L1 = float(mom_kwargs.get("ThetaL1_override", default_Theta_L1))
  Theta_L2 = float(mom_kwargs.get("ThetaL2_override", default_Theta_L2))

  # ---------------------------------------------------------------------------
  # Option A parity on the a1,a2 lattice:
  #   η_{u,v}=(-1)^{uv+u+v}
  # so for L1=(m1,m2) in this basis:
  #   η_L1 = (-1)^{m1 m2 + m1 + m2}
  # and similarly for L2.
  # This contributes (η_L)^{Ne} = exp[i π (Ne mod2) * parity_L].
  # Therefore define Θ' = Θ - Ne*arg(η_L).
  # ---------------------------------------------------------------------------
  parity_L1 = (m1 * m2 + m1 + m2) & 1
  parity_L2 = (n1 * n2 + n1 + n2) & 1

  Theta1p = Theta_L1 - (np.pi * float(Ne_par) * float(parity_L1))
  Theta2p = Theta_L2 - (np.pi * float(Ne_par) * float(parity_L2))

  # ---------------------------------------------------------------------------
  # Momentum label k in Z^2/(M Z^2):
  # Use your projector(...) to reduce k_in modulo M (recommended).
  # ---------------------------------------------------------------------------
  k_in = jnp.array([int(momind1), int(momind2)], dtype=jnp.int32)
  if bool(mom_kwargs.get("use_projector", True)):
    k_rep, _, _ = targetmom.projector(k_in, jnp.asarray(abs_lattice, dtype=jnp.int32))
    k_rep_np = np.array(k_rep, dtype=np.int64)
  else:
    k_rep_np = np.array([int(momind1), int(momind2)], dtype=np.int64)

  # ---------------------------------------------------------------------------
  # Solve for α from the congruence:
  #   M α = Θ' + 2π k   (mod 2π)
  # Pick the representative:
  #   α = M^{-1} (Θ' + 2π k)
  # This is the IMPORTANT fix vs using (phi1,phi2) in k=(m+phi)g.
  # ---------------------------------------------------------------------------
  two_pi = 2.0 * np.pi
  rhs = np.array([Theta1p, Theta2p], dtype=np.float64) + two_pi * k_rep_np.astype(np.float64)
  alpha = np.linalg.inv(M_int.astype(np.float64)) @ rhs
  alpha1 = jnp.asarray(alpha[0], dtype=jnp.float32)
  alpha2 = jnp.asarray(alpha[1], dtype=jnp.float32)

  # ---------------------------------------------------------------------------
  # Enumerate translations Δ = u a1 + v a2 in the supercell (Nphi of them)
  # ---------------------------------------------------------------------------
  max_entry = int(np.max(np.abs(M_int)))
  searchlimit = int(mom_kwargs.get("searchlimit_override", max(10, 4 * max_entry + 2)))

  translations, uv_pairs = targetmom.find_all_translations_in_supercell(
      jnp.asarray(lattice, dtype=jnp.float32),
      jnp.asarray(unit_cell_vectors, dtype=jnp.float32),
      searchlimit=searchlimit,
      tolerance=1e-3,
  )

  if translations.shape[0] != Nphi:
    raise ValueError(
        f"find_all_translations_in_supercell returned {translations.shape[0]} translations, "
        f"expected Nphi={Nphi}. Increase searchlimit (currently {searchlimit})."
    )

  translations = jnp.asarray(translations, dtype=jnp.float32)  # (Nphi,2)
  uv_pairs = jnp.asarray(uv_pairs, dtype=jnp.int32)            # (Nphi,2)

  u_int = uv_pairs[:, 0]
  v_int = uv_pairs[:, 1]

  # ---------------------------------------------------------------------------
  # Option A weights:
  #   exp[-i(u α1 + v α2)] * (η_{u,v})^{Ne}
  #
  # IMPORTANT FIXES:
  #  (1) Use u α1 + v α2 (NOT k·Δ from g1g2+phi unless diagonal case).
  #  (2) Actually include η-phase (your current code computed it but then dropped it).
  #  (3) Reduce parity mod 2; only Ne mod 2 matters.
  # ---------------------------------------------------------------------------
  parity_uv = jnp.mod(u_int * v_int + u_int + v_int, 2).astype(jnp.float32)  # 0 or 1
  eta_phase = jnp.pi * float(Ne_par) * parity_uv                              # 0 or π

  u_f = u_int.astype(jnp.float32)
  v_f = v_int.astype(jnp.float32)
  fourier_phase = u_f * alpha1 + v_f * alpha2                                 # (Nphi,)

  weight_phases = eta_phase - fourier_phase                                   # (Nphi,)

  num_translations = translations.shape[0]

  # ---------------------------------------------------------------------------
  # Position-dependent magnetic COM phase: exp[-i (Δ × Σ r_i)/(2ℓ_B^2)]
  # Total phase used in the sum: com_phase + weight_phases
  # ---------------------------------------------------------------------------
  def compute_phases_for_translations(pos_xy: jnp.ndarray) -> jnp.ndarray:
    R_sum = jnp.sum(pos_xy, axis=0)  # (2,)
    cross = translations[:, 0] * R_sum[1] - translations[:, 1] * R_sum[0]  # (Nphi,)
    com_phase = -cross / (2.0 * ellB2)
    return com_phase + weight_phases

  # ---------------------------------------------------------------------------
  # Apply
  # ---------------------------------------------------------------------------
  def network_apply(params, pos, spins, atoms, charges):
    assert options.states == 0, "This projector assumes states=0."

    full_ndim = options.ndim
    Ne_local = sum(nspins)

    if pos.ndim == 1:
      if pos.shape[0] != Ne_local * full_ndim:
        raise ValueError("Bad flattened pos length.")
      pos_full = pos.reshape(Ne_local, full_ndim)
      original_shape = "flat"
    elif pos.ndim == 2:
      if pos.shape != (Ne_local, full_ndim):
        raise ValueError("Bad pos shape.")
      pos_full = pos
      original_shape = "matrix"
    else:
      raise ValueError("pos must be 1D or 2D.")

    pos_xy = pos_full[:, :2]
    phases_per_translation = compute_phases_for_translations(pos_xy)  # (Nphi,)

    def translated_config(i):
      shift_xy = translations[i]
      new_xy = pos_xy + shift_xy[None, :]
      if full_ndim == 2:
        return new_xy
      higher = pos_full[:, 2:]
      return jnp.concatenate([new_xy, higher], axis=1)

    translated_full = jax.vmap(translated_config)(jnp.arange(num_translations))  # (Nphi,Ne,ndim)

    def eval_translation(i):
      t_pos = (translated_full[i].reshape(Ne_local * full_ndim)
               if original_shape == "flat" else translated_full[i])
      orbitals = orbitals_apply(params, t_pos, spins, atoms, charges)
      phase_i, logdet_i = network_blocks.logdet_matmul(orbitals)
      phase_i = phase_i + phases_per_translation[i]
      return phase_i, logdet_i

    phases_i, logdets_i = jax.vmap(eval_translation)(jnp.arange(num_translations))

    max_logdet = jnp.max(logdets_i)
    shifted = jnp.exp(logdets_i - max_logdet + 1j * phases_i)
    total = jnp.sum(shifted) / num_translations

    overall_log = max_logdet + jnp.log(jnp.abs(total))
    overall_phase = jnp.angle(total)

    if "state_scale" in params:
      overall_log = overall_log + params["state_scale"]

    return overall_phase, overall_log

  return networks.Network(
      options=options,
      init=network_init,
      apply=network_apply,
      orbitals=orbitals_apply,
  )

def make_vortexformer(
    nspins: Tuple[int, ...],
    charges: jnp.ndarray,
    *,
    ndim: int = 3,
    determinants: int = 16,
    states: int = 0,
    envelope: Optional[envelopes.Envelope] = None,
    feature_layer: Optional[networks.FeatureLayer] = None,
    jastrow: Union[str, jastrows.JastrowType] = jastrows.JastrowType.SIMPLE_EE,
    jastrow_kwargs: dict = {},
    complex_output: bool = True,     # usually True for LLL/quasi-periodic phases
    bias_orbitals: bool = False,     # unused here
    rescale_inputs: bool = False,
    num_layers: int = 2,
    num_heads: int = 4,
    heads_dim: int = 64,
    mlp_hidden_dims: Tuple[int, ...] = (256,),
    use_layer_norm: bool = False,
    pbc_lattice: jnp.ndarray = None,
    N_holo: int = 0,
) -> networks.Network:
  """Envelope-only Psiformer: determinants built purely from envelope factors whose zeros are NN-projected."""
  if envelope is None:
    raise ValueError("Provide a PRE_DETERMINANT LLL-like envelope.")
  if envelope.apply_type != envelopes.EnvelopeType.PRE_DETERMINANT:
    raise ValueError("Envelope must be PRE_DETERMINANT.")

  if feature_layer is None:
    natoms = charges.shape[0]
    feature_layer = networks.make_ferminet_features(natoms, nspins, ndim=ndim, rescale_inputs=rescale_inputs)

  if isinstance(jastrow, str):
    jastrow = jastrows.JastrowType.SIMPLE_EE if jastrow.upper() == 'DEFAULT' else jastrows.JastrowType[jastrow.upper()]

  options = PsiformerOptions(
      ndim=ndim,
      determinants=determinants,
      states=states,
      envelope=envelope,
      feature_layer=feature_layer,
      jastrow=jastrow,
      jastrow_kwargs=jastrow_kwargs,
      complex_output=complex_output,
      bias_orbitals=bias_orbitals,
      full_det=True,
      rescale_inputs=rescale_inputs,
      num_layers=num_layers,
      num_heads=num_heads,
      heads_dim=heads_dim,
      mlp_hidden_dims=mlp_hidden_dims,
      use_layer_norm=use_layer_norm,
      pbc_lattice=pbc_lattice,
  )

  psiformer_layers = make_psiformer_layers(nspins, charges.shape[0], options)

  # new orbital builder
  orbitals_init, orbitals_apply = networks.make_orbitals_envelope_only_zero_projection(
      nspins=nspins, charges=charges, options=options,
      equivariant_layers=psiformer_layers, N_holo=N_holo,
  )

  def network_init(key: chex.PRNGKey) -> networks.ParamTree:
    return orbitals_init(key)

  def network_apply(params, pos, spins, atoms, charges):
    orbitals = orbitals_apply(params, pos, spins, atoms, charges)
    if options.states:
      batch_logdet_matmul = jax.vmap(network_blocks.logdet_matmul, in_axes=0)
      orbitals = [jnp.reshape(orb, (options.states, -1) + orb.shape[1:]) for orb in orbitals]
      result = batch_logdet_matmul(orbitals)
    else:
      result = network_blocks.logdet_matmul(orbitals)
    if 'state_scale' in params:
      result = result[0], result[1] + params['state_scale']
    return result

  return networks.Network(
      options=options, init=network_init, apply=network_apply, orbitals=orbitals_apply
  )

def make_vortexformer_projection_COM_Ky(
    nspins: Tuple[int, ...],
    charges: jnp.ndarray,
    *,
    ndim: int = 3,
    determinants: int = 16,
    states: int = 0,
    envelope: Optional[envelopes.Envelope] = None,
    feature_layer: Optional[networks.FeatureLayer] = None,
    jastrow: Union[str, jastrows.JastrowType] = jastrows.JastrowType.SIMPLE_EE,
    jastrow_kwargs: dict = {},
    complex_output: bool = True,     # usually True for LLL/quasi-periodic phases
    bias_orbitals: bool = False,
    rescale_inputs: bool = False,
    num_layers: int = 2,
    num_heads: int = 4,
    heads_dim: int = 64,
    mlp_hidden_dims: Tuple[int, ...] = (256,),
    use_layer_norm: bool = False,
    pbc_lattice: jnp.ndarray = None,
    N_holo: int = 0,
    # COM momentum label K_y and geometry:
    momind: int = 0,
    mom_kwargs: dict = None,
) -> networks.Network:
  """Vortexformer (envelope-only Psiformer) projected onto COM momentum K_y.

  We build
      Ψ_{K_y}({r_i}) = (1/N_φ) Σ_{m=0}^{N_φ-1} c_m T_COM(m L2/N_φ) Ψ({r_i}),
  with coefficients
      c_m ∝ exp[ -i m (π N_e + 2π K_y/N_φ) ].

  Then the COM step operator
      U_2 = T_COM(L2/N_φ)
  has eigenvalue
      U_2 Ψ_{K_y} = exp[i (π N_e + 2π K_y/N_φ)] Ψ_{K_y},
  and
      U_2^{N_φ} Ψ_{K_y} = T_COM(L2) Ψ_{K_y} = (-1)^{N_φ N_e} Ψ_{K_y}.
  """

  if envelope is None:
    raise ValueError("Provide a PRE_DETERMINANT LLL-like envelope.")
  if envelope.apply_type != envelopes.EnvelopeType.PRE_DETERMINANT:
    raise ValueError("Envelope must be PRE_DETERMINANT.")

  if mom_kwargs is None:
    raise ValueError("mom_kwargs (with abs_lattice, unit_cell_vectors) must be provided.")

  if feature_layer is None:
    natoms = charges.shape[0]
    feature_layer = networks.make_ferminet_features(
        natoms, nspins, ndim=ndim, rescale_inputs=rescale_inputs
    )

  if isinstance(jastrow, str):
    jastrow = (
        jastrows.JastrowType.SIMPLE_EE
        if jastrow.upper() == "DEFAULT"
        else jastrows.JastrowType[jastrow.upper()]
    )

  options = PsiformerOptions(
      ndim=ndim,
      determinants=determinants,
      states=states,
      envelope=envelope,
      feature_layer=feature_layer,
      jastrow=jastrow,
      jastrow_kwargs=jastrow_kwargs,
      complex_output=complex_output,
      bias_orbitals=bias_orbitals,
      full_det=True,
      rescale_inputs=rescale_inputs,
      num_layers=num_layers,
      num_heads=num_heads,
      heads_dim=heads_dim,
      mlp_hidden_dims=mlp_hidden_dims,
      use_layer_norm=use_layer_norm,
      pbc_lattice=pbc_lattice,
  )

  psiformer_layers = make_psiformer_layers(nspins, charges.shape[0], options)

  orbitals_init, orbitals_apply = networks.make_orbitals_envelope_only_zero_projection(
      nspins=nspins,
      charges=charges,
      options=options,
      equivariant_layers=psiformer_layers,
      N_holo=N_holo,
  )

  def network_init(key: chex.PRNGKey) -> networks.ParamTree:
    return orbitals_init(key)

  # ---------------------------------------------------------------------------
  # Magnetic geometry and COM-translation data
  # ---------------------------------------------------------------------------

  abs_lattice = mom_kwargs["abs_lattice"]              # 2x2 matrix defining the supercell
  unit_cell_vectors = mom_kwargs["unit_cell_vectors"]  # (2,2): magnetic a1,a2

  lattice = targetmom.lattice_vecs(
      unit_cell_vectors[0], unit_cell_vectors[1], abs_lattice
  )  # shape (2,2)
  T1 = lattice[:, 0]
  T2 = lattice[:, 1]

  a1 = unit_cell_vectors[0]
  a2 = unit_cell_vectors[1]

  area_cell = jnp.abs(a1[0] * a2[1] - a1[1] * a2[0])
  area_super = jnp.abs(T1[0] * T2[1] - T1[1] * T2[0])

  # ℓ_B^2 from |a1×a2| = 2π ℓ_B^2 (not used explicitly if you fix ℓ_B^2 = 1)
  ellB2 = area_cell / (2.0 * jnp.pi)

  area_cell_f = float(area_cell)
  area_super_f = float(area_super)
  Nphi = int(round(area_super_f / area_cell_f))
  if Nphi <= 0:
    raise ValueError(f"Invalid Nphi={Nphi} from areas: cell={area_cell_f}, super={area_super_f}")

  Ne = sum(nspins)

  Ky = int(momind)
  if not (0 <= Ky < Nphi):
    raise ValueError(f"K_y index momind={momind} out of range for Nphi={Nphi}.")

  # COM step along L2: b = L2 / Nphi
  L2 = T2
  step = L2 / float(Nphi)  # (2,)

  m_vals = jnp.arange(Nphi, dtype=jnp.float32)        # (Nphi,)
  translations = m_vals[:, None] * step[None, :]      # (Nphi,2)

  # Coefficient phases c_m:
  #   c_m ∝ exp[-i m (π N_e + 2π K_y/N_phi)].
  ky_phases = m_vals * (
      -jnp.pi * float(Ne) - 2.0 * jnp.pi * float(Ky) / float(Nphi)
  )  # (Nphi,)

  num_translations = translations.shape[0]

  # ---------------------------------------------------------------------------
  # Helper: COM phase for all translations, given positions
  # ---------------------------------------------------------------------------

  def compute_phases_for_translations(pos_xy: jnp.ndarray) -> jnp.ndarray:
    """Return phases[m] for all COM translations a_m, given electron positions.

    pos_xy: (Ne,2)
    a_m × R_sum = a_x R_y − a_y R_x.
    """
    R_sum = jnp.sum(pos_xy, axis=0)  # (2,)
    cross = translations[:, 0] * R_sum[1] - translations[:, 1] * R_sum[0]  # (Nphi,)

    # You had a minus sign here in earlier working code; keep that convention
    # (ℓ_B^2 effectively set to 1 in your units).
    com_phase = -cross / 2.0  # (Nphi,)

    # Total exponent in exp(i * phase) factor
    total_phase = com_phase + ky_phases
    return total_phase  # (Nphi,)

  # ---------------------------------------------------------------------------
  # Apply: COM K_y–projected vortexformer
  # ---------------------------------------------------------------------------

  def network_apply(
      params,
      pos: jnp.ndarray,
      spins: jnp.ndarray,
      atoms: jnp.ndarray,
      charges: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    if options.states:
      raise NotImplementedError("COM K_y projector currently assumes states=0.")

    Ne_local = sum(nspins)
    full_ndim = options.ndim

    if pos.ndim == 1:
      if pos.shape[0] != Ne_local * full_ndim:
        raise ValueError(
            f"Flattened pos has length {pos.shape[0]}, expected {Ne_local * full_ndim}."
        )
      pos_full = pos.reshape(Ne_local, full_ndim)
      original_shape = "flat"
    elif pos.ndim == 2:
      if pos.shape != (Ne_local, full_ndim):
        raise ValueError(
            f"pos has shape {pos.shape}, expected ({Ne_local}, {full_ndim})."
        )
      pos_full = pos
      original_shape = "matrix"
    else:
      raise ValueError("pos must be 1D ((Ne*ndim,)) or 2D ((Ne,ndim)).")

    pos_xy = pos_full[:, :2]  # (Ne,2)

    phases_per_translation = compute_phases_for_translations(pos_xy)  # (Nphi,)

    def translated_config(i):
      base = pos_full
      shift_xy = translations[i]           # (2,)
      new_xy = pos_xy + shift_xy[None, :]  # (Ne,2)
      if full_ndim == 2:
        out = new_xy
      else:
        higher = base[:, 2:]
        out = jnp.concatenate([new_xy, higher], axis=1)
      return out

    translated_full = jax.vmap(translated_config)(
        jnp.arange(num_translations)
    )  # (Nphi,Ne,full_ndim)

    def eval_translation(i):
      if original_shape == "flat":
        t_pos = translated_full[i].reshape(Ne_local * full_ndim)
      else:
        t_pos = translated_full[i]

      orbitals = orbitals_apply(params, t_pos, spins, atoms, charges)
      phase_i, logdet_i = network_blocks.logdet_matmul(orbitals)
      phase_i = phase_i + phases_per_translation[i]
      return phase_i, logdet_i

    phases_i, logdets_i = jax.vmap(eval_translation)(
        jnp.arange(num_translations)
    )

    max_logdet = jnp.max(logdets_i)
    shifted_exps = jnp.exp(logdets_i - max_logdet + 1j * phases_i)
    total = jnp.sum(shifted_exps) / num_translations

    overall_log = max_logdet + jnp.log(jnp.abs(total))
    overall_phase = jnp.angle(total)

    if "state_scale" in params:
      overall_log = overall_log + params["state_scale"]

    return overall_phase, overall_log

  return networks.Network(
      options=options,
      init=network_init,
      apply=network_apply,
      orbitals=orbitals_apply,
  )

# def make_vortexformer_projection_COM_Kx(
#     nspins: Tuple[int, ...],
#     charges: jnp.ndarray,
#     *,
#     ndim: int = 3,
#     determinants: int = 16,
#     states: int = 0,
#     envelope: Optional[envelopes.Envelope] = None,
#     feature_layer: Optional[networks.FeatureLayer] = None,
#     jastrow: Union[str, jastrows.JastrowType] = jastrows.JastrowType.SIMPLE_EE,
#     jastrow_kwargs: dict = {},
#     complex_output: bool = True,     # usually True for LLL/quasi-periodic phases
#     bias_orbitals: bool = False,
#     rescale_inputs: bool = False,
#     num_layers: int = 2,
#     num_heads: int = 4,
#     heads_dim: int = 64,
#     mlp_hidden_dims: Tuple[int, ...] = (256,),
#     use_layer_norm: bool = False,
#     pbc_lattice: jnp.ndarray = None,
#     N_holo: int = 0,
#     # COM momentum label K_x and geometry:
#     momind: int = 0,
#     mom_kwargs: dict = None,
# ) -> networks.Network:
#   """Vortexformer (envelope-only Psiformer) projected onto COM momentum K_x.

#   As above but using COM translations along L1:

#       Ψ_{K_x}({r_i}) = (1/N_φ) Σ_{m=0}^{N_φ-1} c_m T_COM(m L1/N_φ) Ψ({r_i}),
#       c_m ∝ exp[-i m (π N_e + 2π K_x/N_φ)].

#   Then
#       U_1 = T_COM(L1/N_φ)
#       U_1 Ψ_{K_x} = exp[i (π N_e + 2π K_x/N_φ)] Ψ_{K_x}.
#   """

#   if envelope is None:
#     raise ValueError("Provide a PRE_DETERMINANT LLL-like envelope.")
#   if envelope.apply_type != envelopes.EnvelopeType.PRE_DETERMINANT:
#     raise ValueError("Envelope must be PRE_DETERMINANT.")

#   if mom_kwargs is None:
#     raise ValueError("mom_kwargs (with abs_lattice, unit_cell_vectors) must be provided.")

#   if feature_layer is None:
#     natoms = charges.shape[0]
#     feature_layer = networks.make_ferminet_features(
#         natoms, nspins, ndim=ndim, rescale_inputs=rescale_inputs
#     )

#   if isinstance(jastrow, str):
#     jastrow = (
#         jastrows.JastrowType.SIMPLE_EE
#         if jastrow.upper() == "DEFAULT"
#         else jastrows.JastrowType[jastrow.upper()]
#     )

#   options = PsiformerOptions(
#       ndim=ndim,
#       determinants=determinants,
#       states=states,
#       envelope=envelope,
#       feature_layer=feature_layer,
#       jastrow=jastrow,
#       jastrow_kwargs=jastrow_kwargs,
#       complex_output=complex_output,
#       bias_orbitals=bias_orbitals,
#       full_det=True,
#       rescale_inputs=rescale_inputs,
#       num_layers=num_layers,
#       num_heads=num_heads,
#       heads_dim=heads_dim,
#       mlp_hidden_dims=mlp_hidden_dims,
#       use_layer_norm=use_layer_norm,
#       pbc_lattice=pbc_lattice,
#   )

#   psiformer_layers = make_psiformer_layers(nspins, charges.shape[0], options)

#   orbitals_init, orbitals_apply = networks.make_orbitals_envelope_only_zero_projection(
#       nspins=nspins,
#       charges=charges,
#       options=options,
#       equivariant_layers=psiformer_layers,
#       N_holo=N_holo,
#   )

#   def network_init(key: chex.PRNGKey) -> networks.ParamTree:
#     return orbitals_init(key)

#   # ---------------------------------------------------------------------------
#   # Magnetic geometry and COM-translation data
#   # ---------------------------------------------------------------------------

#   abs_lattice = mom_kwargs["abs_lattice"]              # 2x2 matrix defining the supercell
#   unit_cell_vectors = mom_kwargs["unit_cell_vectors"]  # (2,2): magnetic a1,a2

#   lattice = targetmom.lattice_vecs(
#       unit_cell_vectors[0], unit_cell_vectors[1], abs_lattice
#   )  # shape (2,2)
#   T1 = lattice[:, 0]
#   T2 = lattice[:, 1]

#   a1 = unit_cell_vectors[0]
#   a2 = unit_cell_vectors[1]

#   area_cell = jnp.abs(a1[0] * a2[1] - a1[1] * a2[0])
#   area_super = jnp.abs(T1[0] * T2[1] - T1[1] * T2[0])

#   ellB2 = area_cell / (2.0 * jnp.pi)

#   area_cell_f = float(area_cell)
#   area_super_f = float(area_super)
#   Nphi = int(round(area_super_f / area_cell_f))
#   if Nphi <= 0:
#     raise ValueError(f"Invalid Nphi={Nphi} from areas: cell={area_cell_f}, super={area_super_f}")

#   Ne = sum(nspins)

#   Kx = int(momind)
#   if not (0 <= Kx < Nphi):
#     raise ValueError(f"K_x index momind={momind} out of range for Nphi={Nphi}.")

#   # COM step along L1: b = L1 / Nphi
#   L1 = T1
#   step = L1 / float(Nphi)

#   m_vals = jnp.arange(Nphi, dtype=jnp.float32)
#   translations = m_vals[:, None] * step[None, :]      # (Nphi,2)

#   # Coefficient phases for Kx projector:
#   kx_phases = m_vals * (
#       -jnp.pi * float(Ne) - 2.0 * jnp.pi * float(Kx) / float(Nphi)
#   )  # (Nphi,)

#   num_translations = translations.shape[0]

#   def compute_phases_for_translations(pos_xy: jnp.ndarray) -> jnp.ndarray:
#     R_sum = jnp.sum(pos_xy, axis=0)  # (2,)
#     cross = translations[:, 0] * R_sum[1] - translations[:, 1] * R_sum[0]
#     com_phase = -cross / 2.0
#     total_phase = com_phase + kx_phases
#     return total_phase

#   def network_apply(
#       params,
#       pos: jnp.ndarray,
#       spins: jnp.ndarray,
#       atoms: jnp.ndarray,
#       charges: jnp.ndarray,
#   ) -> Tuple[jnp.ndarray, jnp.ndarray]:
#     if options.states:
#       raise NotImplementedError("COM K_x projector currently assumes states=0.")

#     Ne_local = sum(nspins)
#     full_ndim = options.ndim

#     if pos.ndim == 1:
#       if pos.shape[0] != Ne_local * full_ndim:
#         raise ValueError(
#             f"Flattened pos has length {pos.shape[0]}, expected {Ne_local * full_ndim}."
#         )
#       pos_full = pos.reshape(Ne_local, full_ndim)
#       original_shape = "flat"
#     elif pos.ndim == 2:
#       if pos.shape != (Ne_local, full_ndim):
#         raise ValueError(
#             f"pos has shape {pos.shape}, expected ({Ne_local}, {full_ndim})."
#         )
#       pos_full = pos
#       original_shape = "matrix"
#     else:
#       raise ValueError("pos must be 1D ((Ne*ndim,)) or 2D ((Ne,ndim)).")

#     pos_xy = pos_full[:, :2]

#     phases_per_translation = compute_phases_for_translations(pos_xy)

#     def translated_config(i):
#       base = pos_full
#       shift_xy = translations[i]
#       new_xy = pos_xy + shift_xy[None, :]
#       if full_ndim == 2:
#         out = new_xy
#       else:
#         higher = base[:, 2:]
#         out = jnp.concatenate([new_xy, higher], axis=1)
#       return out

#     translated_full = jax.vmap(translated_config)(
#         jnp.arange(num_translations)
#     )

#     def eval_translation(i):
#       if original_shape == "flat":
#         t_pos = translated_full[i].reshape(Ne_local * full_ndim)
#       else:
#         t_pos = translated_full[i]

#       orbitals = orbitals_apply(params, t_pos, spins, atoms, charges)
#       phase_i, logdet_i = network_blocks.logdet_matmul(orbitals)
#       phase_i = phase_i + phases_per_translation[i]
#       return phase_i, logdet_i

#     phases_i, logdets_i = jax.vmap(eval_translation)(
#         jnp.arange(num_translations)
#     )

#     max_logdet = jnp.max(logdets_i)
#     shifted_exps = jnp.exp(logdets_i - max_logdet + 1j * phases_i)
#     total = jnp.sum(shifted_exps) / num_translations

#     overall_log = max_logdet + jnp.log(jnp.abs(total))
#     overall_phase = jnp.angle(total)

#     if "state_scale" in params:
#       overall_log = overall_log + params["state_scale"]

#     return overall_phase, overall_log

#   return networks.Network(
#       options=options,
#       init=network_init,
#       apply=network_apply,
#       orbitals=orbitals_apply,
#   )

def make_fermi_net_momentum_projected_COM_Kx(
    nspins: Tuple[int, ...],
    charges: jnp.ndarray,
    *,
    ndim: int = 3,
    determinants: int = 16,
    states: int = 0,
    envelope: Optional[envelopes.Envelope] = None,
    feature_layer: Optional[networks.FeatureLayer] = None,
    jastrow: Union[str, jastrows.JastrowType] = jastrows.JastrowType.SIMPLE_EE,
    jastrow_kwargs: dict = {},
    complex_output: bool = False,
    bias_orbitals: bool = False,
    rescale_inputs: bool = False,
    # Psiformer-specific kwargs:
    num_layers: int = 2,
    num_heads: int = 4,
    heads_dim: int = 64,
    mlp_hidden_dims: Tuple[int, ...] = (256,),
    use_layer_norm: bool = False,
    pbc_lattice: jnp.ndarray = None,
    # COM momentum K_x label:
    momind: int = 0,
    mom_kwargs: dict = None,
) -> networks.Network:
  """Psiformer projected to a COM-momentum sector along L1 (magnetic torus).

  Uses COM translations by L1/N_phi, where

    T_COM(L1/N_phi)^{N_phi} = T_COM(L1),
    T_COM(L1) ψ = (-1)^{N_phi * N_e} ψ,

  and projects onto an eigenstate of T_COM(L1/N_phi) with eigenvalue

    λ_x(Kx) = exp[i (π N_e + 2π Kx / N_phi)].

  The projector is

    Ψ_{Kx}({r_i}) ∝ Σ_{m=0}^{N_phi-1}
        exp[-i m (π N_e + 2π Kx / N_phi)] T_COM(m L1/N_phi) Ψ({r_i}).

  Implementation details:
  - Same Psiformer + determinants structure as make_fermi_net_momentum_projected.
  - Magnetic COM translation logic and streaming log-sum-exp like
    make_fermi_net_with_zero_projection_COM_Kx.
  """

  # ---------------------------------------------------------------------------
  # Network scaffolding (same style as make_fermi_net_momentum_projected)
  # ---------------------------------------------------------------------------

  if envelope is None:
    envelope = envelopes.make_isotropic_envelope()

  if feature_layer is None:
    natoms = charges.shape[0]
    feature_layer = networks.make_ferminet_features(
        natoms, nspins, ndim=ndim, rescale_inputs=rescale_inputs
    )

  if isinstance(jastrow, str):
    if jastrow.upper() == "DEFAULT":
      jastrow = jastrows.JastrowType.SIMPLE_EE
    else:
      jastrow = jastrows.JastrowType[jastrow.upper()]

  options = PsiformerOptions(
      ndim=ndim,
      determinants=determinants,
      states=states,
      envelope=envelope,
      feature_layer=feature_layer,
      jastrow=jastrow,
      jastrow_kwargs=jastrow_kwargs,
      complex_output=complex_output,
      bias_orbitals=bias_orbitals,
      full_det=True,
      rescale_inputs=rescale_inputs,
      num_layers=num_layers,
      num_heads=num_heads,
      heads_dim=heads_dim,
      mlp_hidden_dims=mlp_hidden_dims,
      use_layer_norm=use_layer_norm,
      pbc_lattice=pbc_lattice,
  )

  psiformer_layers = make_psiformer_layers(nspins, charges.shape[0], options)

  # Standard Psiformer orbitals (no NN-zero projection here)
  orbitals_init, orbitals_apply = networks.make_orbitals(
      nspins=nspins,
      charges=charges,
      options=options,
      equivariant_layers=psiformer_layers,
  )

  def network_init(key: chex.PRNGKey) -> networks.ParamTree:
    return orbitals_init(key)

  if mom_kwargs is None:
    raise ValueError("mom_kwargs (with abs_lattice, unit_cell_vectors) must be provided.")

  # ---------------------------------------------------------------------------
  # Magnetic geometry + COM translation data (outside apply, JIT-static)
  # ---------------------------------------------------------------------------

  abs_lattice = jnp.asarray(mom_kwargs["abs_lattice"], dtype=jnp.float32)              # (2,2)
  unit_cell_vectors = jnp.asarray(mom_kwargs["unit_cell_vectors"], dtype=jnp.float32)  # (2,2)

  # Supercell lattice vectors T1,T2 (columns)
  lattice = targetmom.lattice_vecs(
      unit_cell_vectors[0], unit_cell_vectors[1], abs_lattice
  )  # (2,2) but may be numpy
  lattice = jnp.asarray(lattice, dtype=jnp.float32)
  T1 = lattice[:, 0]
  T2 = lattice[:, 1]

  # Magnetic cell vectors a1,a2 (1 flux quantum per cell)
  a1 = unit_cell_vectors[0]
  a2 = unit_cell_vectors[1]

  # Areas
  area_cell = jnp.abs(a1[0] * a2[1] - a1[1] * a2[0])   # |a1 x a2|
  area_super = jnp.abs(T1[0] * T2[1] - T1[1] * T2[0])  # |T1 x T2|

  # Magnetic length ℓ_B^2: |a1×a2| = 2π ℓ_B^2
  ellB2 = area_cell / (2.0 * jnp.pi)

  # Number of flux quanta N_φ = area_super / area_cell (Python int, static)
  area_cell_f = float(jax.device_get(area_cell))
  area_super_f = float(jax.device_get(area_super))
  Nphi = int(round(area_super_f / area_cell_f))
  if Nphi <= 0:
    raise ValueError(f"Invalid Nphi={Nphi} from areas: cell={area_cell_f}, super={area_super_f}")

  Ne = sum(nspins)
  Kx = int(momind)
  if not (0 <= Kx < Nphi):
    raise ValueError(f"K_x index momind={momind} out of range for Nphi={Nphi}.")

  # Fundamental COM translation step along L1: b = L1 / Nphi
  L1 = T1
  step = L1 / jnp.asarray(float(Nphi), dtype=jnp.float32)  # (2,)

  # COM translation vectors a_m = m * step, m = 0..Nphi-1
  m_vals = jnp.arange(Nphi, dtype=jnp.float32)        # (Nphi,)
  translations = m_vals[:, None] * step[None, :]      # (Nphi,2)

  # Projector phases for powers of T_COM(L1/Nphi):
  # eigenvalue λ_x(Kx) = exp[i(π Ne + 2π Kx / Nphi)]
  # projector weight exp[-i m (π Ne + 2π Kx / Nphi)]
  Ne_f = jnp.asarray(Ne, dtype=jnp.float32)
  Kx_f = jnp.asarray(Kx, dtype=jnp.float32)
  Nphi_f = jnp.asarray(float(Nphi), dtype=jnp.float32)

  projector_phases = (
      -jnp.pi * Ne_f * m_vals
      - 2.0 * jnp.pi * Kx_f * m_vals / Nphi_f
  )  # (Nphi,)

  # ---------------------------------------------------------------------------
  # Helper: COM magnetic-translation phase for all a_m
  #   φ_m^COM(X) = (a_m × Σ_i r_i)/(2ℓ_B^2)
  # ---------------------------------------------------------------------------

  def compute_com_phases(pos_xy: jnp.ndarray) -> jnp.ndarray:
    """Return com_phase[m] = (a_m × Σ_i r_i)/(2ℓ_B^2) for all a_m."""
    R_sum = jnp.sum(pos_xy, axis=0)  # (2,)
    cross = translations[:, 0] * R_sum[1] - translations[:, 1] * R_sum[0]  # (Nphi,)
    com_phase = -cross / (2.0 * ellB2)  # (Nphi,)
    return com_phase

  # ---------------------------------------------------------------------------
  # Apply: COM-momentum–projected Psiformer (K_x sector)
  # ---------------------------------------------------------------------------

  def network_apply(
      params,
      pos: jnp.ndarray,
      spins: jnp.ndarray,
      atoms: jnp.ndarray,
      charges: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Evaluate Ψ_{Kx} at positions `pos`."""

    assert options.states == 0, "COM K_x–projected version currently assumes states=0."

    full_ndim = options.ndim
    Ne_local = sum(nspins)

    # Accept flat or (Ne, ndim)
    if pos.ndim == 1:
      if pos.shape[0] != Ne_local * full_ndim:
        raise ValueError(
            f"Flattened pos has length {pos.shape[0]}, "
            f"expected {Ne_local * full_ndim}."
        )
      pos_full = pos.reshape(Ne_local, full_ndim)
      original_shape = "flat"
    elif pos.ndim == 2:
      if pos.shape != (Ne_local, full_ndim):
        raise ValueError(
            f"pos has shape {pos.shape}, expected ({Ne_local}, {full_ndim})."
        )
      pos_full = pos
      original_shape = "matrix"
    else:
      raise ValueError("pos must be 1D ((Ne*ndim,)) or 2D ((Ne,ndim)).")

    pos_full = jnp.asarray(pos_full)
    pos_xy = pos_full[:, :2]  # (Ne,2)

    # Total phase per translation: COM phase + projector phase
    com_phases = compute_com_phases(pos_xy)          # (Nphi,)
    total_phases = com_phases + projector_phases     # (Nphi,)

    float_dtype = pos_full.dtype
    complex_dtype = jnp.complex64 if float_dtype == jnp.float32 else jnp.complex128

    # Streaming complex log-sum-exp across translations (folx-friendly)
    max_logdet = jnp.array(-jnp.inf, dtype=float_dtype)
    sum_norm = jnp.zeros((), dtype=complex_dtype)

    for m in range(Nphi):
      shift_xy = translations[m]      # (2,)
      phase_i = total_phases[m]       # scalar

      # COM shift: r_i -> r_i + a_m
      new_xy = pos_xy + shift_xy[None, :]
      if full_ndim == 2:
        t_pos_full = new_xy
      else:
        higher = pos_full[:, 2:]
        t_pos_full = jnp.concatenate([new_xy, higher], axis=-1)

      if original_shape == "flat":
        t_pos = t_pos_full.reshape(Ne_local * full_ndim)
      else:
        t_pos = t_pos_full

      orbitals = orbitals_apply(params, t_pos, spins, atoms, charges)
      phase_det, logdet = network_blocks.logdet_matmul(orbitals)
      total_phase = phase_det + phase_i

      cond = logdet > max_logdet

      term1 = sum_norm + jnp.exp(logdet - max_logdet + 1j * total_phase)
      term2 = sum_norm * jnp.exp(max_logdet - logdet) + jnp.exp(1j * total_phase)

      sum_norm = jnp.where(cond, term2, term1)
      max_logdet = jnp.where(cond, logdet, max_logdet)

    total = sum_norm * jnp.exp(max_logdet)
    total = total / Nphi_f

    abs_total = jnp.abs(total)
    eps = jnp.asarray(1e-38, dtype=float_dtype)
    abs_total = jnp.where(abs_total == 0.0, eps, abs_total)

    overall_log = jnp.log(abs_total)
    overall_phase = jnp.angle(total)

    if "state_scale" in params:
      overall_log = overall_log + params["state_scale"]

    return overall_phase, overall_log

  return networks.Network(
      options=options,
      init=network_init,
      apply=network_apply,
      orbitals=orbitals_apply,
  )

# def make_fermi_net_momentum_projected(
#     nspins: Tuple[int, ...],
#     charges: jnp.ndarray,
#     *,
#     ndim: int = 3,
#     determinants: int = 16,
#     states: int = 0,
#     envelope: Optional[envelopes.Envelope] = None,
#     feature_layer: Optional[networks.FeatureLayer] = None,
#     jastrow: Union[str, jastrows.JastrowType] = jastrows.JastrowType.SIMPLE_EE,
#     complex_output: bool = False,
#     bias_orbitals: bool = False,
#     rescale_inputs: bool = False,
#     momind : int,
#     mom_kwargs : dict,
#     # Psiformer-specific kwargs below.
#     num_layers: int,
#     num_heads: int,
#     heads_dim: int,
#     mlp_hidden_dims: Tuple[int, ...],
#     use_layer_norm: bool
# ) -> networks.Network:
#   """Psiformer with stacked Self Attention layers.

#   Includes standard envelope and determinants.

#   Args:
#     nspins: Tuple of the number of spin-up and spin-down electrons.
#     charges: (natom) array of atom nuclear charges.
#     ndim: Dimension of the system. Change only with caution.
#     determinants: Number of determinants.
#     states: Number of outputs, one per excited (or ground) state. Ignored if 0.
#     envelope: Envelope to use to impose orbitals go to zero at infinity.
#     feature_layer: Input feature construction.
#     jastrow: Type of Jastrow factor if used, or 'simple_ee' if 'default'.
#     complex_output: If true, the wavefunction output is complex-valued.
#     bias_orbitals: If true, include a bias in the final linear layer to shape
#       the outputs into orbitals.
#     rescale_inputs: If true, rescale the inputs so they grow as log(|r|).
#     num_layers: Number of stacked self-attention layers.
#     num_heads: Number of self-attention heads.
#     heads_dim: Embedding dimension per-head for self-attention.
#     mlp_hidden_dims: Tuple of hidden dimensions of the MLP.
#     use_layer_norm: If true, use layer_norm on both attention and MLP.

#   Returns:
#     Network object containing init, apply, orbitals, options, where init and
#     apply are callables which initialise the network parameters and apply the
#     network respectively, orbitals is a callable which applies the network up to
#     the orbitals, and options specifies the settings used in the network.
#   """

#   if not envelope:
#     envelope = envelopes.make_isotropic_envelope()

#   if not feature_layer:
#     natoms = charges.shape[0]
#     feature_layer = networks.make_ferminet_features(
#         natoms, nspins, ndim=ndim, rescale_inputs=rescale_inputs
#     )

#   if isinstance(jastrow, str):
#     if jastrow.upper() == 'DEFAULT':
#       jastrow = jastrows.JastrowType.SIMPLE_EE
#     else:
#       jastrow = jastrows.JastrowType[jastrow.upper()]

#   options = PsiformerOptions(
#       ndim=ndim,
#       determinants=determinants,
#       states=states,
#       envelope=envelope,
#       feature_layer=feature_layer,
#       jastrow=jastrow,
#       complex_output=complex_output,
#       bias_orbitals=bias_orbitals,
#       full_det=True,  # Required for Psiformer.
#       rescale_inputs=rescale_inputs,
#       num_layers=num_layers,
#       num_heads=num_heads,
#       heads_dim=heads_dim,
#       mlp_hidden_dims=mlp_hidden_dims,
#       use_layer_norm=use_layer_norm,

#   )  # pytype: disable=wrong-keyword-args

#   psiformer_layers = make_psiformer_layers(nspins, charges.shape[0], options)

#   orbitals_init, orbitals_apply = networks.make_orbitals(
#       nspins=nspins,
#       charges=charges,
#       options=options,
#       equivariant_layers=psiformer_layers,
#   )

#   def network_init(key: chex.PRNGKey) -> networks.ParamTree:
#     return orbitals_init(key)

#   abs_lattice = mom_kwargs['abs_lattice']
#   unit_cell_vectors = mom_kwargs['unit_cell_vectors']
#   # Compute lattice and reciprocal lattice vectors
#   lattice = targetmom.lattice_vecs(unit_cell_vectors[0], unit_cell_vectors[1], abs_lattice)
#   reciprocal = targetmom.reciprocal_vecs(unit_cell_vectors[0], unit_cell_vectors[1], np.array([[1,0], [0, 1]]))
#   g1, g2, _, _, _ = targetmom.g1g2(abs_lattice, reciprocal[:,0], reciprocal[:,1])
#   klabels = targetmom.kpoints(abs_lattice)

#   # Find all translations in the supercell
#   translations, _ = targetmom.find_all_translations_in_supercell(lattice, unit_cell_vectors)
#   num_translation = translations.shape[0]
#   mn_result = jnp.array(targetmom.mn(klabels[momind], g1, g2))

#   # Compute the phase shift for all translations at once
#   phase_shifts = jnp.dot(translations, mn_result)  # Shape: (num_translations,)
#   # Reshape phase_shifts to align with orbital_outputs
#   #phase_shifts = jnp.array(phase_shifts).reshape(-1, 1)  # Shape: (num_translations, 1)
#   #phase_shifts = jnp.reshape(phase_shifts, (num_translation, 1, 1, 1))
#   def network_apply(
#       params,
#       pos: jnp.ndarray,
#       spins: jnp.ndarray,
#       atoms: jnp.ndarray,
#       charges: jnp.ndarray,
#   ) -> Tuple[jnp.ndarray, jnp.ndarray]:
#     """Forward evaluation of the Psiformer.

#     Args:
#       params: network parameter tree.
#       pos: The electron positions, a 3N dimensional vector.
#       spins: The electron spins, an N dimensional vector.
#       atoms: Array with positions of atoms.
#       charges: Array with nuclear charges.

#     Returns:
#       Output of antisymmetric neural network in log space, i.e. a tuple of sign
#       of and log absolute value of the network evaluated at x.
#     """


#     # Reshape pos to (N, ndim) for particle-wise operations
#     ndim = 2
#     N = pos.shape[0] // ndim  # Number of particles (assuming ndim is globally defined)
#     reshaped_pos = pos.reshape(N, ndim)

#     # Tile translations to match the positions
#     tiled_translations = jnp.repeat(translations[:, None, :], N, axis=1)  # Shape: (num_translations, N, ndim)
#     translated_positions = reshaped_pos[None, :, :] + tiled_translations  # Shape: (num_translations, N, ndim)

#     # Flatten translated_positions back to (num_translations, N * ndim)
#     flattened_translated_positions = translated_positions.reshape(translated_positions.shape[0], -1)

#     # Compute the orbitals for all translations
#     #orbital_outputs = jax.vmap(
#     #    lambda t_pos: orbitals_apply(params, t_pos, spins, atoms, charges)[0]
#     #)(flattened_translated_positions)  # Shape: (num_translations, N_det)

#     # Use jax.vmap to vectorize the application of the orbitals_apply function
#     """    orbital_outputs = jax.vmap(lambda t_pos: orbitals_apply(params, t_pos, spins, atoms, charges)[0])(
#         flattened_translated_positions
#     )"""
#     orbital_outputs = []
#     for t_pos in flattened_translated_positions:
#       orbital_outputs.append(orbitals_apply(params, t_pos, spins, atoms, charges)[0])


#     # Convert the result to a JAX array (if not already a JAX array)
#     orbital_outputs = jnp.array(orbital_outputs)
    
#     assert options.states == 0

#     def compute_phase_and_logdet(orbital_outputs: jnp.ndarray, phase_shifts: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
#       """
#       Compute the phase and log of the determinant by summing over translations using the logsumexp trick.

#       Args:
#           orbital_outputs: Array of shape (N_trans, N_det, N, N).
#           phase_shifts: Array of shape (N_trans,).

#       Returns:
#           A tuple containing the overall phase and log of the determinant.
#       """
#       log_determinants = []
#       phases = []

#       for i in range(orbital_outputs.shape[0]):  # Loop over N_trans
#           # Apply logdet_matmul to get phase and logdet
#           phase, logdet = network_blocks.logdet_matmul([orbital_outputs[i]])

#           # Add the phase shift to the phase
#           phase += phase_shifts[i]

#           # Store the logdet and phase
#           log_determinants.append(logdet)
#           phases.append(phase)


#       # Convert to JAX arrays
#       log_determinants = jnp.array(log_determinants)
#       phases = jnp.array(phases)

#       # Use the logsumexp trick for numerical stability
#       max_logdet = jnp.max(log_determinants)
#       shifted_exps = jnp.exp(log_determinants - max_logdet + 1j * phases)  # Include phase in the exponential
#       total_determinant = jnp.sum(shifted_exps)
#       total_determinant /= orbital_outputs.shape[0]


#       # Compute the overall log and phase
#       overall_log = max_logdet + jnp.log(jnp.abs(total_determinant))
#       overall_phase = jnp.angle(total_determinant)

#       return overall_phase, overall_log
  

#     # Perform element-wise multiplication and summation for each translation
#     #projected_orbitals = sum(
#     #[orbital_output * phase for orbital_output, phase in zip(orbital_outputs, phase_shifts)])
#     #projected_orbitals = sum(orbi)
    
#     overall_phase, overall_log = compute_phase_and_logdet(orbital_outputs, phase_shifts)

#     return overall_phase, overall_log

#   return networks.Network(
#       options=options,
#       init=network_init,
#       apply=network_apply,
#       orbitals=orbitals_apply,
#   )




def make_fermi_net_momentum_projected(
    nspins: Tuple[int, ...],
    charges: jnp.ndarray,
    *,
    ndim: int = 3,
    determinants: int = 16,
    states: int = 0,
    envelope: Optional[envelopes.Envelope] = None,
    feature_layer: Optional[networks.FeatureLayer] = None,
    jastrow: Union[str, jastrows.JastrowType] = jastrows.JastrowType.SIMPLE_EE,
    jastrow_kwargs: dict = {},
    complex_output: bool = False,
    bias_orbitals: bool = False,
    rescale_inputs: bool = False,
    momind : int,
    mom_kwargs : dict,
    # Psiformer-specific kwargs below.
    num_layers: int,
    num_heads: int,
    heads_dim: int,
    mlp_hidden_dims: Tuple[int, ...],
    use_layer_norm: bool,
    pbc_lattice: jnp.ndarray,
) -> networks.Network:
  """Psiformer with stacked Self Attention layers.

  Includes standard envelope and determinants.

  Args:
    nspins: Tuple of the number of spin-up and spin-down electrons.
    charges: (natom) array of atom nuclear charges.
    ndim: Dimension of the system. Change only with caution.
    determinants: Number of determinants.
    states: Number of outputs, one per excited (or ground) state. Ignored if 0.
    envelope: Envelope to use to impose orbitals go to zero at infinity.
    feature_layer: Input feature construction.
    jastrow: Type of Jastrow factor if used, or 'simple_ee' if 'default'.
    complex_output: If true, the wavefunction output is complex-valued.
    bias_orbitals: If true, include a bias in the final linear layer to shape
      the outputs into orbitals.
    rescale_inputs: If true, rescale the inputs so they grow as log(|r|).
    num_layers: Number of stacked self-attention layers.
    num_heads: Number of self-attention heads.
    heads_dim: Embedding dimension per-head for self-attention.
    mlp_hidden_dims: Tuple of hidden dimensions of the MLP.
    use_layer_norm: If true, use layer_norm on both attention and MLP.

  Returns:
    Network object containing init, apply, orbitals, options, where init and
    apply are callables which initialise the network parameters and apply the
    network respectively, orbitals is a callable which applies the network up to
    the orbitals, and options specifies the settings used in the network.
  """

  if not envelope:
    envelope = envelopes.make_isotropic_envelope()

  if not feature_layer:
    natoms = charges.shape[0]
    feature_layer = networks.make_ferminet_features(
        natoms, nspins, ndim=ndim, rescale_inputs=rescale_inputs
    )

  if isinstance(jastrow, str):
    if jastrow.upper() == 'DEFAULT':
      jastrow = jastrows.JastrowType.SIMPLE_EE
    else:
      jastrow = jastrows.JastrowType[jastrow.upper()]

  options = PsiformerOptions(
      ndim=ndim,
      determinants=determinants,
      states=states,
      envelope=envelope,
      feature_layer=feature_layer,
      jastrow=jastrow,
      jastrow_kwargs=jastrow_kwargs,
      complex_output=complex_output,
      bias_orbitals=bias_orbitals,
      full_det=True,  # Required for Psiformer.
      rescale_inputs=rescale_inputs,
      num_layers=num_layers,
      num_heads=num_heads,
      heads_dim=heads_dim,
      mlp_hidden_dims=mlp_hidden_dims,
      use_layer_norm=use_layer_norm,
      pbc_lattice = pbc_lattice,
  )

  psiformer_layers = make_psiformer_layers(nspins, charges.shape[0], options)

  orbitals_init, orbitals_apply = networks.make_orbitals(
      nspins=nspins,
      charges=charges,
      options=options,
      equivariant_layers=psiformer_layers,
  )

  def network_init(key: chex.PRNGKey) -> networks.ParamTree:
    return orbitals_init(key)

  abs_lattice = mom_kwargs['abs_lattice']
  unit_cell_vectors = mom_kwargs['unit_cell_vectors']
  # Compute lattice and reciprocal lattice vectors
  lattice = targetmom.lattice_vecs(unit_cell_vectors[0], unit_cell_vectors[1], abs_lattice)
  reciprocal = targetmom.reciprocal_vecs(unit_cell_vectors[0], unit_cell_vectors[1], np.array([[1,0], [0, 1]]))
  g1, g2, _, _, _ = targetmom.g1g2(abs_lattice, reciprocal[:,0], reciprocal[:,1])
  klabels = targetmom.kpoints(abs_lattice)

  # Find all translations in the supercell
  translations, _ = targetmom.find_all_translations_in_supercell(lattice, unit_cell_vectors)
  num_translation = translations.shape[0]
  mn_result = jnp.array(targetmom.mn(klabels[momind], g1, g2))

  # Compute the phase shift for all translations at once
  phase_shifts = jnp.dot(translations, mn_result)  # Shape: (num_translations,)
  # Reshape phase_shifts to align with orbital_outputs
  #phase_shifts = jnp.array(phase_shifts).reshape(-1, 1)  # Shape: (num_translations, 1)
  #phase_shifts = jnp.reshape(phase_shifts, (num_translation, 1, 1, 1))
  def network_apply(
    params,
    pos: jnp.ndarray,
    spins: jnp.ndarray,
    atoms: jnp.ndarray,
    charges: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Forward evaluation of the Psiformer.

    Args:
      params: network parameter tree.
      pos: The electron positions, a 3N dimensional vector.
      spins: The electron spins, an N dimensional vector.
      atoms: Array with positions of atoms.
      charges: Array with nuclear charges.

    Returns:
      Output of antisymmetric neural network in log space, i.e. a tuple of sign
      of and log absolute value of the network evaluated at x.
    """

    # Reshape pos to (N, ndim) for particle-wise operations
    ndim = 2
    N = pos.shape[0] // ndim  # Number of particles (assuming ndim is globally defined)
    reshaped_pos = pos.reshape(N, ndim)

    # Tile translations to match the positions
    tiled_translations = jnp.repeat(translations[:, None, :], N, axis=1)  # Shape: (num_translations, N, ndim)
    translated_positions = reshaped_pos[None, :, :] + tiled_translations  # Shape: (num_translations, N, ndim)

    # Flatten translated_positions back to (num_translations, N * ndim)
    flattened_translated_positions = translated_positions.reshape(translated_positions.shape[0], -1)

    assert options.states == 0

    def compute_phase_and_logdet_on_the_fly(phase_shifts: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute the phase and log of the determinant by summing over translations using the logsumexp trick.

        Args:
            phase_shifts: Array of shape (N_trans,).

        Returns:
            A tuple containing the overall phase and log of the determinant.
        """
        log_determinants = []
        phases = []

        for i, t_pos in enumerate(flattened_translated_positions):
            # Compute the orbitals for the current translation on the fly

            # Apply logdet_matmul to get phase and logdet
            phase, logdet = network_blocks.logdet_matmul(orbitals_apply(params, t_pos, spins, atoms, charges))

            # Add the phase shift to the phase
            phase += phase_shifts[i]

            # Store the logdet and phase
            log_determinants.append(logdet)
            phases.append(phase)

        # Convert to JAX arrays
        log_determinants = jnp.array(log_determinants)
        phases = jnp.array(phases)

        # Use the logsumexp trick for numerical stability
        max_logdet = jnp.max(log_determinants)
        shifted_exps = jnp.exp(log_determinants - max_logdet + 1j * phases)  # Include phase in the exponential
        total_determinant = jnp.sum(shifted_exps)
        total_determinant /= len(flattened_translated_positions)

        # Compute the overall log and phase
        overall_log = max_logdet + jnp.log(jnp.abs(total_determinant))
        overall_phase = jnp.angle(total_determinant)

        return overall_phase, overall_log

    overall_phase, overall_log = compute_phase_and_logdet_on_the_fly(phase_shifts)

    return overall_phase, overall_log

  return networks.Network(
      options=options,
      init=network_init,
      apply=network_apply,
      orbitals=orbitals_apply,
  )


# def make_fermi_net_momentum_projected(
#     nspins: Tuple[int, ...],
#     charges: jnp.ndarray,
#     *,
#     ndim: int = 3,
#     determinants: int = 16,
#     states: int = 0,
#     envelope: Optional[envelopes.Envelope] = None,
#     feature_layer: Optional[networks.FeatureLayer] = None,
#     jastrow: Union[str, jastrows.JastrowType] = jastrows.JastrowType.SIMPLE_EE,
#     complex_output: bool = False,
#     bias_orbitals: bool = False,
#     rescale_inputs: bool = False,
#     momind: int,
#     mom_kwargs: dict,
#     # Psiformer-specific kwargs below.
#     num_layers: int,
#     num_heads: int,
#     heads_dim: int,
#     mlp_hidden_dims: Tuple[int, ...],
#     use_layer_norm: bool
#   ) -> networks.Network:
#     """Psiformer with stacked Self Attention layers.

#     Includes standard envelope and determinants.

#     Args:
#         nspins: Tuple of the number of spin-up and spin-down electrons.
#         charges: (natom) array of atom nuclear charges.
#         ndim: Dimension of the system. Change only with caution.
#         determinants: Number of determinants.
#         states: Number of outputs, one per excited (or ground) state. Ignored if 0.
#         envelope: Envelope to use to impose orbitals go to zero at infinity.
#         feature_layer: Input feature construction.
#         jastrow: Type of Jastrow factor if used, or 'simple_ee' if 'default'.
#         complex_output: If true, the wavefunction output is complex-valued.
#         bias_orbitals: If true, include a bias in the final linear layer to shape
#           the outputs into orbitals.
#         rescale_inputs: If true, rescale the inputs so they grow as log(|r|).
#         num_layers: Number of stacked self-attention layers.
#         num_heads: Number of self-attention heads.
#         heads_dim: Embedding dimension per-head for self-attention.
#         mlp_hidden_dims: Tuple of hidden dimensions of the MLP.
#         use_layer_norm: If true, use layer_norm on both attention and MLP.

#     Returns:
#         Network object containing init, apply, orbitals, options, where init and
#         apply are callables which initialise the network parameters and apply the
#         network respectively, orbitals is a callable which applies the network up to
#         the orbitals, and options specifies the settings used in the network.
#     """

#     if not envelope:
#         envelope = envelopes.make_isotropic_envelope()

#     if not feature_layer:
#         natoms = charges.shape[0]
#         feature_layer = networks.make_ferminet_features(
#             natoms, nspins, ndim=ndim, rescale_inputs=rescale_inputs
#         )

#     if isinstance(jastrow, str):
#         if jastrow.upper() == 'DEFAULT':
#             jastrow = jastrows.JastrowType.SIMPLE_EE
#         else:
#             jastrow = jastrows.JastrowType[jastrow.upper()]

#     options = PsiformerOptions(
#         ndim=ndim,
#         determinants=determinants,
#         states=states,
#         envelope=envelope,
#         feature_layer=feature_layer,
#         jastrow=jastrow,
#         complex_output=complex_output,
#         bias_orbitals=bias_orbitals,
#         full_det=True,  # Required for Psiformer.
#         rescale_inputs=rescale_inputs,
#         num_layers=num_layers,
#         num_heads=num_heads,
#         heads_dim=heads_dim,
#         mlp_hidden_dims=mlp_hidden_dims,
#         use_layer_norm=use_layer_norm,
#     )  # pytype: disable=wrong-keyword-args

#     psiformer_layers = make_psiformer_layers(nspins, charges.shape[0], options)

#     orbitals_init, orbitals_apply = networks.make_orbitals(
#         nspins=nspins,
#         charges=charges,
#         options=options,
#         equivariant_layers=psiformer_layers,
#     )

#     def network_init(key: chex.PRNGKey) -> networks.ParamTree:
#         return orbitals_init(key)

#     # --- precompute (ensure device float32) ---
#     abs_lattice = mom_kwargs['abs_lattice']
#     unit_cell_vectors = mom_kwargs['unit_cell_vectors']

#     lattice = targetmom.lattice_vecs(unit_cell_vectors[0], unit_cell_vectors[1], abs_lattice)
#     reciprocal = targetmom.reciprocal_vecs(
#         unit_cell_vectors[0], unit_cell_vectors[1], jnp.array([[1, 0], [0, 1]], dtype=jnp.float32)
#     )
#     g1, g2, _, _, _ = targetmom.g1g2(abs_lattice, reciprocal[:, 0], reciprocal[:, 1])

#     klabels = targetmom.kpoints(abs_lattice)
#     translations, _ = targetmom.find_all_translations_in_supercell(lattice, unit_cell_vectors)

#     translations = jnp.asarray(translations, dtype=jnp.float32)                  # (T, ndim)
#     mn_result = jnp.asarray(targetmom.mn(klabels[momind], g1, g2), jnp.float32)  # (ndim,)
#     phase_shifts = (translations @ mn_result).astype(jnp.float32)                # (T,)
#     T = translations.shape[0]
#     logT = jnp.log(jnp.asarray(T, dtype=jnp.float32))
#     # ... same precompute as your float32 version ...
#     translations = lax.stop_gradient(translations)
#     phase_shifts = lax.stop_gradient(phase_shifts)

#     def network_apply(
#         params,
#         pos: jnp.ndarray,
#         spins: jnp.ndarray,
#         atoms: jnp.ndarray,
#         charges: jnp.ndarray,
#     ) -> Tuple[jnp.ndarray, jnp.ndarray]:
#       """Momentum-projected apply; float32, memory-lean, checkpointed Psiformer forward."""
#       assert options.states == 0

#       # Reshape once
#       ndim_local = options.ndim
#       N = pos.shape[0] // ndim_local
#       reshaped_pos = pos.reshape(N, ndim_local)

#       # Treat geometry/momentum data as constants for AD; keep as float32
#       const_translations = jax.lax.stop_gradient(translations.astype(jnp.float32))   # (T, ndim)
#       const_phase_shifts = jax.lax.stop_gradient(phase_shifts.astype(jnp.float32))   # (T,)
#       T   = const_translations.shape[0]
#       logT = jnp.log(jnp.asarray(T, dtype=jnp.float32))

#       def body(i, carry):
#         m, Sr, Si = carry
#         tvec      = const_translations[i]   # (ndim,)
#         phi_shift = const_phase_shifts[i]   # ()

#         # Translate positions without materializing (T, N, ndim)
#         flatpos = (reshaped_pos + tvec).reshape(-1)  # (N*ndim,)

#         @jax.checkpoint
#         def eval_orbitals(p, x, s, a, c):
#           # Only the heavy forward is rematerialized to save backward memory.
#           return orbitals_apply(p, x, s, a, c)

#         orbital_output = eval_orbitals(params, flatpos, spins, atoms, charges)

#         # Per-translation complex contribution: det = exp(logabs_i) * exp(i*phase_i)
#         phase_i, logabs_i = network_blocks.logdet_matmul(orbital_output)
#         phi_i = phase_i + phi_shift

#         # Stable streaming accumulation in log domain:
#         # keep running max m and normalized complex sum (Sr,Si)
#         m_new     = jnp.maximum(m, logabs_i)
#         scale_old = jnp.exp(m - m_new)
#         scale_new = jnp.exp(logabs_i - m_new)
#         Sr_new    = Sr * scale_old + scale_new * jnp.cos(phi_i)
#         Si_new    = Si * scale_old + scale_new * jnp.sin(phi_i)
#         return (m_new, Sr_new, Si_new)

#       # Tiny carry only (three scalars), all float32
#       init = (
#           jnp.array(-jnp.inf, dtype=jnp.float32),  # running max log-magnitude
#           jnp.array(0.0, dtype=jnp.float32),       # real part of normalized sum
#           jnp.array(0.0, dtype=jnp.float32),       # imag part of normalized sum
#       )

#       m, Sr, Si = jax.lax.fori_loop(0, T, body, init)

#       # Average over translations and return phase/log|sum|
#       overall_phase = jnp.arctan2(Si, Sr)
#       overall_log   = m + jnp.log(jnp.hypot(Sr, Si)) - logT
#       return overall_phase, overall_log

#     return networks.Network(
#         options=options,
#         init=network_init,
#         apply=network_apply,
#         orbitals=orbitals_apply,
#     )

def make_fermi_net_magneticfield(
    nspins: Tuple[int, ...],
    charges: jnp.ndarray,
    *,
    ndim: int = 3,
    determinants: int = 16,
    states: int = 0,
    envelope: Optional[envelopes.Envelope] = None,
    feature_layer: Optional[networks.FeatureLayer] = None,
    jastrow: Union[str, jastrows.JastrowType] = jastrows.JastrowType.SIMPLE_EE,
    complex_output: bool = False,
    bias_orbitals: bool = False,
    rescale_inputs: bool = False,
    momind : int,
    mom_kwargs : dict,
    # Psiformer-specific kwargs below.
    num_layers: int,
    num_heads: int,
    heads_dim: int,
    mlp_hidden_dims: Tuple[int, ...],
    use_layer_norm: bool
) -> networks.Network:
  """Psiformer with stacked Self Attention layers.

  Includes standard envelope and determinants.

  Args:
    nspins: Tuple of the number of spin-up and spin-down electrons.
    charges: (natom) array of atom nuclear charges.
    ndim: Dimension of the system. Change only with caution.
    determinants: Number of determinants.
    states: Number of outputs, one per excited (or ground) state. Ignored if 0.
    envelope: Envelope to use to impose orbitals go to zero at infinity.
    feature_layer: Input feature construction.
    jastrow: Type of Jastrow factor if used, or 'simple_ee' if 'default'.
    complex_output: If true, the wavefunction output is complex-valued.
    bias_orbitals: If true, include a bias in the final linear layer to shape
      the outputs into orbitals.
    rescale_inputs: If true, rescale the inputs so they grow as log(|r|).
    num_layers: Number of stacked self-attention layers.
    num_heads: Number of self-attention heads.
    heads_dim: Embedding dimension per-head for self-attention.
    mlp_hidden_dims: Tuple of hidden dimensions of the MLP.
    use_layer_norm: If true, use layer_norm on both attention and MLP.

  Returns:
    Network object containing init, apply, orbitals, options, where init and
    apply are callables which initialise the network parameters and apply the
    network respectively, orbitals is a callable which applies the network up to
    the orbitals, and options specifies the settings used in the network.
  """

  if not envelope:
    envelope = envelopes.make_isotropic_envelope()

  if not feature_layer:
    natoms = charges.shape[0]
    feature_layer = networks.make_ferminet_features(
        natoms, nspins, ndim=ndim, rescale_inputs=rescale_inputs
    )

  if isinstance(jastrow, str):
    if jastrow.upper() == 'DEFAULT':
      jastrow = jastrows.JastrowType.SIMPLE_EE
    else:
      jastrow = jastrows.JastrowType[jastrow.upper()]

  options = PsiformerOptions(
      ndim=ndim,
      determinants=determinants,
      states=states,
      envelope=envelope,
      feature_layer=feature_layer,
      jastrow=jastrow,
      complex_output=complex_output,
      bias_orbitals=bias_orbitals,
      full_det=True,  # Required for Psiformer.
      rescale_inputs=rescale_inputs,
      num_layers=num_layers,
      num_heads=num_heads,
      heads_dim=heads_dim,
      mlp_hidden_dims=mlp_hidden_dims,
      use_layer_norm=use_layer_norm,

  )  # pytype: disable=wrong-keyword-args

  psiformer_layers = make_psiformer_layers(nspins, charges.shape[0], options)

  orbitals_init, orbitals_apply = networks.make_orbitals(
      nspins=nspins,
      charges=charges,
      options=options,
      equivariant_layers=psiformer_layers,
  )

  def network_init(key: chex.PRNGKey) -> networks.ParamTree:
    return orbitals_init(key)

  abs_lattice = mom_kwargs['abs_lattice']
  unit_cell_vectors = mom_kwargs['unit_cell_vectors']
  ell = mom_kwargs["magnetic_length"]
  # Compute lattice and reciprocal lattice vectors
  lattice = targetmom.lattice_vecs(unit_cell_vectors[0], unit_cell_vectors[1], abs_lattice)
  reciprocal = targetmom.reciprocal_vecs(unit_cell_vectors[0], unit_cell_vectors[1], np.array([[1,0], [0, 1]]))
  g1, g2, _, _, _ = targetmom.g1g2(abs_lattice, reciprocal[:,0], reciprocal[:,1])
  klabels = targetmom.kpoints(abs_lattice)

  # Find all translations in the supercell
  translations, xylist = targetmom.find_all_translations_in_supercell(lattice, unit_cell_vectors)
  num_translation = translations.shape[0]
  mn_result =  jnp.array(targetmom.mn(klabels[momind], g1, g2))

  # Compute the phase shift for all translations at once
  phase_shifts = jnp.dot(translations, mn_result)  # Shape: (num_translations,)
  # Reshape phase_shifts to align with orbital_outputs
  #phase_shifts = jnp.array(phase_shifts).reshape(-1, 1)  # Shape: (num_translations, 1)
  #phase_shifts = jnp.reshape(phase_shifts, (num_translation, 1, 1, 1))
  def network_apply(
      params,
      pos: jnp.ndarray,
      spins: jnp.ndarray,
      atoms: jnp.ndarray,
      charges: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Forward evaluation of the Psiformer.

    Args:
      params: network parameter tree.
      pos: The electron positions, a 3N dimensional vector.
      spins: The electron spins, an N dimensional vector.
      atoms: Array with positions of atoms.
      charges: Array with nuclear charges.

    Returns:
      Output of antisymmetric neural network in log space, i.e. a tuple of sign
      of and log absolute value of the network evaluated at x.
    """


    # Reshape pos to (N, ndim) for particle-wise operations
    ndim = 2
    N = pos.shape[0] // ndim  # Number of particles (assuming ndim is globally defined)
    reshaped_pos = pos.reshape(N, ndim)

    # Tile translations to match the positions
    tiled_translations = jnp.repeat(translations[:, None, :], N, axis=1)  # Shape: (num_translations, N, ndim)
    translated_positions = reshaped_pos[None, :, :] + tiled_translations  # Shape: (num_translations, N, ndim)

    # Flatten translated_positions back to (num_translations, N * ndim)
    flattened_translated_positions = translated_positions.reshape(translated_positions.shape[0], -1)

    def compute_magnetic_phase(translations, positions, l_b):
      # Reshape positions from (N * ndim,) to (N, ndim)
      ndim = translations.shape[-1]  # Determine the dimensionality (e.g., 2 for 2D)
      positions = positions.reshape(-1, ndim)  # Reshape positions to (N, ndim)

      # Reshape positions to (N, 1, ndim) and translations to (1, N_translation, ndim) for broadcasting
      positions = positions[:, None, :]  # (N, 1, ndim)
      translations = translations[None, :, :]  # (1, N_translation, ndim)

      # Compute the cross product r × R (only the z-component matters in 2D)
      cross_product_z = positions[..., 0] * translations[..., 1] - positions[..., 1] * translations[..., 0]  # (N, N_translation)

      # Sum over positions (axis=0) instead of translations
      magnetic_phase =  jnp.sum(cross_product_z, axis=0) / (2*l_b**2)  # (N_translation,)
      
      magnetic_phase =  jnp.mod(magnetic_phase, 2 * jnp.pi)  # Wrap the phase to [0, 2*pi)

      # Reshape to (N_translation, 1) for the desired output shape
      return magnetic_phase
      
    magnetic_phase = compute_magnetic_phase(translations, pos, ell)

    # Use jax.vmap to vectorize the application of the orbitals_apply function
    """    orbital_outputs = jax.vmap(lambda t_pos: orbitals_apply(params, t_pos, spins, atoms, charges)[0])(
        flattened_translated_positions
    )"""
    orbital_outputs = []
    for t_pos in flattened_translated_positions:
      orbital_outputs.append(orbitals_apply(params, t_pos, spins, atoms, charges)[0])


    # Convert the result to a JAX array (if not already a JAX array)
    orbital_outputs = jnp.array(orbital_outputs)
    
    assert options.states == 0

    def compute_phase_and_logdet(orbital_outputs: jnp.ndarray, phase_shifts: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
      """
      Compute the phase and log of the determinant by summing over translations using the logsumexp trick.

      Args:
          orbital_outputs: Array of shape (N_trans, N_det, N, N).
          phase_shifts: Array of shape (N_trans,).

      Returns:
          A tuple containing the overall phase and log of the determinant.
      """
      log_determinants = []
      phases = []

      for i in range(orbital_outputs.shape[0]):  # Loop over N_trans
          # Apply logdet_matmul to get phase and logdet
          phase, logdet = network_blocks.logdet_matmul([orbital_outputs[i]])
          # Add the phase shift to the phase
          phase += phase_shifts[i]  + magnetic_phase[i] + cocycles[i]

          # Store the logdet and phase
          log_determinants.append(logdet)
          phases.append(phase)


      # Convert to JAX arrays
      log_determinants = jnp.array(log_determinants)
      phases = jnp.array(phases)

      # Use the logsumexp trick for numerical stability
      max_logdet = jnp.max(log_determinants)
      shifted_exps = jnp.exp(log_determinants - max_logdet + 1j * phases)  # Include phase in the exponential
      total_determinant = jnp.sum(shifted_exps)
      total_determinant /= orbital_outputs.shape[0]


      # Compute the overall log and phase
      overall_log = max_logdet + jnp.log(jnp.abs(total_determinant))
      overall_phase = jnp.angle(total_determinant)

      return overall_phase, overall_log
  

    # Perform element-wise multiplication and summation for each translation
    #projected_orbitals = sum(
    #[orbital_output * phase for orbital_output, phase in zip(orbital_outputs, phase_shifts)])
    #projected_orbitals = sum(orbi)
    
    overall_phase, overall_log = compute_phase_and_logdet(orbital_outputs, phase_shifts)

    return overall_phase, overall_log

  return networks.Network(
      options=options,
      init=network_init,
      apply=network_apply,
      orbitals=orbitals_apply,
  )


def make_boson_net_sum(
    nspins: Tuple[int, ...],
    charges: jnp.ndarray,
    *,
    ndim: int = 3,
    determinants: int = 16,
    states: int = 0,
    envelope: Optional[envelopes.Envelope] = None,
    feature_layer: Optional[networks.FeatureLayer] = None,
    jastrow: Union[str, jastrows.JastrowType] = jastrows.JastrowType.SIMPLE_EE,
    complex_output: bool = False,
    bias_orbitals: bool = False,
    rescale_inputs: bool = False,
    # Psiformer-specific kwargs below.
    num_layers: int,
    num_heads: int,
    heads_dim: int,
    mlp_hidden_dims: Tuple[int, ...],
    use_layer_norm: bool,
) -> networks.Network:
  """Psiformer with stacked Self Attention layers.

  Includes standard envelope and determinants.

  Args:
    nspins: Tuple of the number of spin-up and spin-down electrons.
    charges: (natom) array of atom nuclear charges.
    ndim: Dimension of the system. Change only with caution.
    determinants: Number of determinants.
    states: Number of outputs, one per excited (or ground) state. Ignored if 0.
    envelope: Envelope to use to impose orbitals go to zero at infinity.
    feature_layer: Input feature construction.
    jastrow: Type of Jastrow factor if used, or 'simple_ee' if 'default'.
    complex_output: If true, the wavefunction output is complex-valued.
    bias_orbitals: If true, include a bias in the final linear layer to shape
      the outputs into orbitals.
    rescale_inputs: If true, rescale the inputs so they grow as log(|r|).
    num_layers: Number of stacked self-attention layers.
    num_heads: Number of self-attention heads.
    heads_dim: Embedding dimension per-head for self-attention.
    mlp_hidden_dims: Tuple of hidden dimensions of the MLP.
    use_layer_norm: If true, use layer_norm on both attention and MLP.

  Returns:
    Network object containing init, apply, orbitals, options, where init and
    apply are callables which initialise the network parameters and apply the
    network respectively, orbitals is a callable which applies the network up to
    the orbitals, and options specifies the settings used in the network.
  """

  if not envelope:
    envelope = envelopes.make_isotropic_envelope()

  if not feature_layer:
    natoms = charges.shape[0]
    feature_layer = networks.make_ferminet_features(
        natoms, nspins, ndim=ndim, rescale_inputs=rescale_inputs
    )

  if isinstance(jastrow, str):
    if jastrow.upper() == 'DEFAULT':
      jastrow = jastrows.JastrowType.SIMPLE_EE
    else:
      jastrow = jastrows.JastrowType[jastrow.upper()]

  options = PsiformerOptions(
      ndim=ndim,
      determinants=determinants,
      states=states,
      envelope=envelope,
      feature_layer=feature_layer,
      jastrow=jastrow,
      complex_output=complex_output,
      bias_orbitals=bias_orbitals,
      full_det=True,  # Required for Psiformer.
      rescale_inputs=rescale_inputs,
      num_layers=num_layers,
      num_heads=num_heads,
      heads_dim=heads_dim,
      mlp_hidden_dims=mlp_hidden_dims,
      use_layer_norm=use_layer_norm,
  )  # pytype: disable=wrong-keyword-args

  psiformer_layers = make_psiformer_layers(nspins, charges.shape[0], options)

  orbitals_init, orbitals_apply = networks.make_orbitals_bosons_sum(
      nspins=nspins,
      charges=charges,
      options=options,
      equivariant_layers=psiformer_layers,
  )

  def network_init(key: chex.PRNGKey) -> networks.ParamTree:
    return orbitals_init(key)

  def network_apply(
      params,
      pos: jnp.ndarray,
      spins: jnp.ndarray,
      atoms: jnp.ndarray,
      charges: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Forward evaluation of the Psiformer.

    Args:
      params: network parameter tree.
      pos: The electron positions, a 3N dimensional vector.
      spins: The electron spins, an N dimensional vector.
      atoms: Array with positions of atoms.
      charges: Array with nuclear charges.

    Returns:
      Output of symmetric neural network in log space, i.e. a tuple of sign
      of and log absolute value of the network evaluated at x.
    """
    orbitals = orbitals_apply(params, pos, spins, atoms, charges)
    if options.states:
      batch_logdet_matmul = jax.vmap(network_blocks.logdet_matmul, in_axes=0)
      orbitals = [
          jnp.reshape(orbital, (options.states, -1) + orbital.shape[1:])
          for orbital in orbitals
      ]
      result = batch_logdet_matmul(orbitals)
    else:
      result = network_blocks.logdet_matmul_bosons(orbitals)
    if 'state_scale' in params:
      # only used at inference time for excited states
      result = result[0], result[1] + params['state_scale']
    return result

  return networks.Network(
      options=options,
      init=network_init,
      apply=network_apply,
      orbitals=orbitals_apply,
  )

def make_boson_net_prod(
    nspins: Tuple[int, ...],
    charges: jnp.ndarray,
    *,
    ndim: int = 3,
    determinants: int = 16,
    states: int = 0,
    envelope: Optional[envelopes.Envelope] = None,
    feature_layer: Optional[networks.FeatureLayer] = None,
    jastrow: Union[str, jastrows.JastrowType] = jastrows.JastrowType.SIMPLE_EE,
    complex_output: bool = False,
    bias_orbitals: bool = False,
    rescale_inputs: bool = False,
    # Psiformer-specific kwargs below.
    num_layers: int,
    num_heads: int,
    heads_dim: int,
    mlp_hidden_dims: Tuple[int, ...],
    use_layer_norm: bool,
) -> networks.Network:
  """Psiformer with stacked Self Attention layers.

  Includes standard envelope and determinants.

  Args:
    nspins: Tuple of the number of spin-up and spin-down electrons.
    charges: (natom) array of atom nuclear charges.
    ndim: Dimension of the system. Change only with caution.
    determinants: Number of determinants.
    states: Number of outputs, one per excited (or ground) state. Ignored if 0.
    envelope: Envelope to use to impose orbitals go to zero at infinity.
    feature_layer: Input feature construction.
    jastrow: Type of Jastrow factor if used, or 'simple_ee' if 'default'.
    complex_output: If true, the wavefunction output is complex-valued.
    bias_orbitals: If true, include a bias in the final linear layer to shape
      the outputs into orbitals.
    rescale_inputs: If true, rescale the inputs so they grow as log(|r|).
    num_layers: Number of stacked self-attention layers.
    num_heads: Number of self-attention heads.
    heads_dim: Embedding dimension per-head for self-attention.
    mlp_hidden_dims: Tuple of hidden dimensions of the MLP.
    use_layer_norm: If true, use layer_norm on both attention and MLP.

  Returns:
    Network object containing init, apply, orbitals, options, where init and
    apply are callables which initialise the network parameters and apply the
    network respectively, orbitals is a callable which applies the network up to
    the orbitals, and options specifies the settings used in the network.
  """

  if not envelope:
    envelope = envelopes.make_isotropic_envelope()

  if not feature_layer:
    natoms = charges.shape[0]
    feature_layer = networks.make_ferminet_features(
        natoms, nspins, ndim=ndim, rescale_inputs=rescale_inputs
    )

  if isinstance(jastrow, str):
    if jastrow.upper() == 'DEFAULT':
      jastrow = jastrows.JastrowType.SIMPLE_EE
    else:
      jastrow = jastrows.JastrowType[jastrow.upper()]

  options = PsiformerOptions(
      ndim=ndim,
      determinants=determinants,
      states=states,
      envelope=envelope,
      feature_layer=feature_layer,
      jastrow=jastrow,
      complex_output=complex_output,
      bias_orbitals=bias_orbitals,
      full_det=True,  # Required for Psiformer.
      rescale_inputs=rescale_inputs,
      num_layers=num_layers,
      num_heads=num_heads,
      heads_dim=heads_dim,
      mlp_hidden_dims=mlp_hidden_dims,
      use_layer_norm=use_layer_norm,
  )  # pytype: disable=wrong-keyword-args

  psiformer_layers = make_psiformer_layers(nspins, charges.shape[0], options)

  orbitals_init, orbitals_apply = networks.make_orbitals_bosons_prod(
      nspins=nspins,
      charges=charges,
      options=options,
      equivariant_layers=psiformer_layers,
  )

  def network_init(key: chex.PRNGKey) -> networks.ParamTree:
    return orbitals_init(key)

  def network_apply(
      params,
      pos: jnp.ndarray,
      spins: jnp.ndarray,
      atoms: jnp.ndarray,
      charges: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Forward evaluation of the Psiformer.

    Args:
      params: network parameter tree.
      pos: The electron positions, a 3N dimensional vector.
      spins: The electron spins, an N dimensional vector.
      atoms: Array with positions of atoms.
      charges: Array with nuclear charges.

    Returns:
      Output of symmetric neural network in log space, i.e. a tuple of sign
      of and log absolute value of the network evaluated at x.
    """
    orbitals = orbitals_apply(params, pos, spins, atoms, charges)
    if options.states:
      batch_logdet_matmul = jax.vmap(network_blocks.logdet_matmul, in_axes=0)
      orbitals = [
          jnp.reshape(orbital, (options.states, -1) + orbital.shape[1:])
          for orbital in orbitals
      ]
      result = batch_logdet_matmul(orbitals)
    else:
      result = network_blocks.logdet_matmul_bosons(orbitals)
    if 'state_scale' in params:
      # only used at inference time for excited states
      result = result[0], result[1] + params['state_scale']
    return result

  return networks.Network(
      options=options,
      init=network_init,
      apply=network_apply,
      orbitals=orbitals_apply,
  )



def make_boson_net_sum_momentum_projected(
    nspins: Tuple[int, ...],
    charges: jnp.ndarray,
    *,
    ndim: int = 3,
    determinants: int = 16,
    states: int = 0,
    envelope: Optional[envelopes.Envelope] = None,
    feature_layer: Optional[networks.FeatureLayer] = None,
    jastrow: Union[str, jastrows.JastrowType] = jastrows.JastrowType.SIMPLE_EE,
    complex_output: bool = False,
    bias_orbitals: bool = False,
    rescale_inputs: bool = False,
    momind : int,
    mom_kwargs : dict,
    # Psiformer-specific kwargs below.
    num_layers: int,
    num_heads: int,
    heads_dim: int,
    mlp_hidden_dims: Tuple[int, ...],
    use_layer_norm: bool
) -> networks.Network:
  """Psiformer with stacked Self Attention layers.

  Includes standard envelope and determinants.

  Args:
    nspins: Tuple of the number of spin-up and spin-down electrons.
    charges: (natom) array of atom nuclear charges.
    ndim: Dimension of the system. Change only with caution.
    determinants: Number of determinants.
    states: Number of outputs, one per excited (or ground) state. Ignored if 0.
    envelope: Envelope to use to impose orbitals go to zero at infinity.
    feature_layer: Input feature construction.
    jastrow: Type of Jastrow factor if used, or 'simple_ee' if 'default'.
    complex_output: If true, the wavefunction output is complex-valued.
    bias_orbitals: If true, include a bias in the final linear layer to shape
      the outputs into orbitals.
    rescale_inputs: If true, rescale the inputs so they grow as log(|r|).
    num_layers: Number of stacked self-attention layers.
    num_heads: Number of self-attention heads.
    heads_dim: Embedding dimension per-head for self-attention.
    mlp_hidden_dims: Tuple of hidden dimensions of the MLP.
    use_layer_norm: If true, use layer_norm on both attention and MLP.

  Returns:
    Network object containing init, apply, orbitals, options, where init and
    apply are callables which initialise the network parameters and apply the
    network respectively, orbitals is a callable which applies the network up to
    the orbitals, and options specifies the settings used in the network.
  """

  if not envelope:
    envelope = envelopes.make_isotropic_envelope()

  if not feature_layer:
    natoms = charges.shape[0]
    feature_layer = networks.make_ferminet_features(
        natoms, nspins, ndim=ndim, rescale_inputs=rescale_inputs
    )

  if isinstance(jastrow, str):
    if jastrow.upper() == 'DEFAULT':
      jastrow = jastrows.JastrowType.SIMPLE_EE
    else:
      jastrow = jastrows.JastrowType[jastrow.upper()]

  options = PsiformerOptions(
      ndim=ndim,
      determinants=determinants,
      states=states,
      envelope=envelope,
      feature_layer=feature_layer,
      jastrow=jastrow,
      complex_output=complex_output,
      bias_orbitals=bias_orbitals,
      full_det=True,  # Required for Psiformer.
      rescale_inputs=rescale_inputs,
      num_layers=num_layers,
      num_heads=num_heads,
      heads_dim=heads_dim,
      mlp_hidden_dims=mlp_hidden_dims,
      use_layer_norm=use_layer_norm,

  )  # pytype: disable=wrong-keyword-args

  psiformer_layers = make_psiformer_layers(nspins, charges.shape[0], options)

  orbitals_init, orbitals_apply = networks.make_orbitals_bosons_sum(
      nspins=nspins,
      charges=charges,
      options=options,
      equivariant_layers=psiformer_layers,
  )

  def network_init(key: chex.PRNGKey) -> networks.ParamTree:
    return orbitals_init(key)

  abs_lattice = mom_kwargs['abs_lattice']
  unit_cell_vectors = mom_kwargs['unit_cell_vectors']
  # Compute lattice and reciprocal lattice vectors
  lattice = targetmom.lattice_vecs(unit_cell_vectors[0], unit_cell_vectors[1], abs_lattice)
  reciprocal = targetmom.reciprocal_vecs(unit_cell_vectors[0], unit_cell_vectors[1], np.array([[1,0], [0, 1]]))
  g1, g2, _, _, _ = targetmom.g1g2(abs_lattice, reciprocal[:,0], reciprocal[:,1])
  klabels = targetmom.kpoints(abs_lattice)

  # Find all translations in the supercell
  translations, _ = targetmom.find_all_translations_in_supercell(lattice, unit_cell_vectors)
  num_translation = translations.shape[0]
  mn_result = jnp.array(targetmom.mn(klabels[momind], g1, g2))

  # Compute the phase shift for all translations at once
  phase_shifts = jnp.dot(translations, mn_result)  # Shape: (num_translations,)
  # Reshape phase_shifts to align with orbital_outputs
  #phase_shifts = jnp.array(phase_shifts).reshape(-1, 1)  # Shape: (num_translations, 1)
  #phase_shifts = jnp.reshape(phase_shifts, (num_translation, 1, 1, 1))
  def network_apply(
      params,
      pos: jnp.ndarray,
      spins: jnp.ndarray,
      atoms: jnp.ndarray,
      charges: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Forward evaluation of the Psiformer.

    Args:
      params: network parameter tree.
      pos: The electron positions, a 3N dimensional vector.
      spins: The electron spins, an N dimensional vector.
      atoms: Array with positions of atoms.
      charges: Array with nuclear charges.

    Returns:
      Output of antisymmetric neural network in log space, i.e. a tuple of sign
      of and log absolute value of the network evaluated at x.
    """


    # Reshape pos to (N, ndim) for particle-wise operations
    ndim = 2
    N = pos.shape[0] // ndim  # Number of particles (assuming ndim is globally defined)
    reshaped_pos = pos.reshape(N, ndim)

    # Tile translations to match the positions
    tiled_translations = jnp.repeat(translations[:, None, :], N, axis=1)  # Shape: (num_translations, N, ndim)
    translated_positions = reshaped_pos[None, :, :] + tiled_translations  # Shape: (num_translations, N, ndim)

    # Flatten translated_positions back to (num_translations, N * ndim)
    flattened_translated_positions = translated_positions.reshape(translated_positions.shape[0], -1)

    # Compute the orbitals for all translations
    #orbital_outputs = jax.vmap(
    #    lambda t_pos: orbitals_apply(params, t_pos, spins, atoms, charges)[0]
    #)(flattened_translated_positions)  # Shape: (num_translations, N_det)

    # Use jax.vmap to vectorize the application of the orbitals_apply function
    """    orbital_outputs = jax.vmap(lambda t_pos: orbitals_apply(params, t_pos, spins, atoms, charges)[0])(
        flattened_translated_positions
    )"""
    orbital_outputs = []
    for t_pos in flattened_translated_positions:
      orbital_outputs.append(orbitals_apply(params, t_pos, spins, atoms, charges)[0])


    # Convert the result to a JAX array (if not already a JAX array)
    orbital_outputs = jnp.array(orbital_outputs)
    
    assert options.states == 0

    def compute_phase_and_logdet(orbital_outputs: jnp.ndarray, phase_shifts: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
      """
      Compute the phase and log of the determinant by summing over translations using the logsumexp trick.

      Args:
          orbital_outputs: Array of shape (N_trans, N_det, N, N).
          phase_shifts: Array of shape (N_trans,).

      Returns:
          A tuple containing the overall phase and log of the determinant.
      """
      log_determinants = []
      phases = []

      for i in range(orbital_outputs.shape[0]):  # Loop over N_trans
          # Apply logdet_matmul to get phase and logdet
          phase, logdet = network_blocks.logdet_matmul_bosons([orbital_outputs[i]])

          # Add the phase shift to the phase
          phase += phase_shifts[i]

          # Store the logdet and phase
          log_determinants.append(logdet)
          phases.append(phase)


      # Convert to JAX arrays
      log_determinants = jnp.array(log_determinants)
      phases = jnp.array(phases)

      # Use the logsumexp trick for numerical stability
      max_logdet = jnp.max(log_determinants)
      shifted_exps = jnp.exp(log_determinants - max_logdet + 1j * phases)  # Include phase in the exponential
      total_determinant = jnp.sum(shifted_exps)
      total_determinant /= orbital_outputs.shape[0]


      # Compute the overall log and phase
      overall_log = max_logdet + jnp.log(jnp.abs(total_determinant))
      overall_phase = jnp.angle(total_determinant)

      return overall_phase, overall_log
  

    # Perform element-wise multiplication and summation for each translation
    #projected_orbitals = sum(
    #[orbital_output * phase for orbital_output, phase in zip(orbital_outputs, phase_shifts)])
    #projected_orbitals = sum(orbi)
    
    overall_phase, overall_log = compute_phase_and_logdet(orbital_outputs, phase_shifts)

    return overall_phase, overall_log

  return networks.Network(
      options=options,
      init=network_init,
      apply=network_apply,
      orbitals=orbitals_apply,
  )

def make_boson_net_prod_momentum_projected(
    nspins: Tuple[int, ...],
    charges: jnp.ndarray,
    *,
    ndim: int = 3,
    determinants: int = 16,
    states: int = 0,
    envelope: Optional[envelopes.Envelope] = None,
    feature_layer: Optional[networks.FeatureLayer] = None,
    jastrow: Union[str, jastrows.JastrowType] = jastrows.JastrowType.SIMPLE_EE,
    complex_output: bool = False,
    bias_orbitals: bool = False,
    rescale_inputs: bool = False,
    momind : int,
    mom_kwargs : dict,
    # Psiformer-specific kwargs below.
    num_layers: int,
    num_heads: int,
    heads_dim: int,
    mlp_hidden_dims: Tuple[int, ...],
    use_layer_norm: bool
) -> networks.Network:
  """Psiformer with stacked Self Attention layers.

  Includes standard envelope and determinants.

  Args:
    nspins: Tuple of the number of spin-up and spin-down electrons.
    charges: (natom) array of atom nuclear charges.
    ndim: Dimension of the system. Change only with caution.
    determinants: Number of determinants.
    states: Number of outputs, one per excited (or ground) state. Ignored if 0.
    envelope: Envelope to use to impose orbitals go to zero at infinity.
    feature_layer: Input feature construction.
    jastrow: Type of Jastrow factor if used, or 'simple_ee' if 'default'.
    complex_output: If true, the wavefunction output is complex-valued.
    bias_orbitals: If true, include a bias in the final linear layer to shape
      the outputs into orbitals.
    rescale_inputs: If true, rescale the inputs so they grow as log(|r|).
    num_layers: Number of stacked self-attention layers.
    num_heads: Number of self-attention heads.
    heads_dim: Embedding dimension per-head for self-attention.
    mlp_hidden_dims: Tuple of hidden dimensions of the MLP.
    use_layer_norm: If true, use layer_norm on both attention and MLP.

  Returns:
    Network object containing init, apply, orbitals, options, where init and
    apply are callables which initialise the network parameters and apply the
    network respectively, orbitals is a callable which applies the network up to
    the orbitals, and options specifies the settings used in the network.
  """

  if not envelope:
    envelope = envelopes.make_isotropic_envelope()

  if not feature_layer:
    natoms = charges.shape[0]
    feature_layer = networks.make_ferminet_features(
        natoms, nspins, ndim=ndim, rescale_inputs=rescale_inputs
    )

  if isinstance(jastrow, str):
    if jastrow.upper() == 'DEFAULT':
      jastrow = jastrows.JastrowType.SIMPLE_EE
    else:
      jastrow = jastrows.JastrowType[jastrow.upper()]

  options = PsiformerOptions(
      ndim=ndim,
      determinants=determinants,
      states=states,
      envelope=envelope,
      feature_layer=feature_layer,
      jastrow=jastrow,
      complex_output=complex_output,
      bias_orbitals=bias_orbitals,
      full_det=True,  # Required for Psiformer.
      rescale_inputs=rescale_inputs,
      num_layers=num_layers,
      num_heads=num_heads,
      heads_dim=heads_dim,
      mlp_hidden_dims=mlp_hidden_dims,
      use_layer_norm=use_layer_norm,

  )  # pytype: disable=wrong-keyword-args

  psiformer_layers = make_psiformer_layers(nspins, charges.shape[0], options)

  orbitals_init, orbitals_apply = networks.make_orbitals_bosons_prod(
      nspins=nspins,
      charges=charges,
      options=options,
      equivariant_layers=psiformer_layers,
  )

  def network_init(key: chex.PRNGKey) -> networks.ParamTree:
    return orbitals_init(key)

  abs_lattice = mom_kwargs['abs_lattice']
  unit_cell_vectors = mom_kwargs['unit_cell_vectors']
  # Compute lattice and reciprocal lattice vectors
  lattice = targetmom.lattice_vecs(unit_cell_vectors[0], unit_cell_vectors[1], abs_lattice)
  reciprocal = targetmom.reciprocal_vecs(unit_cell_vectors[0], unit_cell_vectors[1], np.array([[1,0], [0, 1]]))
  g1, g2, _, _, _ = targetmom.g1g2(abs_lattice, reciprocal[:,0], reciprocal[:,1])
  klabels = targetmom.kpoints(abs_lattice)

  # Find all translations in the supercell
  translations, _ = targetmom.find_all_translations_in_supercell(lattice, unit_cell_vectors)
  num_translation = translations.shape[0]
  mn_result = jnp.array(targetmom.mn(klabels[momind], g1, g2))

  # Compute the phase shift for all translations at once
  phase_shifts = jnp.dot(translations, mn_result)  # Shape: (num_translations,)
  # Reshape phase_shifts to align with orbital_outputs
  #phase_shifts = jnp.array(phase_shifts).reshape(-1, 1)  # Shape: (num_translations, 1)
  #phase_shifts = jnp.reshape(phase_shifts, (num_translation, 1, 1, 1))
  def network_apply(
      params,
      pos: jnp.ndarray,
      spins: jnp.ndarray,
      atoms: jnp.ndarray,
      charges: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Forward evaluation of the Psiformer.

    Args:
      params: network parameter tree.
      pos: The electron positions, a 3N dimensional vector.
      spins: The electron spins, an N dimensional vector.
      atoms: Array with positions of atoms.
      charges: Array with nuclear charges.

    Returns:
      Output of antisymmetric neural network in log space, i.e. a tuple of sign
      of and log absolute value of the network evaluated at x.
    """


    # Reshape pos to (N, ndim) for particle-wise operations
    ndim = 2
    N = pos.shape[0] // ndim  # Number of particles (assuming ndim is globally defined)
    reshaped_pos = pos.reshape(N, ndim)

    # Tile translations to match the positions
    tiled_translations = jnp.repeat(translations[:, None, :], N, axis=1)  # Shape: (num_translations, N, ndim)
    translated_positions = reshaped_pos[None, :, :] + tiled_translations  # Shape: (num_translations, N, ndim)

    # Flatten translated_positions back to (num_translations, N * ndim)
    flattened_translated_positions = translated_positions.reshape(translated_positions.shape[0], -1)

    # Compute the orbitals for all translations
    #orbital_outputs = jax.vmap(
    #    lambda t_pos: orbitals_apply(params, t_pos, spins, atoms, charges)[0]
    #)(flattened_translated_positions)  # Shape: (num_translations, N_det)

    # Use jax.vmap to vectorize the application of the orbitals_apply function
    """    orbital_outputs = jax.vmap(lambda t_pos: orbitals_apply(params, t_pos, spins, atoms, charges)[0])(
        flattened_translated_positions
    )"""
    orbital_outputs = []
    for t_pos in flattened_translated_positions:
      orbital_outputs.append(orbitals_apply(params, t_pos, spins, atoms, charges)[0])


    # Convert the result to a JAX array (if not already a JAX array)
    orbital_outputs = jnp.array(orbital_outputs)
    
    assert options.states == 0

    def compute_phase_and_logdet(orbital_outputs: jnp.ndarray, phase_shifts: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
      """
      Compute the phase and log of the determinant by summing over translations using the logsumexp trick.

      Args:
          orbital_outputs: Array of shape (N_trans, N_det, N, N).
          phase_shifts: Array of shape (N_trans,).

      Returns:
          A tuple containing the overall phase and log of the determinant.
      """
      log_determinants = []
      phases = []

      for i in range(orbital_outputs.shape[0]):  # Loop over N_trans
          # Apply logdet_matmul to get phase and logdet
          phase, logdet = network_blocks.logdet_matmul_bosons([orbital_outputs[i]])

          # Add the phase shift to the phase
          phase += phase_shifts[i]

          # Store the logdet and phase
          log_determinants.append(logdet)
          phases.append(phase)


      # Convert to JAX arrays
      log_determinants = jnp.array(log_determinants)
      phases = jnp.array(phases)

      # Use the logsumexp trick for numerical stability
      max_logdet = jnp.max(log_determinants)
      shifted_exps = jnp.exp(log_determinants - max_logdet + 1j * phases)  # Include phase in the exponential
      total_determinant = jnp.sum(shifted_exps)
      total_determinant /= orbital_outputs.shape[0]


      # Compute the overall log and phase
      overall_log = max_logdet + jnp.log(jnp.abs(total_determinant))
      overall_phase = jnp.angle(total_determinant)

      return overall_phase, overall_log
  

    # Perform element-wise multiplication and summation for each translation
    #projected_orbitals = sum(
    #[orbital_output * phase for orbital_output, phase in zip(orbital_outputs, phase_shifts)])
    #projected_orbitals = sum(orbi)
    
    overall_phase, overall_log = compute_phase_and_logdet(orbital_outputs, phase_shifts)

    return overall_phase, overall_log

  return networks.Network(
      options=options,
      init=network_init,
      apply=network_apply,
      orbitals=orbitals_apply,
  )

################### Magnetic-field #################
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

def log_wsigma_prod(z: jnp.ndarray, w1: jnp.ndarray, w2: jnp.ndarray, M: int = 11) -> jnp.ndarray:
    """
    Complex log of σ̂ using the finite product over lattice generated by (w1,w2),
    where w1,w2 are HALF-periods (ω1,ω2). Vectorized over z. Avoids boolean indexing.
    """
    z = jnp.asarray(z, dtype=jnp.complex128)
    zf = z.reshape(-1)  # (P,)

    # Build integer grids
    ms = jnp.arange(-M, M+1, dtype=jnp.int32)
    ns = jnp.arange(-M, M+1, dtype=jnp.int32)
    Mgrid, Ngrid = jnp.meshgrid(ms, ns, indexing="ij")            # (2M+1, 2M+1)

    # Full periods from half-periods: L = 2 m w1 + 2 n w2
    L_full = (2.0*Mgrid.astype(jnp.complex128) * w1 +
              2.0*Ngrid.astype(jnp.complex128) * w2)              # (2M+1, 2M+1)

    # Exclude (m,n)=(0,0) WITHOUT boolean slicing
    mask = jnp.logical_not(jnp.logical_and(Mgrid == 0, Ngrid == 0))
    K = (2*M + 1)*(2*M + 1) - 1                                   # static
    flat_idx = jnp.nonzero(mask.reshape(-1), size=K, fill_value=False)[0]  # (K,)
    L = L_full.reshape(-1)[flat_idx]                               # (K,)

    # Product in log space
    t = zf[:, None] / L[None, :]                                   # (P,K)
    one_minus = 1.0 - t
    s = jnp.sum(jnp.log(one_minus + 0j) + (t + 0.5*t*t), axis=1)   # (P,)
    out = jnp.log(zf + 1e-300) + s
    return out.reshape(z.shape)

def wsigma_prod(z: jnp.ndarray, w1: jnp.ndarray, w2: jnp.ndarray, M: int = 11) -> jnp.ndarray:
    return jnp.exp(log_wsigma_prod(z, w1, w2, M=M))

    

def LLL_single(z: jnp.ndarray, N_phi: jnp.ndarray, L1: jnp.ndarray, L2: jnp.ndarray,
               almost: jnp.ndarray, M: int = 11) -> jnp.ndarray:
    """
    σ̂(z; ω=L/2)*exp(-|z|^2/(2Nφ))*exp(-almost z^2/2)
    where ω1=L1_com/2, ω2=L2_com/2 and L*_com = (Lx+iLy)/√2.
    """
    L1com = to_cplx_divsqrt2(L1); L2com = to_cplx_divsqrt2(L2)
    w1, w2 = L1com/2.0, L2com/2.0
    sig = wsigma_prod(z, w1, w2, M=M)
    gauss = jnp.exp(- (z*jnp.conj(z)).real / (2.0*N_phi))
    quad  = jnp.exp(-0.5 * almost * (z*z))
    return sig * gauss * quad

def LLL_single_log(z: jnp.ndarray, N_phi: jnp.ndarray, L1: jnp.ndarray, L2: jnp.ndarray,
                   almost: jnp.ndarray, M: int = 11) -> jnp.ndarray:
    """
    Compute the log of σ̂(z; ω=L/2)*exp(-|z|^2/(2Nφ))*exp(-almost z^2/2)
    in log space.
    """
    L1com = to_cplx_divsqrt2(L1); L2com = to_cplx_divsqrt2(L2)
    w1, w2 = L1com / 2.0, L2com / 2.0

    # Compute log(σ̂(z; ω))
    log_sig = log_wsigma_prod(z, w1, w2, M=M)

    # Compute log of the Gaussian term
    log_gauss = -(z * jnp.conj(z)).real / (2.0 * N_phi)

    # Compute log of the quadratic term
    log_quad = -0.5 * almost * (z * z)

    # Combine all terms in log space
    log_result = log_sig + log_gauss + log_quad

    return log_result

def LLL_with_zeros(z: jnp.ndarray, N_phi: jnp.ndarray, L1: jnp.ndarray, L2: jnp.ndarray,
                   almost: jnp.ndarray, zeros: jnp.ndarray, M: int = 100) -> jnp.ndarray:
    """
    ∏_a σ̂(z-a)*exp((conj(a)z - conj(z)a)/(2Nφ))*exp(-|z|^2/(2Nφ))*exp(-almost z^2/2)
    """
    L1com = to_cplx_divsqrt2(L1); L2com = to_cplx_divsqrt2(L2)
    w1, w2 = L1com/2.0, L2com/2.0
    z = jnp.asarray(z, dtype=jnp.complex128)
    zeros = jnp.asarray(zeros, dtype=jnp.complex128)

    # Σ_a log σ̂(z - a)
    #log_sig = jnp.sum(log_wsigma_prod(z - zeros[:, None], w1, w2, M=M), axis=0)
    sig = jnp.prod(wsigma_prod(z - zeros[:, None], w1, w2, M=M), axis=0)
    gauge = jnp.exp((jnp.sum(jnp.conj(zeros))*z - jnp.conj(z)*jnp.sum(zeros)) / (2.0*N_phi))
    gauss = jnp.exp(- (z*jnp.conj(z)).real / (2.0*N_phi))
    quad  = jnp.exp( -0.5 * almost * (z*z))

    return sig * gauge * gauss * quad

# def LLL_with_zeros_log(z: jnp.ndarray, N_phi: jnp.ndarray, L1: jnp.ndarray, L2: jnp.ndarray,
#                        almost: jnp.ndarray, zeros: jnp.ndarray, M: int = 100) -> jnp.ndarray:
#     """
#     Compute the output in log space:
#     log(∏_a σ̂(z-a)*exp((conj(a)z - conj(z)a)/(2Nφ))*exp(-|z|^2/(2Nφ))*exp(-almost z^2/2))
#     """
#     L1com = to_cplx_divsqrt2(L1); L2com = to_cplx_divsqrt2(L2)
#     w1, w2 = L1com / 2.0, L2com / 2.0

#     # Compute log(σ̂(z - a)) and sum over all zeros
#     #log_sig = jnp.sum(ellipticfunctions.log_wsigma(z - zeros[:, None], omega = (w1,w2)), axis=0)
#     log_sig = jnp.sum(jnp.array([ellipticfunctions.log_weierstrass_sigma(z - a, w1,w2) for a in zeros]))

#     # Compute the log of the gauge term
#     log_gauge = (jnp.sum(jnp.conj(zeros)) * z - jnp.conj(z) * jnp.sum(zeros)) / (2.0 * N_phi)

#     # Compute the log of the Gaussian term
#     log_gauss = -(z * jnp.conj(z)).real / (2.0 * N_phi)

#     # Compute the log of the quadratic term
#     log_quad = -0.5 * almost * (z * z)

#     # Combine all terms in log space
#     log_result = log_sig + log_gauge + log_gauss + log_quad
#     # re = jnp.real(log_result)
#     # m  = jax.lax.stop_gradient(jnp.mean(re))
#     # re_c = re - m
#     # log_result = re_c + 1j * jnp.imag(log_result)
#     return log_result 

def LLL_with_zeros_log(z, N_phi, L1, L2, almost, zeros, M: int = 100):
    """
    log(∏_a σ(z-a)) + [(Σ conj(a)) z - conj(z) (Σ a)]/(2 Nφ)
      - n_z * |z|^2/(2 Nφ) - (n_z/2) * almost * z^2
    where n_z = number of zeros
    """
    L1com = to_cplx_divsqrt2(L1); L2com = to_cplx_divsqrt2(L2)
    w1, w2 = L1com / 2.0, L2com / 2.0

    nz    = zeros.size

    # sum over zeros of log σ(z-a)
    log_sig = jnp.sum(jnp.array(
        [ellipticfunctions.log_weierstrass_sigma(z - a, w1, w2) for a in zeros]
    ))

    # gauge term (sums once over zeros)
    A = jnp.sum(zeros)
    log_gauge = (jnp.conj(A) * z - jnp.conj(z) * A) / (2.0 * N_phi)

    # Gaussian and quadratic appear once per zero in your Julia loop
    log_gauss = - nz * (z * jnp.conj(z)).real / (2.0 * N_phi)
    log_quad  = - 0.5 * nz * almost * (z * z)
    def safe_log(zlog, lo=-80.0, hi=80.0):
      re = jnp.clip(jnp.real(zlog), a_min=lo, a_max=hi)
      return re + 1j*jnp.imag(zlog)

    log_result = log_sig + log_gauge + log_gauss + log_quad
    log_result = safe_log(log_result)                 # keeps fp32 stable
    return log_result.astype(jnp.complex64)



def make_fermi_net_magfield(
  nspins: Tuple[int, ...],
  charges: jnp.ndarray,
  *,
  ndim: int = 2,
  determinants: int = 16,
  states: int = 0,
  envelope: Optional[envelopes.Envelope] = None,
  feature_layer: Optional[networks.FeatureLayer] = None,
  jastrow: Union[str, jastrows.JastrowType] = jastrows.JastrowType.SIMPLE_EE,
  complex_output: bool = False,
  bias_orbitals: bool = False,
  rescale_inputs: bool = False,
  magfield_kwargs: dict,
  # Psiformer-specific kwargs below.
  num_layers: int,
  num_heads: int,
  heads_dim: int,
  mlp_hidden_dims: Tuple[int, ...],
  use_layer_norm: bool,
) -> networks.Network:
  """Psiformer with stacked Self Attention layers.

  Includes standard envelope and determinants.

  Args:
    nspins: Tuple of the number of spin-up and spin-down electrons.
    charges: (natom) array of atom nuclear charges.
    ndim: Dimension of the system. Change only with caution.
    determinants: Number of determinants.
    states: Number of outputs, one per excited (or ground) state. Ignored if 0.
    envelope: Envelope to use to impose orbitals go to zero at infinity.
    feature_layer: Input feature construction.
    jastrow: Type of Jastrow factor if used, or 'simple_ee' if 'default'.
    complex_output: If true, the wavefunction output is complex-valued.
    bias_orbitals: If true, include a bias in the final linear layer to shape
      the outputs into orbitals.
    rescale_inputs: If true, rescale the inputs so they grow as log(|r|).
    num_layers: Number of stacked self-attention layers.
    num_heads: Number of self-attention heads.
    heads_dim: Embedding dimension per-head for self-attention.
    mlp_hidden_dims: Tuple of hidden dimensions of the MLP.
    use_layer_norm: If true, use layer_norm on both attention and MLP.

  Returns:
    Network object containing init, apply, orbitals, options, where init and
    apply are callables which initialise the network parameters and apply the
    network respectively, orbitals is a callable which applies the network up to
    the orbitals, and options specifies the settings used in the network.
  """

  if not envelope:
    envelope = envelopes.make_isotropic_envelope()

  if not feature_layer:
    natoms = charges.shape[0]
    feature_layer = networks.make_ferminet_features(
        natoms, nspins, ndim=ndim, rescale_inputs=rescale_inputs
    )

  if isinstance(jastrow, str):
    if jastrow.upper() == 'DEFAULT':
      jastrow = jastrows.JastrowType.SIMPLE_EE
    else:
      jastrow = jastrows.JastrowType[jastrow.upper()]

  options = PsiformerOptions(
      ndim=ndim,
      determinants=determinants,
      states=states,
      envelope=envelope,
      feature_layer=feature_layer,
      jastrow=jastrow,
      complex_output=complex_output,
      bias_orbitals=bias_orbitals,
      full_det=True,  # Required for Psiformer.
      rescale_inputs=rescale_inputs,
      num_layers=num_layers,
      num_heads=num_heads,
      heads_dim=heads_dim,
      mlp_hidden_dims=mlp_hidden_dims,
      use_layer_norm=use_layer_norm,
  )  # pytype: disable=wrong-keyword-args

  psiformer_layers = make_psiformer_layers(nspins, charges.shape[0], options)

  orbitals_init, orbitals_apply = networks.make_orbitals(
      nspins=nspins,
      charges=charges,
      options=options,
      equivariant_layers=psiformer_layers,
  )

  def network_init(key: chex.PRNGKey) -> networks.ParamTree:
    return orbitals_init(key)

  def modify_orbitals_matrix(pos, L1, L2, N_phi,almost, zeros,rescale, M=300 ,nmax=300):
    """
    Modify the orbitals[0] matrix such that its determinant is multiplied
    by the symmetric product of LLL_single(z_i) over all particles.

    Parameters:
        orbitals (list of jnp.ndarray): List of matrices, where orbitals[0] is (N, N).
        pos (jnp.ndarray): 2N-dimensional vector of shape (N*2,).
        L1 (jnp.ndarray): L1 vector for the LLL_single function.
        L2 (jnp.ndarray): L2 vector for the LLL_single function.
        N_phi (jnp.ndarray): N_phi parameter for the LLL_single function.
        almost (jnp.ndarray): Almost modular parameter for the LLL_single function.
        M (int): Parameter for wsigma_prod in LLL_single.
        nmax (int): Parameter for eisenstein in LLL_single.

    Returns:
        jnp.ndarray: Modified orbitals[0] matrix.
    """
    # Number of particles
    N = pos.shape[0] // 2

    # Reshape pos into (N, 2) and convert to complex numbers using to_cplx_divsqrt2
    positions = pos.reshape(N, 2)
    z = jnp.array([to_cplx_divsqrt2(p) for p in positions])

    # Compute LLL_single values for each z
    lll_values = jnp.array([LLL_with_zeros_log(z_i, N_phi, L1, L2, almost, zeros, M=M) for z_i in z]).squeeze()
    #lll_values2 = jnp.array([LLL_single(z_i, N_phi, L1, L2, almost, M=M) for z_i in z])

    return jnp.sum(lll_values) + jnp.log(1/rescale)


  def network_apply(
      params,
      pos: jnp.ndarray,
      spins: jnp.ndarray,
      atoms: jnp.ndarray,
      charges: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Forward evaluation of the Psiformer.

    Args:
      params: network parameter tree.
      pos: The electron positions, a 2N dimensional vector.
      spins: The electron spins, an N dimensional vector.
      atoms: Array with positions of atoms.
      charges: Array with nuclear charges.

    Returns:
      Output of antisymmetric neural network in log space, i.e. a tuple of sign
      of and log absolute value of the network evaluated at x.
    """
    almost = almost_modular(magfield_kwargs['lattice'][:,0], magfield_kwargs['lattice'][:,1], magfield_kwargs["N_phi"], nmax=300)
    print(almost)
    orbitals = orbitals_apply(params, pos, spins, atoms, charges)
    lll_values = modify_orbitals_matrix(pos, magfield_kwargs['lattice'][:,0], magfield_kwargs['lattice'][:,1], magfield_kwargs["N_phi"], almost,magfield_kwargs['zeros'],magfield_kwargs['rescale'])
    if options.states:
      batch_logdet_matmul = jax.vmap(network_blocks.logdet_matmul, in_axes=0)
      orbitals = [
          jnp.reshape(orbital, (options.states, -1) + orbital.shape[1:])
          for orbital in orbitals
      ]
      result = batch_logdet_matmul(orbitals)
    else:
      result = network_blocks.logdet_matmul(orbitals)
      lll_phase = jnp.imag(lll_values)  # Phase of the complex number
      lll_log_abs = jnp.real(lll_values)  # Log of the absolute value

      # Add these components to the result
      result_phase = result[0] + lll_phase
      result_log_abs = result[1] + lll_log_abs

      # Update result with the new values
      result = (result_phase, result_log_abs)
    if 'state_scale' in params:
      # only used at inference time for excited states
      result = result[0], result[1] + params['state_scale']
    return result

  return networks.Network(
      options=options,
      init=network_init,
      apply=network_apply,
      orbitals=orbitals_apply,
  )


# ============================================================
#  Stable σ̂-based LLL torus utilities + Psiformer wrapper
#  (antisymmetry-safe for complex_output True/False)
# ============================================================


# ============================================================
#  σ̂-based LLL torus utilities + Psiformer wrapper (quasi-periodic)
#  - preserves quasi-periodicity using Gaussian + "almost" quadratic
#  - NO forced zero-centering; optional sector anchor supported
#  - antisymmetry-safe combination for complex_output True/False
# ============================================================

# ============================================================
#  σ̂-based LLL torus utilities + Psiformer wrapper (quasi-periodic)
#  - symmetric-gauge magnetic boundary condition preserved
#  - Gaussian + "almost" quadratic included
#  - NO wrapping of positions (keep magnetic phases)
#  - NO per-config centering (preserve exact quasi-periodicity)
#  - antisymmetry-safe for complex_output True/False
# ============================================================

# from typing import Optional, Tuple, Union
# import jax
# import jax.numpy as jnp
# import chex

# # ------------------------------------------------------------
# # Utilities
# # ------------------------------------------------------------

# def to_cplx_divsqrt2(v2: jnp.ndarray) -> jnp.ndarray:
#     """(x,y) → (x+iy)/√2 (LLL complex convention)."""
#     return (v2[0] + 1j * v2[1]) / jnp.sqrt(2.0)

# def eisenstein(L1com: jnp.ndarray, L2com: jnp.ndarray, nmax: int = 200) -> jnp.ndarray:
#     """
#     G2(ω1, ω2) ≈ (2π^2/ω1^2) * (1/6 + Σ_{n=1..nmax} 1/sin^2(nπ τ)), τ=ω2/ω1.
#     L1com, L2com are FULL periods in our complex convention; τ = L2com/L1com.
#     """
#     ratio = L2com / L1com
#     n = jnp.arange(1, nmax + 1, dtype=jnp.float64)
#     s = jnp.sum(1.0 / (jnp.sin(jnp.pi * n * ratio) ** 2))
#     return (2.0 * (jnp.pi ** 2) / (L1com * L1com)) * (1.0 / 6.0 + s)

# def almost_modular(L1: jnp.ndarray, L2: jnp.ndarray, N_phi: jnp.ndarray, nmax: int = 200) -> jnp.ndarray:
#     """
#     almost = G2(L1com,L2com) - (conj(L1com)/L1com)/N_phi, with Li_com=(Lix+iLiy)/√2.
#     This quadratic compensator is REQUIRED with σ̂ to get the correct quasi-periodicity.
#     """
#     L1com = to_cplx_divsqrt2(L1).astype(jnp.complex128)
#     L2com = to_cplx_divsqrt2(L2).astype(jnp.complex128)
#     G2 = eisenstein(L1com, L2com, nmax=nmax)
#     return G2 - (jnp.conj(L1com) / (N_phi * L1com))

# def adjust_zeros_to_anchor(zeros: jnp.ndarray, A_target: Optional[jnp.ndarray]) -> jnp.ndarray:
#     """
#     Shift the zero set so that sum(zeros) == A_target (mod lattice). If A_target is None,
#     leave zeros as provided (NO centering!). This preserves the chosen quasi-periodic sector.
#     """
#     zeros = jnp.asarray(zeros, dtype=jnp.complex128)
#     if A_target is None:
#         return zeros
#     A = jnp.sum(zeros)
#     shift = (A_target - A) / zeros.shape[0]
#     return zeros + shift

# # ------------------------------------------------------------
# # Renormalized Weierstrass σ̂ (log-space product; numerically stable)
# # ------------------------------------------------------------

# def _pairwise_sum(x: jnp.ndarray) -> jnp.ndarray:
#     """Numerically friendlier tree reduction (pairwise sum)."""
#     x = x.reshape(-1)
#     while x.shape[0] > 1:
#         even = x[0::2]
#         odd  = x[1::2]
#         if odd.shape[0] < even.shape[0]:
#             odd = jnp.pad(odd, (0, even.shape[0] - odd.shape[0]))
#         x = even + odd
#     return x[0]

# def log_wsigma_prod(z: jnp.ndarray, w1: jnp.ndarray, w2: jnp.ndarray, M: int = 11) -> jnp.ndarray:
#     """
#     Complex log of renormalized σ̂ using a finite Weierstrass product over the lattice
#     generated by (w1,w2), where w1,w2 are HALF-periods (ω1, ω2).
#     Renormalized formula:
#       log σ̂(zz) = log(zz) + Σ_{L≠0} [ log1p(-zz/L) + zz/L + (zz/L)^2/2 ].
#     Uses log1p and pairwise reduction for stability. Vectorized over z (any shape).
#     """
#     z = jnp.asarray(z, dtype=jnp.complex128)
#     zf = z.reshape(-1)  # (P,)

#     # Integer grid (2M+1)^2 points minus (0,0)
#     ms = jnp.arange(-M, M + 1, dtype=jnp.int32)
#     ns = jnp.arange(-M, M + 1, dtype=jnp.int32)
#     Mgrid, Ngrid = jnp.meshgrid(ms, ns, indexing="ij")

#     L_full = (2.0 * Mgrid.astype(jnp.complex128) * w1 +
#               2.0 * Ngrid.astype(jnp.complex128) * w2).reshape(-1)
#     mask = jnp.logical_not(jnp.logical_and(Mgrid == 0, Ngrid == 0)).reshape(-1)
#     L = L_full[mask]  # (K,)

#     def log_sigma_single(zz):
#         t = zz / L
#         terms = jnp.log1p(-t) + t + 0.5 * (t * t)
#         s = _pairwise_sum(terms)
#         return jnp.log(zz + 1e-300) + s  # tiny to avoid branch at 0

#     out = jax.vmap(log_sigma_single)(zf)
#     return out.reshape(z.shape)

# def wsigma_prod(z: jnp.ndarray, w1: jnp.ndarray, w2: jnp.ndarray, M: int = 11) -> jnp.ndarray:
#     return jnp.exp(log_wsigma_prod(z, w1, w2, M=M))

# # ------------------------------------------------------------
# # LLL building blocks (WITH Gaussian and WITH almost-quadratic)
# # ------------------------------------------------------------

# def L1L2_to_half_periods(L1: jnp.ndarray, L2: jnp.ndarray):
#     L1com = to_cplx_divsqrt2(L1).astype(jnp.complex128)
#     L2com = to_cplx_divsqrt2(L2).astype(jnp.complex128)
#     return L1com / 2.0, L2com / 2.0, L1com, L2com

# def LLL_single_log(z: jnp.ndarray, N_phi: jnp.ndarray, L1: jnp.ndarray, L2: jnp.ndarray,
#                    almost: jnp.ndarray, M: int = 11) -> jnp.ndarray:
#     """
#     log σ̂(z; ω=L/2) + [ - |z|^2/(2Nφ) ] + [ - (almost/2) * z^2 ].
#     KEEP the almost-quadratic: it's needed with σ̂ to enforce correct quasi-periodicity.
#     """
#     w1, w2, _, _ = L1L2_to_half_periods(L1, L2)
#     log_sig   = log_wsigma_prod(z, w1, w2, M=M)
#     log_gauss = -(z * jnp.conj(z)).real / (2.0 * N_phi)
#     log_quad  = -0.5 * almost * (z * z)
#     return log_sig + log_gauss + log_quad

# def LLL_with_zeros_log(z: jnp.ndarray, N_phi: jnp.ndarray, L1: jnp.ndarray, L2: jnp.ndarray,
#                        almost: jnp.ndarray, zeros: jnp.ndarray, M: int = 11) -> jnp.ndarray:
#     """
#     log(∏_a σ̂(z-a)) + (conj(A)z - conj(z)A)/(2Nφ) - |z|^2/(2Nφ) - (almost/2) z^2 ,  A=Σa.
#     (Quasi-periodic under z -> z + L1/L2 with the correct magnetic phases.)
#     """
#     w1, w2, _, _ = L1L2_to_half_periods(L1, L2)
#     z     = jnp.asarray(z, dtype=jnp.complex128)
#     zeros = jnp.asarray(zeros, dtype =jnp.complex128)

#     log_sig   = jnp.sum(log_wsigma_prod(z - zeros[:, None], w1, w2, M=M), axis=0)
#     A         = jnp.sum(zeros)
#     log_gauge = (jnp.conj(A) * z - jnp.conj(z) * A) / (2.0 * N_phi)
#     log_gauss = -(z * jnp.conj(z)).real / (2.0 * N_phi)
#     log_quad  = -0.5 * almost * (z * z)

#     return log_sig + log_gauge + log_gauss + log_quad

# # ------------------------------------------------------------
# # Per-configuration LLL log builder
# # (NO wrapping; optional sector anchor for zeros; NO per-config centering)
# # ------------------------------------------------------------

# def _build_lll_log_sum(pos: jnp.ndarray, L1: jnp.ndarray, L2: jnp.ndarray, N_phi: jnp.ndarray,
#                        zeros: jnp.ndarray, almost: jnp.ndarray,
#                        zeros_anchor: Optional[jnp.ndarray],
#                        M: int = 250) -> jnp.ndarray:
#     """
#     Returns a single complex scalar: sum_i log(LLL_with_zeros(z_i)).
#     - DO NOT wrap positions (preserves quasi-periodic phases).
#     - Adjust zeros to the desired sector anchor A_target only if provided.
#     - NO per-configuration centering; preserve exact magnetic boundary law.
#     """
#     N = pos.shape[0] // 2

#     # Respect user's sector (no forced centering)
#     if zeros_anchor is None:
#         zeros_adj = jnp.asarray(zeros, dtype=jnp.complex128)
#     else:
#         A = jnp.sum(zeros)
#         shift = (zeros_anchor - A) / zeros.shape[0]
#         zeros_adj = jnp.asarray(zeros + shift, dtype=jnp.complex128)

#     # Electron positions to complex (NO wrapping)
#     positions = pos.reshape(N, 2)
#     z = jnp.array([to_cplx_divsqrt2(p) for p in positions], dtype=jnp.complex128)

#     # Per-electron complex logs (with Gaussian + almost)
#     lll_vals = jnp.array(
#         [LLL_with_zeros_log(zi, N_phi, L1, L2, almost, zeros_adj, M=M)
#          for zi in z],
#         dtype=jnp.complex128
#     )

#     return jnp.sum(lll_vals)

# # ------------------------------------------------------------
# # Psiformer/FermiNet wrapper
# # ------------------------------------------------------------

# def make_fermi_net_magfield(
#     nspins: Tuple[int, ...],
#     charges: jnp.ndarray,
#     *,
#     ndim: int = 2,
#     determinants: int = 16,
#     states: int = 0,
#     envelope: Optional['envelopes.Envelope'] = None,
#     feature_layer: Optional['networks.FeatureLayer'] = None,
#     jastrow: Union[str, 'jastrows.JastrowType'] = 'SIMPLE_EE',
#     complex_output: bool = True,     # supports False too; see combination below
#     bias_orbitals: bool = False,
#     rescale_inputs: bool = False,
#     magfield_kwargs: dict,
#     # Psiformer-specific kwargs:
#     num_layers: int,
#     num_heads: int,
#     heads_dim: int,
#     mlp_hidden_dims: Tuple[int, ...],
#     use_layer_norm: bool,
# ) -> 'networks.Network':
#   """Psiformer with stacked Self Attention layers + σ̂-based torus LLL factor (quasi-periodic)."""

#   # --- options & layers (your usual plumbing) ---
#   if not envelope:
#       envelope = envelopes.make_isotropic_envelope()

#   if not feature_layer:
#       natoms = charges.shape[0]
#       feature_layer = networks.make_ferminet_features(
#           natoms, nspins, ndim=ndim, rescale_inputs=rescale_inputs
#       )

#   if isinstance(jastrow, str):
#       if jastrow.upper() == 'DEFAULT':
#           jastrow = jastrows.JastrowType.SIMPLE_EE
#       else:
#           jastrow = jastrows.JastrowType[jastrow.upper()]

#   options = PsiformerOptions(
#       ndim=ndim,
#       determinants=determinants,
#       states=states,
#       envelope=envelope,
#       feature_layer=feature_layer,
#       jastrow=jastrow,
#       complex_output=complex_output,
#       bias_orbitals=bias_orbitals,
#       full_det=True,  # Required for Psiformer.
#       rescale_inputs=rescale_inputs,
#       num_layers=num_layers,
#       num_heads=num_heads,
#       heads_dim=heads_dim,
#       mlp_hidden_dims=mlp_hidden_dims,
#       use_layer_norm=use_layer_norm,
#   )  # pytype: disable=wrong-keyword-args

#   psiformer_layers = make_psiformer_layers(nspins, charges.shape[0], options)
#   orbitals_init, orbitals_apply = networks.make_orbitals(
#       nspins=nspins,
#       charges=charges,
#       options=options,
#       equivariant_layers=psiformer_layers,
#   )

#   def network_init(key: chex.PRNGKey) -> 'networks.ParamTree':
#       return orbitals_init(key)

#   def network_apply(
#       params,
#       pos: jnp.ndarray,
#       spins: jnp.ndarray,
#       atoms: jnp.ndarray,
#       charges: jnp.ndarray,
#   ) -> Tuple[jnp.ndarray, jnp.ndarray]:
#       """Forward evaluation in log-space (phase/log|ψ|) or (sign/log|ψ|)."""

#       # Geometry / sector from kwargs
#       L1 = magfield_kwargs['lattice'][:, 0]
#       L2 = magfield_kwargs['lattice'][:, 1]
#       N_phi = magfield_kwargs["N_phi"]
#       zeros = magfield_kwargs['zeros']
#       zeros_anchor = magfield_kwargs.get('zeros_anchor', None)      # complex or None
#       M = magfield_kwargs.get('M', 250)
#       rescale = magfield_kwargs.get('rescale', 1.0)

#       # Compute the almost compensator (REQUIRED)
#       almost = almost_modular(L1, L2, N_phi, nmax=200)

#       # Base orbitals/determinant from Psiformer
#       orbitals = orbitals_apply(params, pos, spins, atoms, charges)

#       # σ̂-based LLL factor — complex log per configuration
#       sum_lll = _build_lll_log_sum(
#           pos, L1, L2, N_phi, zeros, almost, zeros_anchor, M=M
#       ) + jnp.log(1.0 / rescale)

#       # Determinant logs
#       if options.states:
#           batch_logdet_matmul = jax.vmap(network_blocks.logdet_matmul, in_axes=0)
#           orbitals_b = [
#               jnp.reshape(orbital, (options.states, -1) + orbital.shape[1:])
#               for orbital in orbitals
#           ]
#           result = batch_logdet_matmul(orbitals_b)
#       else:
#           result = network_blocks.logdet_matmul(orbitals)

#       # ---- Antisymmetry-safe combination (handles both modes) ----
#       if options.complex_output:
#           # result[0] := phase (real), result[1] := log|det|
#           out_phase   = result[0] + jnp.imag(sum_lll)
#           out_logabs  = result[1] + jnp.real(sum_lll)
#           result = (out_phase, out_logabs)
#       else:
#           # result[0] := sign ∈ {−1,+1}. Do NOT add any phase here.
#           out_logabs  = result[1] + jnp.real(sum_lll)
#           result = (result[0], out_logabs)

#       if 'state_scale' in params:
#           result = (result[0], result[1] + params['state_scale'])

#       return result

#   return networks.Network(
#       options=options,
#       init=network_init,
#       apply=network_apply,
#       orbitals=orbitals_apply,
#   )