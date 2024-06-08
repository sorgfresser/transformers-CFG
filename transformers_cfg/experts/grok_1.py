""" Grok model configuration"""

from transformers.configuration_utils import PretrainedConfig
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import (
    LogitsProcessorList,
    StoppingCriteriaList,
    GenerationConfig,
)
import copy

from transformers_cfg.switches import TooManyExpertsError
from transformers_cfg.generation.logits_process import GrammarConstrainedLogitsProcessor
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import (
    MoeCausalLMOutputWithPast,
    MoeModelOutputWithPast,
)
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
)

from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import is_torch_greater_or_equal_than_1_13
from transformers.utils import (
    logging,
)
from transformers.utils.import_utils import is_torch_fx_available
from transformers.generation.utils import (
    GenerateNonBeamOutput,
    GenerateEncoderDecoderOutput,
    GenerateDecoderOnlyOutput,
)
from dataclasses import dataclass


logger = logging.get_logger(__name__)


@dataclass
class MoeModelOutputWithPastExperts(MoeModelOutputWithPast):
    experts: Optional[Tuple[torch.FloatTensor]] = None
    expert_caches: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class MoeCausalLMOutputWithPastExperts(MoeCausalLMOutputWithPast):
    experts: Optional[Tuple[torch.FloatTensor]] = None
    expert_caches: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class GenerateDecoderOnlyOutputExperts(GenerateDecoderOnlyOutput):
    num_fallbacks: Optional[int] = None
    num_switches: Optional[int] = None
    num_switches_wo_fallback: Optional[int] = None


class GenerationConfigRoutable(GenerationConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.switch_experts = kwargs.pop("switch_experts", None)


class GrokConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`GrokModel`].

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the Mixtral model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`MixtralModel`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 14336):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to `8`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to `4096*32`):
            The maximum sequence length that this model might ever be used with. Mixtral's sliding window attention
            allows sequence of up to 4096*32 tokens.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*):
            The id of the padding token.
        bos_token_id (`int`, *optional*, defaults to 1):
            The id of the "beginning-of-sequence" token.
        eos_token_id (`int`, *optional*, defaults to 2):
            The id of the "end-of-sequence" token.
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether the model's input and output word embeddings should be tied.
        rope_theta (`float`, *optional*, defaults to 100000.0):
            The base period of the RoPE embeddings.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        num_experts_per_tok (`int`, *optional*, defaults to 2):
            The number of experts to root per-token, can be also interpreted as the `top-p` routing
            parameter
        num_local_experts (`int`, *optional*, defaults to 8):
            Number of experts per Sparse MLP layer.
        output_router_logits (`bool`, *optional*, defaults to `False`):
            Whether the router logits should be returned by the model. Enabling this will also
            allow the model to output the auxiliary loss. See [here]() for more details
        router_aux_loss_coef (`float`, *optional*, defaults to 0.001):
            The aux loss factor for the total loss.

    """

    model_type = "grok"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=131072,
        hidden_size=6144,
        intermediate_size=32768,
        num_hidden_layers=64,
        num_attention_heads=48,
        num_key_value_heads=8,
        hidden_act="silu",
        max_position_embeddings=4096,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=True,
        rope_theta=1e5,
        attention_dropout=0.0,
        num_experts_per_tok=2,
        num_local_experts=8,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        output_multiplier_scale=0.5773502691896257,
        embedding_multiplier_scale=78.38367176906169,
        attn_output_multiplier=0.08838834764831845,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout

        self.num_experts_per_tok = num_experts_per_tok
        self.num_local_experts = num_local_experts
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.output_multiplier_scale = output_multiplier_scale
        self.embedding_multiplier_scale = embedding_multiplier_scale
        self.attn_output_multiplier = attn_output_multiplier
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


# This makes `_prepare_4d_causal_attention_mask` a leaf function in the FX graph.
# It means that the function will not be traced through and simply appear as a node in the graph.
if is_torch_fx_available():
    if not is_torch_greater_or_equal_than_1_13:
        import torch.fx

    _prepare_4d_causal_attention_mask = torch.fx.wrap(_prepare_4d_causal_attention_mask)


logger = logging.get_logger(__name__)


# Copied from transformers.models.llama.modeling_llama._get_unpad_data
def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->Grok
class GrokRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        GrokRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.float32))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


# Copied from transformers.models.mistral.modeling_mistral.MistralRotaryEmbedding with Mistral->Grok
class GrokRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device)
                / self.dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=torch.int64
        ).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.mistral.modeling_mistral.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def load_balancing_loss_func(
    gate_logits: torch.Tensor,
    num_experts: torch.Tensor,
    top_k=2,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.FloatTensor:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits (Union[`torch.Tensor`, Tuple[torch.Tensor]):
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        attention_mask (`torch.Tensor`, None):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.
        num_experts (`int`):
            Number of experts

    Returns:
        The auxiliary loss.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return torch.tensor(0.0)

    compute_device = gate_logits[0].device
    concatenated_gate_logits = torch.cat(
        [layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0
    )

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (
            batch_size * sequence_length
        )

        # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand(
                (num_hidden_layers, batch_size, sequence_length, top_k, num_experts)
            )
            .reshape(-1, top_k, num_experts)
            .to(compute_device)
        )

        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.sum(
            expert_mask.float() * expert_attention_mask, dim=0
        ) / torch.sum(expert_attention_mask, dim=0)

        # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(compute_device)
        )

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.sum(
            routing_weights * router_per_expert_attention_mask, dim=0
        ) / torch.sum(router_per_expert_attention_mask, dim=0)

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts


def update_combinations_mask(
    combinations_mask, combinations, experts_used, switching_mask
):
    experts_used = experts_used.squeeze(0)
    switching_mask = switching_mask.squeeze(0)
    seq_len, num_combinations, comb_size = combinations.shape
    experts_seq_len, m, _ = experts_used.shape
    # Kv-caching
    if experts_seq_len != seq_len:
        experts_used = experts_used[-1, ...].unsqueeze(0)
        switching_mask = switching_mask[-1, ...].unsqueeze(0)

    # Expand switching_mask to match the shape of combinations_mask for masking
    expanded_switching_mask = switching_mask.unsqueeze(1).expand(-1, num_combinations)

    # Reshape combinations and experts_used for broadcasting and comparison
    combinations_reshaped = combinations.view(seq_len, num_combinations, 1, comb_size)
    experts_used_reshaped = experts_used.view(seq_len, 1, m, comb_size)

    # Check for equality and reduce across the combination size dimension
    match_matrix = torch.all(combinations_reshaped == experts_used_reshaped, dim=-1)

    # Any match across experts_used combinations means the combination should be masked out
    matches = torch.any(match_matrix, dim=-1)

    # Apply the mask where switching_mask is True
    combinations_mask = combinations_mask & ~(expanded_switching_mask & matches)

    return combinations_mask


class GrokAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper.
    """

    def __init__(self, config: GrokConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.attn_output_multiplier = config.attn_output_multiplier
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.query = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.key = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.value = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.linear = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

        self.rotary_emb = GrokRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        bsz, q_len, _ = hidden_states.size()

        query_states = self.query(hidden_states)
        key_states = self.key(hidden_states)
        value_states = self.value(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = (
            torch.matmul(query_states, key_states.transpose(2, 3))
            * self.attn_output_multiplier
        )
        attn_weights = 30 * torch.tanh(attn_weights / 30)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.linear(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class GrokBlockSparseTop2MLP(nn.Module):
    def __init__(self, config: GrokConfig):
        super().__init__()
        self.ffn_dim = config.intermediate_size
        self.hidden_dim = config.hidden_size

        self.linear_v = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.linear_1 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.linear = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)

        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        current_hidden_states = self.act_fn(self.linear(hidden_states)) * self.linear_v(
            hidden_states
        )
        current_hidden_states = self.linear_1(current_hidden_states)
        return current_hidden_states


class GrokSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok

        # gating
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)

        self.experts = nn.ModuleList(
            [GrokBlockSparseTop2MLP(config) for _ in range(self.num_experts)]
        )
        self.combinations_idx = torch.combinations(
            torch.arange(self.num_experts), self.top_k
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        experts_used: Optional[torch.IntTensor] = None,
        switching_mask: Optional[torch.BoolTensor] = None,
        experts_cache: Optional[torch.Tensor] = None,
        cache_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        assert (
            batch_size == 1 or experts_used is None
        )  # experts_used is only supported for batch_size=1, TODO: fix
        hidden_states = hidden_states.view(-1, hidden_dim)
        switching_mask = (
            switching_mask.to(hidden_states.device)
            if switching_mask is not None
            else None
        )
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        # Select the top-k expert combination not used already
        routing_combs = routing_weights[:, None].expand(
            -1, self.combinations_idx.shape[0], -1
        )
        combinations = self.combinations_idx.to(routing_combs.device)[None].expand(
            router_logits.shape[0], -1, -1
        )
        combinations_mask = torch.ones(
            combinations.shape[:-1], dtype=torch.bool, device=combinations.device
        )  # (batch * sequence_length, n_combinations)
        # Remove expert combs already used from combinations
        if experts_used is not None:
            combinations_mask = update_combinations_mask(
                combinations_mask, combinations, experts_used, switching_mask
            )

        # Apply mask
        routing_combs = routing_combs.masked_fill(~combinations_mask[:, :, None], 0)
        routing_combs = routing_combs.gather(dim=-1, index=combinations).sum(dim=-1)
        # Select max
        selected_idx = torch.argmax(routing_combs, dim=-1)
        # Assert sanity
        assert torch.all(
            combinations_mask[torch.arange(router_logits.shape[0]), selected_idx]
        )
        selected_experts = combinations[
            torch.arange(router_logits.shape[0]), selected_idx
        ]
        # Select from routing weights
        routing_weights = routing_weights.gather(dim=-1, index=selected_experts)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.num_experts
        ).permute(2, 1, 0)

        # Assert experts cache
        assert experts_cache is None or experts_cache.shape == (
            batch_size * sequence_length,
            self.num_experts,
            hidden_dim,
        )
        experts_cache = (
            experts_cache.to(routing_combs.device)
            if experts_cache is not None
            else torch.zeros(
                (batch_size * sequence_length, self.num_experts, hidden_dim),
                dtype=hidden_states.dtype,
                device=routing_combs.device,
            )
        )
        cache_mask = (
            cache_mask.to(routing_combs.device)
            if cache_mask is not None
            else torch.zeros(
                (batch_size * sequence_length, self.num_experts),
                dtype=torch.bool,
                device=routing_combs.device,
            )
        )

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            # Compute if cache miss
            indexed_mask = cache_mask[top_x, expert_idx]
            input_state = hidden_states[None, top_x[~indexed_mask]].reshape(
                -1, hidden_dim
            )
            expert_output = expert_layer(input_state)
            # Merge cache and computed
            experts_cache[top_x[~indexed_mask], expert_idx] = expert_output
            current_hidden_states = (
                experts_cache[top_x, expert_idx] * routing_weights[top_x, idx, None]
            )

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(hidden_states.dtype)
            )
        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim
        )
        return final_hidden_states, router_logits, selected_experts, experts_cache


class GrokDecoderLayer(nn.Module):
    def __init__(self, config: GrokConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok

        self.multi_head_attention = GrokAttention(config, layer_idx)
        self.block_sparse_moe = GrokSparseMoeBlock(config)

        self.rms_norm = GrokRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rms_norm_1 = GrokRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rms_norm_2 = GrokRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rms_norm_3 = GrokRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        output_experts: Optional[bool] = False,
        experts_used: Optional[torch.IntTensor] = None,
        switching_mask: Optional[torch.BoolTensor] = None,
        experts_cache: Optional[torch.Tensor] = None,
        experts_cache_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_router_logits (`bool`, *optional*):
                Whether or not to return the logits of all the routers. They are useful for computing the router loss, and
                should not be returned during inference.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
        """

        residual = hidden_states

        hidden_states = self.rms_norm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.multi_head_attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + self.rms_norm_1(hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.rms_norm_2(hidden_states)
        hidden_states, router_logits, experts, experts_cache = self.block_sparse_moe(
            hidden_states,
            experts_used,
            switching_mask,
            experts_cache,
            experts_cache_mask,
        )

        hidden_states = residual + self.rms_norm_3(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if output_experts:
            outputs += (experts_cache,)

        if output_router_logits:
            outputs += (router_logits,)
        if output_experts:
            outputs += (experts,)

        return outputs


# Copied from transformers.models.mistral.modeling_mistral.MistralPreTrainedModel with Mistral->Grok
class GrokPreTrainedModel(PreTrainedModel):
    config_class = GrokConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True
    _no_split_modules = ["GrokDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _keys_to_ignore_on_load_missing = [r"lm_head.*."]
    _supports_flash_attn_2 = False
    _supports_sdpa = False

    def _init_weights(self, module):
        pass


# Copied from transformers.models.mistral.modeling_mistral.MistralModel with Mistral->Grok
class GrokModel(GrokPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`GrokDecoderLayer`]

    Args:
        config: GrokConfig
    """

    def __init__(self, config: GrokConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embedding_multiplier_scale = config.embedding_multiplier_scale

        self.in_out_embed = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.decoder_layer = nn.ModuleList(
            [
                GrokDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.rms_norm = GrokRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.in_out_embed

    def set_input_embeddings(self, value):
        self.in_out_embed = value

    # Ignore copy
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        experts_used: Optional[List[torch.IntTensor]] = None,
        switching_mask: Optional[List[torch.BoolTensor]] = None,
        experts_caches: Optional[List[torch.Tensor]] = None,
        experts_cache_masks: Optional[List[torch.Tensor]] = None,
        output_experts: bool = False,
        all_hidden_states: Optional[tuple[torch.FloatTensor]] = None,
    ) -> Union[Tuple, MoeModelOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_router_logits = (
            output_router_logits
            if output_router_logits is not None
            else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds"
            )
        if (experts_used is None) != (switching_mask is None):
            raise ValueError("experts_used and switching_mask must be used together.")

        if (experts_caches is None) != (experts_cache_masks is None):
            raise ValueError(
                "experts_caches and experts_cache_masks must be used together."
            )

        past_key_values_length = 0

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.in_out_embed(input_ids)

        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )

        if all_hidden_states is None:
            hidden_states = inputs_embeds
            hidden_states *= self.embedding_multiplier_scale
        else:
            hidden_states = all_hidden_states[0]

        # decoder layers
        all_hidden_states_new = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None
        all_experts = () if output_experts else None
        all_experts_caches = () if output_experts else None
        next_decoder_cache = None
        compute_new = False

        for idx, decoder_layer in enumerate(self.decoder_layer):
            if output_hidden_states:
                all_hidden_states_new += (hidden_states,)
            if experts_used is not None and torch.any(switching_mask[idx]):
                compute_new = True
            if (
                not compute_new
                and all_hidden_states is not None
                and (idx < len(all_hidden_states) - 1)
            ):  # Offset by 1 because the first hidden state is the input embeds
                hidden_states = all_hidden_states[idx + 1]
                continue
            compute_new = True  # If one layer computes new hidden states, all subsequent layers will do so as well

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    output_router_logits,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    output_router_logits=output_router_logits,
                    use_cache=use_cache,
                    experts_used=(
                        experts_used[idx] if experts_used is not None else None
                    ),
                    switching_mask=(
                        switching_mask[idx] if switching_mask is not None else None
                    ),
                    output_experts=output_experts,
                    experts_cache=(
                        experts_caches[idx] if experts_caches is not None else None
                    ),
                    experts_cache_mask=(
                        experts_cache_masks[idx]
                        if experts_cache_masks is not None
                        else None
                    ),
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if output_router_logits:
                all_router_logits += (layer_outputs[-2 if output_experts else -1],)
            if output_experts:
                all_experts_caches += (
                    layer_outputs[
                        (
                            3
                            if output_attentions and use_cache
                            else (2 if use_cache or output_attentions else 1)
                        )
                    ],
                )
                all_experts += (layer_outputs[-1],)

        hidden_states = self.rms_norm(hidden_states) if compute_new else hidden_states

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states_new += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache()
                if next_decoder_cache is not None and use_legacy_cache
                else next_decoder_cache
            )

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_cache,
                    all_hidden_states_new,
                    all_self_attns,
                    all_router_logits,
                    all_experts_caches,
                    all_experts,
                ]
                if v is not None
            )
        return MoeModelOutputWithPastExperts(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states_new,
            attentions=all_self_attns,
            router_logits=all_router_logits,
            experts=all_experts,
            expert_caches=all_experts_caches,
        )


class GrokForCausalLM(GrokPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = GrokModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_local_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.output_multiplier_scale = config.output_multiplier_scale
        # Initialize weights and apply final processing
        self.post_init()

    def _expand_outputs_for_switching(
        self,
        outputs: MoeCausalLMOutputWithPastExperts,
        expand_size: int = 1,
        batch_size: int = 1,
    ):
        """
        Expand the outputs of the model for switching experts processing to the correct size

        Args:
            outputs: The outputs of the model
            expand_size: The size to expand the outputs to
        """
        assert batch_size == 1, "Batch size > 1 is not supported for switching experts"
        # If they match, we don't need to expand
        if expand_size * batch_size != outputs.router_logits[0].shape[0]:
            # Takes the last token of each layer, requires batch size of 1
            outputs.router_logits = [
                outputs.router_logits[j][-1].view(1, 1, -1)
                for j in range(len(outputs.router_logits))
            ]
            # Simply expand this as we're using the last token later on, this will drop the router logits
            # for tokens that are not the last
            outputs.router_logits = [
                outputs.router_logits[j].expand(1, expand_size, -1)
                for j in range(len(outputs.router_logits))
            ]
        # If they match, we don't need to expand
        if expand_size * batch_size != outputs.experts[0].shape[0]:
            # Simply expand this as we're using the last token later on, this will drop the experts
            # for tokens that are not the last
            outputs.experts = [
                outputs.experts[j][-1].view(1, 1, 1, -1)
                for j in range(len(outputs.experts))
            ]
            outputs.experts = [
                outputs.experts[j].expand(1, expand_size, 1, -1)
                for j in range(len(outputs.experts))
            ]
        return outputs

    def _merge_cached(
        self,
        old_outputs: MoeCausalLMOutputWithPastExperts,
        newly_computed: MoeCausalLMOutputWithPastExperts,
        batch_size: int,
        batch_idx: int = None,
    ):
        """
        Merge the newly computed outputs with the cached outputs

        Args:
            old_outputs: The cached outputs
            newly_computed: The newly computed outputs
            batch_size: The batch size
            batch_idx: The batch index, if newly_computed is only a single input of the batch
        """
        assert (
            batch_idx is not None and batch_size == 1
        ), "Batch idx must be set as multiple batch entries are not supported yet"
        computed_layers = len(newly_computed.router_logits)
        newly_computed.router_logits = [
            old_outputs.router_logits[j].view(
                batch_size, -1, old_outputs.router_logits[j].shape[1]
            )[batch_idx, ...]
            for j in range(len(old_outputs.router_logits) - computed_layers)
        ] + list(newly_computed.router_logits)
        newly_computed.experts = [
            old_outputs.experts[j].view(
                batch_size, -1, old_outputs.experts[j].shape[1]
            )[batch_idx, ...]
            for j in range(len(old_outputs.experts) - computed_layers)
        ] + list(newly_computed.experts)
        newly_computed.expert_caches = [
            old_outputs.expert_caches[j].view(
                batch_size, -1, *old_outputs.expert_caches[j].shape[1:]
            )[batch_idx, ...]
            for j in range(len(old_outputs.expert_caches) - computed_layers)
        ] + list(newly_computed.expert_caches)
        return newly_computed

    def get_input_embeddings(self):
        return self.transformer.in_out_embed

    def set_input_embeddings(self, value):
        self.transformer.in_out_embed = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.transformer = decoder

    def get_decoder(self):
        return self.transformer

    def _tie_weights(self):
        self._tie_or_clone_weights(self.lm_head, self.get_input_embeddings())

    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfigRoutable,
        synced_gpus: bool,
        streamer: Optional["BaseStreamer"],
        logits_warper: Optional[LogitsProcessorList] = None,
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            generation_config ([`~generation.GenerationConfig`]):
                The generation configuration to be used as parametrization of the decoding method.
            synced_gpus (`bool`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            logits_warper (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
                to warp the prediction score distribution of the language modeling head applied before multinomial
                sampling at each generation step. Only required with sampling strategies (i.e. `do_sample` is set in
                `generation_config`)
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`:
            A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.
        """
        # init values
        pad_token_id = generation_config.pad_token_id
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        has_eos_stopping_criteria = any(
            hasattr(criteria, "eos_token_id") for criteria in stopping_criteria
        )
        do_sample = generation_config.do_sample
        if do_sample is True and not isinstance(logits_warper, LogitsProcessorList):
            raise ValueError(
                "`do_sample` is set to `True`, `logits_warper` must be a `LogitsProcessorList` instance (it is "
                f"{logits_warper})."
            )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = (
            () if (return_dict_in_generate and output_attentions) else None
        )
        cross_attentions = (
            () if (return_dict_in_generate and output_attentions) else None
        )
        decoder_hidden_states = (
            () if (return_dict_in_generate and output_hidden_states) else None
        )

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = (
                model_kwargs["encoder_outputs"].get("attentions")
                if output_attentions
                else None
            )
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states")
                if output_hidden_states
                else None
            )

        # keep track of which sequences are already finished
        batch_size = input_ids.shape[0]
        this_peer_finished = False
        unfinished_sequences = torch.ones(
            batch_size, dtype=torch.long, device=input_ids.device
        )
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)
        num_fallbacks = 0
        num_switches = 0
        num_switches_wo_fallback = 0

        while self._has_unfinished_sequences(
            this_peer_finished, synced_gpus, device=input_ids.device
        ):
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(
                input_ids,
                output_router_logits=generation_config.output_router_logits,
                **model_kwargs,
            )
            if generation_config.switch_experts is not None:
                if not model_inputs["output_router_logits"]:
                    logger.warning(
                        "switch_experts is set but output_router_logits is False. Setting output_router_logits to True."
                    )
                model_inputs["output_router_logits"] = True
            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_token_scores = logits_processor(
                input_ids, next_token_logits, batch_idx=None
            )

            if generation_config.switch_experts is not None:
                # for i in range(batch_size):
                #     # last layer, last token of batch
                #     experts_tried[-1][
                #         int(
                #             (i + 1) * (outputs.router_logits[0].shape[0] // batch_size)
                #             - 1
                #         )
                #     ] = [
                #         tuple(
                #             outputs.experts[-1][
                #                 (i + 1)
                #                 * (outputs.router_logits[0].shape[0] // batch_size)
                #                 - 1,
                #                 :,
                #             ].tolist()
                #         )
                #     ]
                # TODO: allow batching
                for i in range(batch_size):
                    fallbacked = False
                    switches = 0
                    batch_inputs = copy.deepcopy(model_inputs)
                    # Only use the i-th batch
                    # We keep the first dimension to be able to implement batching easier
                    batch_inputs["input_ids"] = batch_inputs["input_ids"][i, ...][
                        None, ...
                    ]
                    batch_inputs["attention_mask"] = batch_inputs["attention_mask"][
                        i, ...
                    ][None, ...]
                    batch_inputs["position_ids"] = batch_inputs["position_ids"][i, ...][
                        None, ...
                    ]
                    batch_inputs["past_key_values"] = (
                        [
                            (key_cache[i, ...][None, ...], val_cache[i, ...][None, ...])
                            for key_cache, val_cache in batch_inputs["past_key_values"]
                        ]
                        if batch_inputs["past_key_values"] is not None
                        else None
                    )
                    batch_inputs["all_hidden_states"] = [
                        hidden_states[i, ...][None, ...]
                        for hidden_states in outputs.hidden_states
                    ]
                    # Do not build kv cache, that is job of initial run per token
                    batch_inputs["use_cache"] = (
                        batch_inputs["past_key_values"] is not None
                    )
                    batch_outputs = self(
                        **batch_inputs,
                        return_dict=True,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                    )

                    # Set not computed (because of cache) batch outputs based on outputs
                    batch_outputs = self._merge_cached(
                        outputs, batch_outputs, batch_size, i
                    )

                    next_token_logits_batch = batch_outputs.logits[:, -1, :]

                    # pre-process distribution
                    for proc in logits_processor:
                        if isinstance(proc, GrammarConstrainedLogitsProcessor):
                            proc.undo_last_step()

                    next_token_scores_batch = logits_processor(
                        input_ids[i, ...][None, ...],
                        next_token_logits_batch,
                        batch_idx=i,
                    )

                    batch_outputs = self._expand_outputs_for_switching(
                        batch_outputs, input_ids.shape[1], 1
                    )
                    router_logits = [
                        batch_outputs.router_logits[j].view(1, input_ids.shape[1], -1)
                        for j in range(len(batch_outputs.router_logits))
                    ]
                    experts_tried = [
                        batch_outputs.experts[j].view(1, input_ids.shape[1], 1, -1)
                        for j in range(len(batch_outputs.experts))
                    ]
                    switching_mask = [
                        torch.zeros(router_logits[j].shape[:2], dtype=torch.bool)
                        for j in range(len(router_logits))
                    ]
                    # Only last token switching is supported
                    # for i in range(len(outputs.router_logits)):
                    #     switching_mask[i][:, -1] = True
                    # Alternative: only last token and last layer
                    switching_mask[-1][:, -1] = True

                    allowed_tokens = (
                        next_token_scores_batch != -float("inf")
                    ).nonzero()
                    cache_mask = [
                        torch.zeros(
                            outputs.expert_caches[0].shape[:-1],
                            dtype=torch.bool,
                            device=experts_tried[0].device,
                        )
                        for _ in range(len(experts_tried))
                    ]
                    try:
                        while generation_config.switch_experts(
                            next_token_logits_batch[0],
                            allowed_tokens[allowed_tokens[:, 0] == i][:, -1],
                            router_logits[-1][0, -1],  # last layer, last token of batch
                            experts_tried[-1][0, -1],  # last layer, last token of batch
                        ):
                            switches += 1

                            # Update model inputs
                            batch_inputs["experts_used"] = experts_tried
                            batch_inputs["switching_mask"] = switching_mask
                            # Update cache
                            batch_inputs["experts_caches"] = batch_outputs.expert_caches
                            # Add newly cached to mask, every expert in experts_tried is cached
                            cache_mask = [
                                cache_mask[layer]
                                .to(experts_tried[0].device)
                                .scatter_(
                                    1,
                                    experts_tried[layer][
                                        :,
                                        -batch_outputs.expert_caches[
                                            0
                                        ]  # expert caches with batch size as separate dimension
                                        .view(
                                            experts_tried[layer].shape[0],
                                            -1,
                                            *batch_outputs.expert_caches[0].shape[1:],
                                        )
                                        .shape[1] :,
                                    ]
                                    .flatten(2, 3)
                                    .view(
                                        -1,
                                        experts_tried[layer].shape[2]
                                        * experts_tried[layer].shape[3],
                                    ),
                                    1,
                                )
                                for layer in range(len(cache_mask))
                            ]
                            batch_inputs["experts_cache_masks"] = cache_mask

                            batch_outputs = self(
                                **batch_inputs,
                                return_dict=True,
                                output_attentions=output_attentions,
                                output_hidden_states=output_hidden_states,
                            )

                            batch_outputs = self._merge_cached(
                                outputs, batch_outputs, batch_size, i
                            )

                            next_token_logits_batch = batch_outputs.logits[:, -1, :]

                            # pre-process distribution
                            next_token_scores_batch = logits_processor(
                                input_ids[i, ...][None, ...],
                                next_token_logits_batch,
                                batch_idx=i,
                            )
                            batch_outputs = self._expand_outputs_for_switching(
                                batch_outputs, input_ids.shape[1], 1
                            )
                            router_logits = [
                                batch_outputs.router_logits[j].view(
                                    1, input_ids.shape[1], -1
                                )
                                for j in range(len(batch_outputs.router_logits))
                            ]
                            # Concat experts
                            for layer in range(len(experts_tried)):
                                experts_tried[layer] = torch.cat(
                                    [
                                        experts_tried[layer],
                                        batch_outputs.experts[layer].view(
                                            1, input_ids.shape[1], 1, -1
                                        ),
                                    ],
                                    dim=2,
                                )

                            allowed_tokens = (
                                next_token_scores_batch != -float("inf")
                            ).nonzero()
                    except TooManyExpertsError:
                        # Fallback
                        logger.warning(
                            "Too many experts tried. Falling back to original model."
                        )
                        num_fallbacks += 1
                        fallbacked = True
                        batch_inputs["experts_used"] = None
                        batch_inputs["switching_mask"] = None
                        batch_outputs = self(
                            **batch_inputs,
                            return_dict=True,
                            output_attentions=output_attentions,
                            output_hidden_states=output_hidden_states,
                        )
                        next_token_logits_batch = batch_outputs.logits[:, -1, :]

                        # pre-process distribution
                        next_token_scores_batch = logits_processor(
                            input_ids[i, ...][None, ...],
                            next_token_logits_batch,
                            batch_idx=i,
                        )
                    next_token_logits[i] = next_token_logits_batch[0]
                    next_token_scores[i] = next_token_scores_batch[0]
                    num_switches += switches
                    if not fallbacked:
                        num_switches_wo_fallback += switches

            if do_sample:
                next_token_scores = logits_warper(input_ids, next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,)
                        if self.config.is_encoder_decoder
                        else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # token selection
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
                    1 - unfinished_sequences
                )

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(
                input_ids, scores
            )
            this_peer_finished = unfinished_sequences.max() == 0

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateDecoderOnlyOutputExperts(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                    num_fallbacks=num_fallbacks,
                    num_switches=num_switches,
                    num_switches_wo_fallback=num_switches_wo_fallback,
                )
        else:
            return input_ids

    # Ignore copy
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        experts_used: Optional[List[torch.LongTensor]] = None,
        switching_mask: Optional[List[torch.BoolTensor]] = None,
        experts_caches: Optional[List[torch.Tensor]] = None,
        experts_cache_masks: Optional[List[torch.Tensor]] = None,
        all_hidden_states: Optional[Tuple[torch.FloatTensor]] = None,
    ) -> Union[Tuple, MoeCausalLMOutputWithPastExperts]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        """

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_router_logits = (
            output_router_logits
            if output_router_logits is not None
            else self.config.output_router_logits
        )

        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            return_dict=return_dict,
            output_experts=True,
            experts_used=experts_used,
            switching_mask=switching_mask,
            experts_caches=experts_caches,
            experts_cache_masks=experts_cache_masks,
            all_hidden_states=all_hidden_states,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits * self.output_multiplier_scale
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        aux_loss = None

        if not return_dict:
            output = (logits,) + outputs[1:]
            if output_router_logits:
                output = (aux_loss,) + output
            return (loss,) + output if loss is not None else output

        return MoeCausalLMOutputWithPastExperts(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
            experts=outputs.experts,
            expert_caches=outputs.expert_caches,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        output_router_logits=None,
        **kwargs,
    ):
        # Omit tokens covered by past_key_values
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if (
                attention_mask is not None
                and attention_mask.shape[1] > input_ids.shape[1]
            ):
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "output_router_logits": output_router_logits,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx.to(past_state.device))
                    for past_state in layer_past
                ),
            )
        return reordered_past
