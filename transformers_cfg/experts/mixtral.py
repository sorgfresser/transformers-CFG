from typing import TYPE_CHECKING, List, Optional, Tuple, Union
from transformers import (
    MixtralForCausalLM,
    LogitsProcessorList,
    StoppingCriteriaList,
    GenerationConfig,
)
from transformers_cfg.switches import TooManyExpertsError
from transformers_cfg.generation.logits_process import GrammarConstrainedLogitsProcessor
from transformers.generation.utils import (
    GenerateNonBeamOutput,
    GenerateEncoderDecoderOutput,
    GenerateDecoderOnlyOutput,
)
import copy
import time
from transformers.models.mixtral.modeling_mixtral import (
    MIXTRAL_INPUTS_DOCSTRING,
    MoeCausalLMOutputWithPast,
    _CONFIG_FOR_DOC,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
    CrossEntropyLoss,
    MixtralModel,
    MoeModelOutputWithPast,
    DynamicCache,
    Cache,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
    MIXTRAL_ATTENTION_CLASSES,
    MixtralRMSNorm,
    MixtralSparseMoeBlock,
)
from transformers.utils import logging
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F
import warnings

if TYPE_CHECKING:
    from transformers import MixtralConfig
    from transformers.generation.utils import BaseStreamer

KVCacheType = Tuple[Tuple["torch.DoubleTensor", "torch.DoubleTensor"], ...]

logger = logging.get_logger(__name__)


@dataclass
class MoeModelOutputWithPastExperts(MoeModelOutputWithPast):
    experts: Optional[Tuple[torch.FloatTensor]] = None
    expert_caches: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class MoeCausalLMOutputWithPastExperts(MoeCausalLMOutputWithPast):
    experts: Optional[Tuple[torch.FloatTensor]] = None
    expert_caches: Optional[Tuple[torch.FloatTensor]] = None


class GenerationConfigRoutable(GenerationConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.switch_experts = kwargs.pop("switch_experts", None)


class MixtralSparseMoeBlockRoutable(MixtralSparseMoeBlock):

    def __init__(self, config):
        super().__init__(config)
        self.combinations_idx = torch.combinations(
            torch.arange(self.num_experts), self.top_k
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        experts_used: Optional[List[List[Tuple[int]]]] = None,
        experts_cache: Optional[torch.Tensor] = None,
        cache_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        assert (
            batch_size == 1 or experts_used is None
        )  # experts_used is only supported for batch_size=1, TODO: fix
        if self.training and self.jitter_noise > 0:
            hidden_states *= torch.empty_like(hidden_states).uniform_(
                1.0 - self.jitter_noise, 1.0 + self.jitter_noise
            )
        hidden_states = hidden_states.view(-1, hidden_dim)
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
            for batch in range(batch_size):
                if len(experts_used[batch]) > sequence_length:
                    # Clip due to kv caching
                    experts_used[batch] = experts_used[batch][-sequence_length:]
                for idx, experts_token in enumerate(experts_used[batch]):
                    if experts_token is None:
                        continue
                    for tok in experts_token:
                        # Set mask accordingly
                        expert_comb = torch.tensor(tok, device=routing_combs.device)
                        combinations_mask[idx] = combinations_mask[idx] & ~(
                            (combinations[idx] == expert_comb).all(dim=-1)
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
            experts_cache.to(hidden_states.device)
            if experts_cache is not None
            else torch.zeros(
                (batch_size * sequence_length, self.num_experts, hidden_dim),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
        )
        cache_mask = (
            cache_mask.to(hidden_states.device)
            if cache_mask is not None
            else torch.zeros(
                (batch_size * sequence_length, self.num_experts),
                dtype=torch.bool,
                device=hidden_states.device,
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
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)

            test_current_hidden_states = (
                expert_layer(current_state) * routing_weights[top_x, idx, None]
            )
            assert torch.allclose(current_hidden_states, test_current_hidden_states, atol=1e-03, rtol=1e-03)

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(hidden_states.dtype)
            )
        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim
        )
        return final_hidden_states, router_logits, selected_experts, experts_cache


class MixtralDecoderLayerRoutable(nn.Module):
    def __init__(self, config: "MixtralConfig", layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = MIXTRAL_ATTENTION_CLASSES[config._attn_implementation](
            config, layer_idx
        )

        self.block_sparse_moe = MixtralSparseMoeBlockRoutable(config)
        self.input_layernorm = MixtralRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = MixtralRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

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
        experts_used: Optional[List[List[Tuple[int]]]] = None,
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

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, router_logits, experts, experts_cache = self.block_sparse_moe(
            hidden_states, experts_used, experts_cache, experts_cache_mask
        )
        hidden_states = residual + hidden_states

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


class MixtralModelRoutable(MixtralModel):
    _no_split_modules = ["MixtralDecoderLayer", "MixtralDecoderLayerRoutable"]

    def __init__(self, config: "MixtralConfig"):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                MixtralDecoderLayerRoutable(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = MixtralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        streamer: Optional["BaseStreamer"],
        logits_warper: Optional[LogitsProcessorList] = None,
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        return super()._sample(
            input_ids,
            logits_processor,
            stopping_criteria,
            generation_config,
            synced_gpus,
            streamer,
            logits_warper,
            **model_kwargs,
        )

    @add_start_docstrings_to_model_forward(MIXTRAL_INPUTS_DOCSTRING)
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
        experts_used: Optional[List[List[Tuple[int]]]] = None,
        experts_caches: Optional[List[torch.Tensor]] = None,
        experts_cache_masks: Optional[List[torch.Tensor]] = None,
        output_experts: bool = False,
        all_hidden_states: Optional[tuple[torch.FloatTensor]] = None,
    ) -> Union[Tuple, MoeModelOutputWithPastExperts]:
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
            inputs_embeds = self.embed_tokens(input_ids)

        if (
            attention_mask is not None
            and self._attn_implementation == "flash_attention_2"
            and use_cache
        ):
            is_padding_right = attention_mask[:, -1].sum().item() != batch_size
            if is_padding_right:
                raise ValueError(
                    "You are attempting to perform batched generation with padding_side='right'"
                    " this may lead to unexpected behaviour for Flash Attention version of Mixtral. Make sure to "
                    " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                )

        if (experts_caches is None) != (experts_cache_masks is None):
            raise ValueError(
                "experts_caches and experts_cache_masks must be used together."
            )

        if self._attn_implementation == "flash_attention_2":
            # 2d mask is passed through the layers
            attention_mask = (
                attention_mask
                if (attention_mask is not None and 0 in attention_mask)
                else None
            )
        elif self._attn_implementation == "sdpa" and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
                sliding_window=self.config.sliding_window,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
                sliding_window=self.config.sliding_window,
            )

        hidden_states = (
            inputs_embeds if all_hidden_states is None else all_hidden_states[0]
        )

        # decoder layers
        all_hidden_states_new = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None
        all_experts = () if output_experts else None
        all_experts_caches = () if output_experts else None
        next_decoder_cache = None
        compute_new = False

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states_new += (hidden_states,)
            if experts_used is not None and any(
                map(lambda x: any(map(lambda y: y is not None, x)), experts_used[idx])
            ):
                compute_new = True
            if (
                all_hidden_states is not None
                and (idx < len(all_hidden_states) - 1)
                and not compute_new
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

        hidden_states = self.norm(hidden_states) if compute_new else hidden_states

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


class MixtralForCausalLMRoutable(MixtralForCausalLM):
    def __init__(self, config: "MixtralConfig"):
        super().__init__(config)
        self.model = MixtralModelRoutable(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_local_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        # Initialize weights and apply final processing
        self.post_init()

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
                    batch_outputs.router_logits = [
                        outputs.router_logits[j].view(
                            batch_size, -1, outputs.router_logits[j].shape[1]
                        )[i, ...]
                        for j in range(len(outputs.router_logits))
                    ]
                    batch_outputs.experts = [
                        outputs.experts[j].view(
                            batch_size, -1, outputs.experts[j].shape[1]
                        )[i, ...]
                        for j in range(len(outputs.experts))
                    ]
                    batch_outputs.expert_caches = [
                        outputs.expert_caches[j].view(
                            batch_size, -1, *outputs.expert_caches[j].shape[1:]
                        )[i, ...]
                        for j in range(len(outputs.expert_caches))
                    ]
                    # batch_outputs.past_key_values = [
                    #     (key[i, ...][None, ...], value[i, ...][None, ...])
                    #     for key, value in outputs.past_key_values
                    # ]

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

                    # Expand router logits to batch size, seq len, num experts
                    # If the sequence length is different, key value caching has been used, thus router logits is only 1
                    if (
                        input_ids.shape[1] * 1
                        != batch_outputs.router_logits[0].shape[0]
                    ):
                        batch_outputs.router_logits = [
                            batch_outputs.router_logits[j][-1].view(1, 1, -1)
                            for j in range(len(batch_outputs.router_logits))
                        ]
                        # Simply expand this as we're using the last token later on
                        batch_outputs.router_logits = [
                            batch_outputs.router_logits[j].expand(
                                1, input_ids.shape[1], -1
                            )
                            for j in range(len(batch_outputs.router_logits))
                        ]
                    router_logits = [
                        batch_outputs.router_logits[j].view(1, input_ids.shape[1], -1)
                        for j in range(len(batch_outputs.router_logits))
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

                    # If the sequence length is different, key value caching has been used, thus experts is only 1
                    if input_ids.shape[1] * 1 != batch_outputs.experts[0].shape[0]:
                        batch_outputs.experts = [
                            batch_outputs.experts[j][-1].view(1, 1, 1, -1)
                            for j in range(len(batch_outputs.experts))
                        ]
                        batch_outputs.experts = [
                            batch_outputs.experts[j].expand(
                                1, input_ids.shape[1], 1, -1
                            )
                            for j in range(len(batch_outputs.experts))
                        ]
                    experts_tried = [
                        batch_outputs.experts[j].view(1, input_ids.shape[1], 1, -1)
                        for j in range(len(batch_outputs.experts))
                    ]
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
                            # Apply mask
                            experts_list = [
                                experts_layer.tolist()
                                for experts_layer in experts_tried
                            ]
                            for layer in range(len(router_logits)):
                                experts_list[layer][0] = [
                                    (
                                        None
                                        if not switching_mask[layer][0, seq]
                                        else experts_list[layer][0][seq]
                                    )
                                    for seq in range(len(experts_list[layer][0]))
                                ]
                            # Update model inputs
                            batch_inputs["experts_used"] = experts_list
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

                            # Set not computed (because of cache) batch outputs based on outputs
                            # the last k layers may not have been cached, combine batch output and regular output
                            computed_layers = len(batch_outputs.router_logits)
                            batch_outputs.router_logits = [
                                outputs.router_logits[j].view(
                                    batch_size, -1, outputs.router_logits[j].shape[1]
                                )[i, ...]
                                for j in range(
                                    len(outputs.router_logits) - computed_layers
                                )
                            ] + list(batch_outputs.router_logits)
                            batch_outputs.experts = [
                                outputs.experts[j].view(
                                    batch_size, -1, outputs.experts[j].shape[1]
                                )[i, ...]
                                for j in range(len(outputs.experts) - computed_layers)
                            ] + list(batch_outputs.experts)
                            batch_outputs.expert_caches = [
                                outputs.expert_caches[j].view(
                                    batch_size, -1, *outputs.expert_caches[j].shape[1:]
                                )[i, ...]
                                for j in range(
                                    len(outputs.expert_caches) - computed_layers
                                )
                            ] + list(batch_outputs.expert_caches)
                            # batch_outputs.past_key_values = [
                            #     (key[i, ...][None, ...], value[i, ...][None, ...])
                            #     for key, value in outputs.past_key_values
                            # ]

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
                            # Expand router logits to batch size, seq len, num experts
                            # If the sequence length is different, key value caching has been used, thus router logits is only 1
                            if (
                                input_ids.shape[1] * 1
                                != batch_outputs.router_logits[0].shape[0]
                            ):
                                batch_outputs.router_logits = [
                                    batch_outputs.router_logits[j][-1].view(1, 1, -1)
                                    for j in range(len(batch_outputs.router_logits))
                                ]
                                # Simply expand this as we're using the last token later on
                                batch_outputs.router_logits = [
                                    batch_outputs.router_logits[j].expand(
                                        1, input_ids.shape[1], -1
                                    )
                                    for j in range(len(batch_outputs.router_logits))
                                ]
                            router_logits = [
                                batch_outputs.router_logits[j].view(
                                    1, input_ids.shape[1], -1
                                )
                                for j in range(len(batch_outputs.router_logits))
                            ]
                            # If the sequence length is different, key value caching has been used, thus experts is only 1
                            if (
                                input_ids.shape[1] * 1
                                != batch_outputs.experts[0].shape[0]
                            ):
                                batch_outputs.experts = [
                                    batch_outputs.experts[j][-1].view(1, 1, 1, -1)
                                    for j in range(len(batch_outputs.experts))
                                ]
                                batch_outputs.experts = [
                                    batch_outputs.experts[j].expand(
                                        1, input_ids.shape[1], 1, -1
                                    )
                                    for j in range(len(batch_outputs.experts))
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
                        batch_inputs["experts_used"] = None
                        batch_outputs = self(
                            **batch_inputs,
                            return_dict=True,
                            output_attentions=output_attentions,
                            output_hidden_states=output_hidden_states,
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
                    next_token_logits[i] = next_token_logits_batch[0]
                    next_token_scores[i] = next_token_scores_batch[0]
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
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids

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

    @add_start_docstrings_to_model_forward(MIXTRAL_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=MoeCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC
    )
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
        experts_used: Optional[List[List[Tuple[int]]]] = None,
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

        Example:

        ```python
        >>> from transformers import AutoTokenizer, MixtralForCausalLM

        >>> model = MixtralForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
        >>> tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

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
        outputs = self.model(
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
            experts_caches=experts_caches,
            experts_cache_masks=experts_cache_masks,
            all_hidden_states=all_hidden_states,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
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
