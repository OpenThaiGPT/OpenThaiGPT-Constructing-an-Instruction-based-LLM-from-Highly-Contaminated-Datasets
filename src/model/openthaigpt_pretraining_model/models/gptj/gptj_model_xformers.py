import torch
import xformers.ops as xops
from transformers.models.gptj.modeling_gptj import (
    GPTJAttention,
    GPTJModel,
    GPTJBlock,
    logger,
)
from transformers import GPTJForCausalLM
from torch import nn

from typing import Optional, Tuple, Union
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
)

XFORMER_ATTENTION_MODE = "xformers"


class GPTJAttentionXFormers(GPTJAttention):
    def __init__(self, config):
        super().__init__(config)
        self.attention_mode = config.attention_mode

    def _attn(
        self,
        query,
        key,
        value,
        attention_mask=None,
        head_mask=None,
    ):
        if self.attention_mode == XFORMER_ATTENTION_MODE:
            if attention_mask is not None:
                raise TypeError("Not support manual attention mask")

            if head_mask is not None:
                raise TypeError("Not support head_mask")

            # Attention output
            attn_output = xops.memory_efficient_attention(
                query.transpose(2, 1),
                key.transpose(2, 1),
                value.transpose(2, 1),
                xops.LowerTriangularMask(),
            ).transpose(2, 1)

            return attn_output, None
        return super()._attn(
            query,
            key,
            value,
            attention_mask,
            head_mask,
        )


class GPTJModelWithCheckpointing(GPTJModel):
    def __init__(self, config):
        super().__init__(config)

        self.embed_dim = config.n_embd
        self.vocab_size = config.vocab_size
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList(
            [GPTJBlockWithCheckpointing(config) for _ in range(config.n_layer)]
        )
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
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

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])  # type: ignore
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device  # type: ignore # noqa

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])  # type: ignore

        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1]).long()  # type: ignore

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))  # type: ignore
        else:
            past_length = past_key_values[0][0].size(-2)

        if position_ids is None:
            position_ids = torch.arange(  # type: ignore
                past_length,
                input_shape[-1] + past_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])  # type: ignore # noqa

        # Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)  # type: ignore
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length] # noqa
            # this attention mask is more simple than the triangular masking of causal attention # noqa
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]  # type: ignore

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and the dtype's smallest value for masked positions. # noqa
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # type: ignore
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x num_attention_heads x N x N
        # head_mask has shape n_layer x batch x num_attention_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        hidden_states = inputs_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."  # noqa
                )
                use_cache = False

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):  # type: ignore # noqa
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct) # noqa
                if layer_past is not None:
                    layer_past = tuple(  # type: ignore
                        past_state.to(hidden_states.device) for past_state in layer_past
                    )
                # Ensure that attention_mask is always on the same device as hidden_states # noqa
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)  # type: ignore # noqa
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)  # type: ignore
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)  # type: ignore
            outputs = block(
                hidden_states=hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                position_ids=position_ids,
                head_mask=head_mask[i],  # type: ignore
                use_cache=use_cache,
                output_attentions=output_attentions,
                gradient_checkpointing=self.gradient_checkpointing,
            )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)  # type: ignore

            if output_attentions:
                all_self_attentions = all_self_attentions + (  # type: ignore
                    outputs[2 if use_cache else 1],
                )

            # Model Parallel: If it's the last layer for that device, put things on the next device # noqa
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)  # type: ignore

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    presents,
                    all_hidden_states,
                    all_self_attentions,
                ]
                if v is not None
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class GPTJBlockWithCheckpointing(GPTJBlock):
    def __init__(self, config):
        super().__init__(config)
        self.attn = GPTJAttentionXFormers(config)

    def forward(
        self,
        hidden_states: Optional[torch.FloatTensor],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        gradient_checkpointing: Optional[bool] = False,
    ) -> Union[
        Tuple[torch.Tensor],
        Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]],
    ]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)

        # Attention
        if gradient_checkpointing and self.training:

            def create_custom_forward(module):
                def custom_forward(
                    hidden_states, layer_past, attention_mask, position_ids, head_mask
                ):
                    return module(
                        hidden_states,
                        layer_past=layer_past,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        head_mask=head_mask,
                    )

                return custom_forward

            attn_outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.attn),
                hidden_states,
                layer_past,
                attention_mask,
                position_ids,
                head_mask,
            )
        else:
            attn_outputs = self.attn(
                hidden_states=hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                position_ids=position_ids,
                head_mask=head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]

        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = attn_output + feed_forward_hidden_states + residual

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs


class GPTJForCausalLMWithCheckpointing(GPTJForCausalLM):
    _keys_to_ignore_on_load_missing = [
        r"h\.\d+\.attn\.masked_bias",
        r"h\.\d+\.attn\.bias",
    ]

    def __init__(self, config):
        super().__init__(config)
        use_checkpointing = config.get("use_checkpointing", False)
        checkpoint_only_attention = config.get("checkpoint_only_attention", False)
        if use_checkpointing and checkpoint_only_attention:
            self.transformer = GPTJModelWithCheckpointing(config)
            print("use model with gradient checkpointing only attention")
        if use_checkpointing:
            self.gradient_checkpointing_enable()
            print("use gradient checkpointing")
        else:
            self.transformer = GPTJModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

        self.vocab_size = config.vocab_size

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()
