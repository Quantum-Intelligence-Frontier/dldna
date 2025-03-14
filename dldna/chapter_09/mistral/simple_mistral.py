import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
import math
from typing import Optional, Tuple, List, Dict, Union

class MistralConfig:
    def __init__(self,
                 vocab_size: int = 32000,
                 hidden_size: int = 4096,
                 intermediate_size: int = 14336,
                 num_hidden_layers: int = 32,
                 num_attention_heads: int = 32,
                 num_key_value_heads: int = 8,  # GQA
                 hidden_act: str = "silu",
                 max_position_embeddings: int = 4096 * 32,
                 rms_norm_eps: float = 1e-6,
                 use_cache: bool = True,
                 pad_token_id: int = -1,
                 bos_token_id: int = 1,
                 eos_token_id: int = 2,
                 pretraining_tp: int = 1,  # TP rank
                 tie_word_embeddings: bool = False,
                 rope_theta: float = 10000.0,
                 sliding_window: Optional[int] = 4096,
                 use_return_dict: bool = True  # 추가된 인자
                 ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.hidden_act = hidden_act
        self.use_cache = use_cache
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pretraining_tp = pretraining_tp
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_theta = rope_theta
        self.sliding_window = sliding_window
        self.use_return_dict = use_return_dict  

        if self.num_attention_heads % self.num_key_value_heads != 0:
           raise ValueError("num_attention_heads must be divisible by num_key_value_heads")


class MistralRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight.to(input_dtype) * hidden_states).to(input_dtype) # Input type casting
    

class MistralAttention(nn.Module):
    """Multi-headed attention with GQA and Sliding Window Attention."""
    def __init__(self, config: MistralConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads  # GQA
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.sliding_window = config.sliding_window # Sliding Window Attention

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                             f" and `num_heads`: {self.num_heads}).")

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.rotary_emb = MistralRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                output_attentions: Optional[bool] = False,
                use_cache: Optional[bool] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            hidden_states (torch.Tensor): (batch_size, sequence_length, hidden_size)
            attention_mask (torch.Tensor, optional): Attention mask.
            past_key_value (tuple, optional): Previously cached key and value tensors.
            output_attentions (bool, optional): Whether to output attention weights (not used here).
            use_cache (bool, optional): Whether to cache key/value tensors.
            
        Returns:
            Tuple[attn_output, None, past_key_value]: attn_output의 shape은 (batch_size, seq_length, hidden_size).
        """
        # use_cache 기본값 지정: 인자가 전달되지 않으면 config.use_cache 사용
        if use_cache is None:
            use_cache = self.config.use_cache

        batch_size, q_len, _ = hidden_states.size()

        # Project to query, key, value
        query_states = self.q_proj(hidden_states).view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(batch_size, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(batch_size, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Get rotary embeddings
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        cos = cos.to(query_states.device)
        sin = sin.to(query_states.device)

        # 각각의 시퀀스 길이에 맞는 position_ids 생성
        position_ids_q = torch.arange(0, q_len, dtype=torch.long, device=query_states.device).unsqueeze(0)
        past_length = past_key_value[0].shape[-2] if past_key_value is not None else 0
        new_key_len = key_states.shape[-2]
        position_ids_k = torch.arange(past_length, past_length + new_key_len, dtype=torch.long, device=query_states.device).unsqueeze(0)

        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids_q, position_ids_k
        )

        # 캐시용 원본 key/value
        cache_key_states = key_states
        cache_value_states = value_states

        # GQA 적용: key/value에 대해 repeat_interleave (attention 연산용)
        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        # 과거 캐시와 결합: past_key_value가 있다면 결합 (repeat 전의 텐서를 사용)
        if past_key_value is not None:
            cache_key_states = torch.cat([past_key_value[0], cache_key_states], dim=2)
            cache_value_states = torch.cat([past_key_value[1], cache_value_states], dim=2)
        past_key_value = (cache_key_states, cache_value_states) if use_cache else None

        if attention_mask is not None:
            attention_mask = attention_mask.bool()

        # Sliding Window Mask 생성 (필요 시)
        if self.sliding_window is not None:
            sliding_window_mask = torch.ones(q_len, q_len, device=query_states.device, dtype=torch.bool)
            for i in range(q_len):
                low = max(0, i - self.sliding_window // 2)
                high = min(q_len, i + self.sliding_window // 2 + 1)
                sliding_window_mask[i, low:high] = False
            sliding_window_mask = sliding_window_mask.view(1, 1, q_len, q_len)
            if attention_mask is not None:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1) & ~sliding_window_mask
            else:
                attention_mask = ~sliding_window_mask
        else:
            if attention_mask is not None:
                attention_mask = attention_mask[:, None, None, :]

        # Scaled dot-product attention 사용 (Causal)
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            is_causal=True,
        )

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch_size, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value



class MistralMLP(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MistralDecoderLayer(nn.Module):

    def __init__(self, config, layer_idx) -> None: 
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = MistralAttention(config)  # config만 전달
        self.mlp = MistralMLP(config)
        self.input_layernorm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

class MistralPreTrainedModel(nn.Module):

    def __init__(self, config):
        super().__init__()

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, MistralModel):
            module.gradient_checkpointing = value

class MistralModel(MistralPreTrainedModel):

    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [MistralDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Dict]:  # Changed to Dict

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        # Decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    past_key_values[layer_idx] if past_key_values else None,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_value=past_key_values[layer_idx] if past_key_values else None,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # Add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        # Return a dictionary (consistent with other parts of the code)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return {
            "last_hidden_state": hidden_states,
            "past_key_values": next_cache,
            "hidden_states": all_hidden_states,
            "attentions": all_self_attns,
        }


class MistralForCausalLM(MistralPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config   
        self.model = MistralModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None
    ):
        # 만약 return_dict가 None이라면 항상 dictionary를 반환하도록 설정
        if return_dict is None:
            return_dict = True

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs["last_hidden_state"]
        logits = self.lm_head(hidden_states)

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)
            loss = loss_fct(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))
            return {"loss": loss, "logits": logits}
        return {"logits": logits}


    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past_key_values
        )

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

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class MistralRotaryEmbedding(nn.Module):

    def __init__(self, dim, max_position_embeddings, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1. / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos()[None, None, :, :].to(device) # to device
        self.sin_cached = emb.sin()[None, None, :, :].to(device) # to device

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )
    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos()[None, None, :, :].to(device)  # to device
        self.sin_cached = emb.sin()[None, None, :, :].to(device)  # to device

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids_q, position_ids_k=None):
    """Applies rotary position embeddings to the query and key tensors using separate position IDs.
    
    If position_ids_k is not provided, it defaults to position_ids_q.
    
    Args:
        q (torch.Tensor): Query tensor of shape [batch, num_heads, q_seq_len, head_dim].
        k (torch.Tensor): Key tensor of shape [batch, num_key_value_heads, kv_seq_len, head_dim].
        cos (torch.Tensor): Cosine embedding tensor of shape [1, 1, seq_len, head_dim].
        sin (torch.Tensor): Sine embedding tensor of shape [1, 1, seq_len, head_dim].
        position_ids_q (torch.Tensor): Position IDs for queries of shape [batch, q_seq_len].
        position_ids_k (torch.Tensor, optional): Position IDs for keys of shape [batch, kv_seq_len].
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Rotated and position-embedded query and key tensors.
    """
    if position_ids_k is None:
        position_ids_k = position_ids_q

    # Remove extra dimensions: from [1, 1, seq_len, head_dim] to [seq_len, head_dim]
    cos = cos.squeeze(1).squeeze(0)
    sin = sin.squeeze(1).squeeze(0)
    
    # Select positional embeddings for query and key using separate IDs
    cos_q = cos[position_ids_q].unsqueeze(1)  # [batch, 1, q_seq_len, head_dim]
    sin_q = sin[position_ids_q].unsqueeze(1)  # [batch, 1, q_seq_len, head_dim]
    cos_k = cos[position_ids_k].unsqueeze(1)  # [batch, 1, kv_seq_len, head_dim]
    sin_k = sin[position_ids_k].unsqueeze(1)  # [batch, 1, kv_seq_len, head_dim]
    
    q_embed = (q * cos_q) + (rotate_half(q) * sin_q)
    k_embed = (k * cos_k) + (rotate_half(k) * sin_k)
    
    return q_embed, k_embed
