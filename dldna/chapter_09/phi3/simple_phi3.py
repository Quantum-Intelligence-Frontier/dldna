import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Dict, Union

# Phi-Mini model's configuration class
class PhiMiniConfig:
    def __init__(self,
                 vocab_size: int = 51200,  # Phi-2 vocab size
                 hidden_size: int = 2560,  # Phi-2 hidden size
                 intermediate_size: int = 10240, # Phi-2 intermediate size
                 num_hidden_layers: int = 32,  # Phi-2 number of layers
                 num_attention_heads: int = 32,  # Phi-2 number of heads
                 num_key_value_heads: int = 32,  # Phi-2 GQA, K=V=Q
                 hidden_act: str = "gelu_new", # Phi-2 uses GELU
                 max_position_embeddings: int = 2048, # Phi-2 context length
                 rms_norm_eps: float = 1e-6, # Phi-2 RMSNorm eps
                 use_cache: bool = True,
                 pad_token_id: int = None, # to be set later
                 bos_token_id: int = 1,  # to be set later
                 eos_token_id: int = 2,  # to be set later
                 rope_theta: float = 10000.0,
                 sliding_window: Optional[int] = None, # No sliding window by default
                 use_return_dict: bool = True):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.rope_theta = rope_theta
        self.sliding_window = sliding_window
        self.use_return_dict = use_return_dict

        # GQA is used, but Phi-2 and Phi-3 Mini use K=V=Q (i.e., MHA). Therefore, the validation logic below is unnecessary.
        # if self.num_attention_heads % self.num_key_value_heads != 0:
        #    raise ValueError("num_attention_heads must be divisible by num_key_value_heads")

# RMSNorm implementation
class PhiMiniRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps  # Keep the name as variance_epsilon

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype) # No need to convert type before multiplying by weight

# Rotary embedding
class PhiMiniRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base  # Keep base as an instance variable
        inv_freq = 1. / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Integrated cache creation and management logic
        self.max_seq_len_cached = 0
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_len=None):
        if seq_len >= self.max_seq_len_cached:  # Changed > to >=
            self.max_seq_len_cached = seq_len
            self._set_cos_sin_cache(seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:, :, :seq_len, ...],
            self.sin_cached[:, :, :seq_len, ...],
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos()[None, None, :, :].to(dtype)
        self.sin_cached = emb.sin()[None, None, :, :].to(dtype)

# Helper functions: rotate_half and apply_rotary_pos_emb
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos_q = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin_q = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    cos_k = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin_k = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos_q) + (rotate_half(q) * sin_q)
    k_embed = (k * cos_k) + (rotate_half(k) * sin_k)
    return q_embed, k_embed

# Phi-Mini Attention module

class PhiMiniAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.qkv_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim * 3, bias=True)

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(f"hidden_size must be divisible by num_heads (got {self.hidden_size} and {self.num_heads}).")

        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=True)
        self.rotary_emb = PhiMiniRotaryEmbedding(self.head_dim,
                                                 max_position_embeddings=self.max_position_embeddings,
                                                 base=self.rope_theta)


    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                output_attentions: Optional[bool] = False,
                use_cache: Optional[bool] = None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        batch_size, q_len, _ = hidden_states.size()

        qkv = self.qkv_proj(hidden_states)
        query_states, key_states, value_states = qkv.split(self.num_heads * self.head_dim, dim=2)
        query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Apply Rotary embedding: first calculate kv_seq_len (new tokens + past tokens)
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        # Calculate cos, sin: update cached values (here, calculate for the entire kv_seq_len)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        if past_key_value is None:
            # If there is no cache: [0, q_len) for the entire q_len
            position_ids = torch.arange(0, q_len, dtype=torch.long, device=query_states.device).unsqueeze(0)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        else:
            # If there is cache (incremental decoding):
            past_len = past_key_value[0].size(2)  # Number of tokens already accumulated
            position_ids_q = torch.arange(past_len, past_len + q_len, dtype=torch.long, device=query_states.device).unsqueeze(0)
            # position_ids_k = torch.arange(0, past_len + q_len, dtype=torch.long, device=query_states.device).unsqueeze(0) # 이 줄은 더 이상 필요하지 않습니다.
            query_states = apply_rotary_pos_emb_single(query_states, cos, sin, position_ids_q)
            # key_states = apply_rotary_pos_emb_single(key_states, cos, sin, position_ids_k) # 이 줄을 제거합니다.
            # value_states = apply_rotary_pos_emb_single(value_states, cos, sin, position_ids_k) # 이 줄도 필요하지 않습니다.

        # unified validation and combination code: verify dimensions of existing cache and newly computed key, value, then combine
        if past_key_value is not None:
            if past_key_value[0].size(-1) != key_states.size(-1):
                raise RuntimeError(f"Key tensor size mismatch: past_key_value[0].size(-1)={past_key_value[0].size(-1)} vs key_states.size(-1)={key_states.size(-1)}")
            if past_key_value[1].size(-1) != value_states.size(-1):
                raise RuntimeError(f"Value tensor size mismatch: past_key_value[1].size(-1)={past_key_value[1].size(-1)} vs value_states.size(-1)={value_states.size(-1)}")
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        past_key_value = (key_states, value_states) if use_cache else None

        # Calculate scaled dot-product attention
        if output_attentions:
            scale = 1.0 / math.sqrt(self.head_dim)
            attn_scores = torch.matmul(query_states, key_states.transpose(-2, -1)) * scale

            q_len_new = query_states.size(-2)
            kv_seq_len_new = key_states.size(-2)

            causal_mask = torch.triu(torch.full((q_len_new, kv_seq_len_new), float('-inf')), diagonal=1).to(query_states.device)

            if attention_mask is not None:
                combined_mask = (attention_mask + causal_mask.unsqueeze(0)).unsqueeze(1)
            else:
                combined_mask = causal_mask.unsqueeze(0).unsqueeze(1)

            attn_scores = attn_scores + combined_mask
            attn_weights = torch.softmax(attn_scores, dim=-1)
            attn_output = torch.matmul(attn_weights, value_states)
        else:
            if attention_mask is not None:
                q_len_new = query_states.size(-2)
                kv_seq_len_new = key_states.size(-2)
                causal_mask = torch.triu(torch.full((q_len_new, kv_seq_len_new), float('-inf')), diagonal=1).to(query_states.device)
                combined_mask = attention_mask + causal_mask.unsqueeze(0)
                attn_output = F.scaled_dot_product_attention(query_states, key_states, value_states,
                                                            attn_mask=combined_mask, is_causal=False)
            else:
                attn_output = F.scaled_dot_product_attention(query_states, key_states, value_states,
                                                            attn_mask=None, is_causal=True)
            attn_weights = None

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch_size, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights, past_key_value


def apply_rotary_pos_emb_single(x, cos, sin, position_ids):
    cos = cos.squeeze(0).squeeze(0)  # (max_seq_len, dim)
    sin = sin.squeeze(0).squeeze(0)  # (max_seq_len, dim)
    cos_part = cos[position_ids].unsqueeze(1)  # (B, 1, seq_len, dim)
    sin_part = sin[position_ids].unsqueeze(1)  # (B, 1, seq_len, dim)
    return (x * cos_part) + (rotate_half(x) * sin_part)


# Phi-Mini MLP module
class PhiMiniMLP(nn.Module):
    def __init__(self, config: PhiMiniConfig) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=True)  # Phi-2 uses bias=True
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=True)    # Phi-2 uses bias=True
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=True)  # Phi-2 uses bias=True
        self.act_fn = nn.GELU(approximate='tanh') # Phi-2 uses GELU with tanh approximation

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

# Phi-Mini decoder layer
class PhiMiniDecoderLayer(nn.Module):
    def __init__(self, config: PhiMiniConfig, layer_idx: int) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = PhiMiniAttention(config)
        self.mlp = PhiMiniMLP(config)
        self.input_layernorm = PhiMiniRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = PhiMiniRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # self.layer_idx = layer_idx # layer_idx is not used


    # Example of the forward method of PhiMiniDecoderLayer after modification
    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                output_attentions: Optional[bool] = False,
                use_cache: Optional[bool] = None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # self-attention block
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # self.self_attn() should be implemented to return (attn_output, attn_weights, present_key_value).
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # MLP block
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        # Return attention weights if output_attentions is True
        if output_attentions:
            outputs = (hidden_states, self_attn_weights)
        else:
            outputs = (hidden_states,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs



# Phi-Mini model (entire decoder)
class PhiMiniModel(nn.Module):
    def __init__(self, config: PhiMiniConfig) -> None:
        super().__init__()
        self.config = config
        self.pad_token_id = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.pad_token_id)
        self.layers = nn.ModuleList([PhiMiniDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = PhiMiniRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # self.gradient_checkpointing = False # Not used

    def forward(self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None) -> Union[Tuple, Dict]:

        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        # Decoder Layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for i, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[i] if past_key_values is not None else None

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[1],)

            if output_attentions:  # Even if `output_attentions` is True, return an empty tuple
                all_self_attns += (None,)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

        return {
            "last_hidden_state": hidden_states,
            "past_key_values": next_cache,
            "hidden_states": all_hidden_states,
            "attentions": all_self_attns,  # Always None (or an empty tuple)
        }

class PhiMiniForCausalLM(nn.Module):
    def __init__(self, config: PhiMiniConfig) -> None:
        super().__init__()
        self.config = config
        self.transformer = PhiMiniModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.transformer.embed_tokens

    def set_input_embeddings(self, value):
        self.transformer.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.transformer = decoder

    def get_decoder(self):
        return self.transformer

    def forward(self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None) -> Union[Tuple, Dict]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0] if not return_dict else transformer_outputs["last_hidden_state"]
        lm_logits = self.lm_head(hidden_states)
        loss = None

        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return {
                    "loss": loss,
                    "logits": lm_logits,
                    "past_key_values": transformer_outputs.get("past_key_values", None),
                    "hidden_states": transformer_outputs.get("hidden_states", None),
                    "attentions": transformer_outputs.get("attentions", None),
                }

    def post_init(self):
        """
        Post-initialization: Tie weights and initialize.
        """
        self.transformer.embed_tokens.weight = self.lm_head.weight # Phi-2, 3 tie weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights like original Phi-Mini (and Llama)"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)  # Llama init
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm): # for LayerNorm (not RMSNorm, but just in case)
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    @torch.no_grad()
    def generate(self, input_ids: torch.LongTensor, max_new_tokens: int = 256, temperature: float = 1.0,
                top_k: int = 0, top_p: float = 0.9, do_sample: bool = True,
                pad_token_id: Optional[int] = None,
                eos_token_id: Optional[int] = None) -> torch.LongTensor:
        """
        PhiMiniForCausalLM.generate() function (for auto-regressive text generation)

        Resolves the size mismatch issue of rotary embeddings in incremental decoding situations.
        If past_key_values exist, only the last token is passed to forward() instead of the entire sequence,
        so that the sequence length (q_len) for the new token becomes 1, making the size correct when applying rotary embeddings.
        """
        if pad_token_id is None:
            pad_token_id = self.config.pad_token_id
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id

        # Check if input_ids is a 2D tensor (batch_size, seq_len)
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)

        batch_size = input_ids.shape[0]
        use_cache = True
        past_key_values = None
        current_length = input_ids.shape[1]
        stop_flags = [False] * batch_size

        # Generate tokens up to max_new_tokens.
        for _ in range(max_new_tokens):
            # Initially, pass the entire sequence, and from then on, only pass the last token.
            if past_key_values is None:
                model_inputs = input_ids
            else:
                model_inputs = input_ids[:, -1:]


            outputs = self.forward(input_ids=model_inputs, use_cache=use_cache, past_key_values=past_key_values)
            logits = outputs["logits"]
            past_key_values = outputs["past_key_values"]

            # Use only the logits for the last token
            next_token_logits = logits[:, -1, :]
            next_token_logits = next_token_logits / temperature

            # Top-k sampling
            if top_k > 0:
                topk_vals, _ = torch.topk(next_token_logits, top_k)
                kth_val = topk_vals[:, -1].unsqueeze(1)
                next_token_logits = torch.where(next_token_logits < kth_val,
                                                torch.full_like(next_token_logits, float("-inf")),
                                                next_token_logits)

            # Top-p (nucleus) sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits = next_token_logits.masked_fill(indices_to_remove, float("-inf"))

            # Calculate probability distribution
            probs = F.softmax(next_token_logits, dim=-1)

            # Choose sampling or greedy decoding
            if do_sample:
                next_tokens = torch.multinomial(probs, num_samples=1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Check if eos_token has occurred
            for i in range(batch_size):
                if not stop_flags[i] and (next_tokens[i].item() == eos_token_id):
                    stop_flags[i] = True

            if all(stop_flags):
                break

            # Add the new token to the existing sequence
            input_ids = torch.cat((input_ids, next_tokens), dim=1)
            current_length += 1

        return input_ids