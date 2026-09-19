from typing import Callable, List, Optional, Tuple, Union

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.utils import logging

from model.kv_caches.cache_utils import Cache, StaticCache, DynamicCache
from model.mor_model.util import ROUTER_TYPES, MoRLayerOutputWithPast
from model.base_model.modeling_llama import apply_rotary_pos_emb, eager_attention_forward, LlamaAttention
from util.misc import get_torch_dtype

logger = logging.get_logger(__name__)


class MoRLlamaAttention(LlamaAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)            
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            # This class is for hybrid KV sharing that leverages shared caches 
            # for inactive positions while updating active ones through actual computation
            if "selected_tokens" in kwargs:
                cache_kwargs["selected_tokens"] = kwargs["selected_tokens"]
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
    
    
class MoRLlamaDecoderLayer(nn.Module):
    """The Mixtures of Depth Block that dynamically which tokens to process in a block.
    Wraps around decoder block to allow for token dropping.
    """

    def __init__(self, config, block, cfg, capacity_factor=1.0, cap_warmup_step=0,):
        super().__init__()
        self.mor = True
        self.mor_type = "expert"
        
        self.config = config
        self.block = block
        self.cfg = cfg
        self.capacity_factor = capacity_factor
        self.cap_warmup_step = cap_warmup_step  # warm_up step for capacity_factor
        
        torch_dtype = get_torch_dtype(cfg)
        for blk in self.block:
            attn = MoRLlamaAttention(config, blk.self_attn.layer_idx).to(torch_dtype)
            # Reuse the projections to preserve their weights and parameter sharing.
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                setattr(attn, name, getattr(blk.self_attn, name))
            blk.self_attn = attn

        self.training_step = 0
        
        self.router_func = cfg.mor.expert.router_func
        self.alpha = cfg.mor.expert.alpha
        self.sampling = cfg.mor.expert.sampling
        
        torch_dtype = get_torch_dtype(cfg)
        
        if not cfg.mor.rand_router:
            self.mor_router = ROUTER_TYPES[cfg.mor.router_type](config).to(torch_dtype)

        if cfg.mor.expert.sampling == "aux_router":
            self.mlp_router = ROUTER_TYPES[cfg.mor.router_type](config).to(torch_dtype)
            
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="sum")
                                        
    def reset_parameters(self):
        for blk in self.block:
            blk.reset_parameters()

    def set_activation_checkpointing(self, strategy):
        for blk in self.block:
            blk.set_activation_checkpointing(strategy)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        prev_selected_tokens: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs]
    ):          
        total_x = x
        bs, seq_len, hidden_dim = total_x.shape
        
        if self.training:
            self.training_step += 1
            if self.cap_warmup_step > 0:
                step_ratio = min(1.0, self.training_step / self.cap_warmup_step)
                decay_factor = 0.5 * (1.0 + math.cos(math.pi * step_ratio))
            else:
                decay_factor = 0.0
            capacity_factor = self.capacity_factor + (1.0 - self.capacity_factor) * decay_factor
        else:
            capacity_factor = self.capacity_factor

        top_k = max(1, int(capacity_factor * seq_len))
        
        # gather the tokens that were processed in the previous layer
        if prev_selected_tokens is not None:
            x = torch.gather(x, 1, index=prev_selected_tokens.expand(-1, -1, hidden_dim))
        
        """STEP 1: get logits and top_k tokens"""
        if not self.cfg.mor.rand_router:
            _router_weights = self.mor_router(x / self.cfg.mor.temp) # [bs, seq_len, 1]
        
            if self.router_func is None:
                router_probs = router_weights = _router_weights
            elif self.router_func == "sigmoid":
                router_weights = F.sigmoid(_router_weights)
                router_probs = router_weights * self.cfg.mor.expert.alpha
            elif self.router_func == "tanh":
                router_weights = F.tanh(_router_weights)
                router_probs = router_weights * self.cfg.mor.expert.alpha
            else:
                raise NotImplementedError("Router function is not implemented")
            
        else:
            router_weights = _router_weights = torch.rand(bs, x.shape[1], 1, device=x.device, dtype=x.dtype)
            router_probs = router_weights * self.cfg.mor.expert.get("alpha", 0.1)
            
        weights, selected_tokens = torch.topk(router_probs, top_k, dim=1, sorted=False) # [bs, k, 1]
        # IMPORTANT: need to sort indices to keep causal order for those tokens that are processed in a block
        selected_tokens, index = torch.sort(selected_tokens, dim=1)
        weights = torch.gather(weights, dim=1, index=index)
        
        """STEP 2: expand indices to process batches with _reduced_ seqlen"""
        # We need to expand indices' dimensions from
        # [bs, k, 1] to [bs, k, hidden_size] for gathering
        indices_expanded = selected_tokens.expand(-1, -1, hidden_dim)
        top_k_tokens = torch.gather(x, dim=1, index=indices_expanded)
        
        sampling_loss = None
        sampling_acc = None
        topk_acc = None
        uniformity = None
        dead_token_seq = None
        
        if self.training and not self.cfg.mor.rand_router:
            targets = torch.zeros_like(router_probs, dtype=router_probs.dtype)
            src = torch.ones_like(selected_tokens, dtype=targets.dtype)
            targets.scatter_(1, selected_tokens, src)
            
            if self.sampling == "aux_router":
                logits = self.mlp_router(x.clone().detach())
                sampling_loss = self.bce_loss(logits.view(-1), targets.view(-1)) / (bs * logits.shape[1])
                prediction = (F.sigmoid(logits) >= 0.5)
                correct_predictions = (prediction == targets).view(-1)
                sampling_acc = correct_predictions.sum() / (bs * logits.shape[1])
                
                aux_router_topk = torch.topk(logits, top_k, dim=1, sorted=False)[1]
                topk_acc = torch.tensor(0.0, device=logits.device)
                for b in range(bs):
                    topk_acc += torch.isin(selected_tokens[b].view(-1), aux_router_topk[b].view(-1)).sum()
                topk_acc = topk_acc / (bs * top_k)
                
            elif self.sampling == "aux_loss":
                if self.router_func is None or self.router_func == "sigmoid":
                    sampling_loss = self.bce_loss(_router_weights.view(-1), targets.view(-1)) / (bs * router_weights.shape[1])
                    prediction = (router_weights >= 0.5)
                elif self.router_func == "tanh":
                    sampling_loss = self.bce_loss(_router_weights.view(-1), targets.view(-1)) / (bs * router_weights.shape[1])
                    prediction = (router_weights >= 0.)
                correct_predictions = (prediction == targets).view(-1)
                sampling_acc = correct_predictions.sum() / (bs * router_weights.shape[1])
                topk_acc = None
            
        """STEP 3: based on total seqlen, prepare input for block forward"""
        # recompute selected_tokens based on total tokens        
        if prev_selected_tokens is not None:
            selected_tokens = torch.gather(prev_selected_tokens, dim=1, index=selected_tokens)
            indices_expanded = selected_tokens.expand(-1, -1, hidden_dim)

        if "kv_sharing" in self.cfg and self.cfg.kv_sharing.enable and "update_cache" in self.cfg.kv_sharing and self.cfg.kv_sharing.update_cache:
            kwargs["selected_tokens"] = selected_tokens
        
        """STEP 4: forward block"""     
        if "kv_sharing" in self.cfg and self.cfg.kv_sharing.enable:
            top_k_tokens = total_x.clone()
                
            for blk in self.block:
                outputs = blk(
                    top_k_tokens,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs
                )
                top_k_tokens = outputs[0]
            top_k_tokens_processed = torch.gather(outputs[0], dim=1, index=indices_expanded)
            
        else:
            if attention_mask is not None: 
                if attention_mask.dim() == 4: 
                    row_indices = selected_tokens.unsqueeze(1).expand(bs, 1, top_k, attention_mask.shape[-1])  
                    mask_rows_selected = torch.gather(attention_mask, 2, row_indices)
                    col_indices = selected_tokens.unsqueeze(1).transpose(2, 3).expand(bs, 1, top_k, top_k)
                    attention_mask = torch.gather(mask_rows_selected, 3, col_indices)
                elif attention_mask.dim() == 2: # TODO
                    raise NotImplementedError("Attention mask is not implemented for inference phase of MoR")
                else: 
                    raise NotImplementedError("Attention mask has unexpected dimensions")
            
            if position_ids is not None: 
                position_ids = position_ids[:, :top_k]            
            if position_embeddings is not None:
                head_dim = position_embeddings[0].shape[-1]
                position_embeddings = tuple([torch.gather(emb.expand(bs, -1, -1), dim=1, index=selected_tokens.expand(-1, -1, head_dim)) 
                                                for emb in position_embeddings])
            if cache_position is not None:
                cache_position = torch.gather(cache_position.expand(bs, -1), dim=1, index=selected_tokens.squeeze(-1)) 
                
            for blk in self.block:
                outputs = blk(
                    top_k_tokens,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs
                )
                top_k_tokens = outputs[0]
            top_k_tokens_processed = outputs[0]
            
        """STEP 5: combine results"""
        _src = top_k_tokens_processed * weights if self.cfg.mor.expert.get("gating", "weighted") == "weighted" else top_k_tokens_processed
        total_x = torch.scatter_add(
            total_x,
            dim=1,
            index=indices_expanded,
            src=_src,
        )
        
        if self.training and "z_loss" in self.cfg.mor and self.cfg.mor.z_loss:            
            router_z_loss = torch.logsumexp(_router_weights, dim=-1)
            router_z_loss = torch.square(router_z_loss)
            router_z_loss = router_z_loss.mean() / (kwargs["num_items_in_batch"] / bs / seq_len)
        else:
            router_z_loss = None
                
        return MoRLayerOutputWithPast(
            hidden_state=total_x,
            attention_weights=outputs[1:],
            selected_tokens=selected_tokens,
            sampling_loss=sampling_loss if self.training else None,
            sampling_acc=sampling_acc if self.training else None,
            sampling_topk_acc=topk_acc if self.training else None,
            uniformity=uniformity if self.training else None,
            dead_token_seq=dead_token_seq if self.training else None,
            balancing_loss=None,
            balancing_ratio=None,
            router_z_loss=router_z_loss if self.training else None,
        )
