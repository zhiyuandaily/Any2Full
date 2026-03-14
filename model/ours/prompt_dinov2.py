from functools import partial
import math
import logging
from typing import Sequence, Tuple, Union, Callable

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn.init import trunc_normal_


from model.ours.depth_anything_v2.dinov2 import DinoVisionTransformer,BlockChunk
from model.ours.depth_anything_v2.dinov2_layers import Mlp, PatchEmbed, SwiGLUFFNFused, MemEffAttention,PromptNestedTensorBlock as Block,NestedTensorBlock as Ori_Block
from model.ours.depth_anything_v2.dinov2_layers import PromptAttention

from model.ours.sparse_depth_embed import SparseDepthEmbed
logger = logging.getLogger("dinov2")

class PromptDinoVisionTransformer(DinoVisionTransformer):
    def __init__(self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=None,  # for layerscale: None or 0 => no layerscale
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=Block,
        ffn_layer="mlp",
        block_chunks=1,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
        blocks_to_take_list=[]
    ):
        super().__init__(img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            ffn_bias=ffn_bias,
            proj_bias=proj_bias,
            drop_path_rate=drop_path_rate,
            drop_path_uniform=drop_path_uniform,
            init_values=init_values,
            embed_layer=embed_layer,
            act_layer=act_layer,
            block_fn=block_fn,
            ffn_layer=ffn_layer,
            block_chunks=block_chunks,
            num_register_tokens=num_register_tokens,
            interpolate_antialias=interpolate_antialias,
            interpolate_offset=interpolate_offset,
        )
        
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        blocks_to_take = range(depth - blocks_to_take_list, depth) if isinstance(blocks_to_take_list, int) else blocks_to_take_list 
        
        if ffn_layer == "mlp":
            logger.info("using MLP layer as FFN")
            ffn_layer = Mlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            logger.info("using SwiGLU layer as FFN")
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":
            logger.info("using Identity layer as FFN")

            def f(*args, **kwargs):
                return nn.Identity()

            ffn_layer = f
        else:
            raise NotImplementedError
        
        blocks_list = [
            
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
            ) 
            if i in blocks_to_take[1:] else
            Ori_Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
                attn_class=MemEffAttention
            )
                
            for i in range(depth)
        ]
        
        if block_chunks > 0:
            self.chunked_blocks = True
            chunked_blocks = []
            chunksize = depth // block_chunks
            for i in range(0, depth, chunksize):
                # this is to keep the block index consistent if we chunk the block list
                chunked_blocks.append([nn.Identity()] * i + blocks_list[i : i + chunksize])
            self.blocks = nn.ModuleList([BlockChunk(p) for p in chunked_blocks])
        else:
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(blocks_list)
            
        
        self.prompt_depth_embedding = SparseDepthEmbed(embed_dim=embed_dim)
        
        self.prompt_detph_norm=norm_layer(embed_dim)
        
        
    def _get_intermediate_layers_not_chunked(self, x, prompt_ori, n=1 ):
        x = self.prepare_tokens_with_masks(x)
        
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        pre_x_list = []
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            if i in blocks_to_take:
                pre_x_list.append(x)
                x = blk(x)
                output.append(x)
            else:
                x = blk(x)
        for i in range(len(blocks_to_take)):
            
            if i==0:
                output[i]=[output[i],[output[i],[],[]]]
                continue
            
            def custom_forward(x, prompt):
                return self.blocks[blocks_to_take[i]](x, prompt, res_prompt=True)

            

            if i==1:
                prompt_value,prompt_mask=self.prompt_depth_embedding(prompt_ori,pre_x_list[i])           
                prompt=[prompt_value,prompt_mask]
                
                prompt = self.blocks[blocks_to_take[i]](pre_x_list[i], prompt, res_prompt=True)
            else:
                prompt = self.blocks[blocks_to_take[i]](pre_x_list[i], prompt, res_prompt=True)
            output[i]=[output[i],prompt]
            
            
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def _get_intermediate_layers_chunked(self, x, prompt, n=1):
        x = self.prepare_tokens_with_masks(x)
        output, i, total_block_len = [], 0, len(self.blocks[-1])
        # If n is an int, take the n last blocks. If it's a list, take them
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for block_chunk in self.blocks:
            for blk in block_chunk[i:]:  # Passing the nn.Identity()
                if i in blocks_to_take:
                    x, prompt = blk(x,prompt)
                else:
                    x = blk(x)
                if i in blocks_to_take:
                    output.append([x,prompt])
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output
    
    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        prompt: list,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        norm=True
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        if self.chunked_blocks:
            outputs = self._get_intermediate_layers_chunked(x, prompt, n)
        else:
            outputs = self._get_intermediate_layers_not_chunked(x, prompt, n)
        if norm:
            outputs = [[self.norm(out[0]),[self.prompt_detph_norm(out[1][0]),out[1][1]]] for out in outputs]
        class_tokens = [out[0][:, 0] for out in outputs]
        prompts = [[out[1][0][:, 1 + self.num_register_tokens:],out[1][1]]for out in outputs]
        outputs = [out[0][:, 1 + self.num_register_tokens:] for out in outputs]
        if reshape:
            B, _, w, h = x.shape
            outputs = [
                out.reshape(B, w // self.patch_size, h // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
            prompts = [
                prompt.reshape(B, w // self.patch_size, h // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for prompt in prompts
            ]
        if return_class_token:
            return tuple(zip(outputs, class_tokens,prompts))
        return tuple(outputs)
    
    
    
def vit_small(patch_size=16, num_register_tokens=0, **kwargs):
    model = PromptDinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=PromptAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_base(patch_size=16, num_register_tokens=0, **kwargs):
    model = PromptDinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=PromptAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_large(patch_size=16, num_register_tokens=0, **kwargs):
    model = PromptDinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=PromptAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_giant2(patch_size=16, num_register_tokens=0, **kwargs):
    """
    Close to ViT-giant, with embed-dim 1536 and 24 heads => embed-dim per head 64
    """
    model = PromptDinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=PromptAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def DINOv2(model_name,blocks_to_take_list):
    model_zoo = {
        "vits": vit_small, 
        "vitb": vit_base, 
        "vitl": vit_large, 
        "vitg": vit_giant2
    }
    
    return model_zoo[model_name](
        img_size=518,
        patch_size=14,
        init_values=1.0,
        ffn_layer="mlp" if model_name != "vitg" else "swiglufused",
        block_chunks=0,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
        blocks_to_take_list=blocks_to_take_list
    )
