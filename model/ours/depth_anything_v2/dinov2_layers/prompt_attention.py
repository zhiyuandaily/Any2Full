# added by zhiyuanzhou

from torch import Tensor
from torch import nn

import torch.nn.functional as F
import torch
from .attention  import Attention

  
class PromptAttention(Attention):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__(dim,
        num_heads,
        qkv_bias,
        proj_bias,
        attn_drop,
        proj_drop)
        self.prompt_depth_norm = nn.LayerNorm(dim)
        self.prompt_depth_bias =  nn.Linear(dim, dim, bias=qkv_bias)
        self.prompt_depth_proj = nn.Linear(dim, dim)
        
        out_dim = self.qkv.out_features
        qk_dim = 2 * out_dim // 3

        self.prompt_depth_qk = nn.Linear(self.qkv.in_features, qk_dim, bias=self.qkv.bias is not None)
        self.prompt_depth_qk.weight.data = self.qkv.weight.data[:qk_dim, :].clone()

        if self.qkv.bias is not None:
            self.prompt_depth_qk.bias.data = self.qkv.bias.data[:qk_dim].clone()

        
        
            
    def update_x(self, new_x: Tensor):
        self.new_x= new_x
        
    
    def forward(self, x_ori: Tensor, prompt=None) -> Tensor:
        # Fuse attention with prompt bias
        B, N, C = x_ori.shape
        
        if prompt is None:
            qkv = self.qkv(x_ori).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
            
            attn_logit = q @ k.transpose(-2, -1)
            attn = attn_logit.softmax(dim=-1)
            
            
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            self.update_x(x)
            return x 
        
        if prompt is not None:
            prompt_value,prompt_mask=prompt[0],prompt[1]
            previous_dtype=prompt_value.dtype

            
            prompt_qk = self.prompt_depth_qk(x_ori.detach()).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)#.to(torch.bfloat16)
            prompt_q, prompt_k= prompt_qk[0] , prompt_qk[1]
            prompt_v = self.prompt_depth_bias(self.prompt_depth_norm(prompt_value)).reshape(B, N,  self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)#.to(torch.bfloat16)
            B, H, N, D = prompt_v.shape
            prompt_mask = prompt_mask.view(B, 1,1, N).expand(B, self.num_heads, N, N)#.to(torch.bfloat16)
            
            
            
            
            # Use PyTorch 2.0+ SDPA
            prompt_output = F.scaled_dot_product_attention(prompt_q, prompt_k, prompt_v, attn_mask=torch.where(prompt_mask == 0, torch.full_like(prompt_mask, float('-inf')), 0.0), dropout_p=self.attn_drop_p, scale=self.scale).transpose(1, 2)
            
            prompt_output = prompt_output.reshape(B, N, C)
           
            prompt_output =self.prompt_depth_proj(prompt_output)
            new_prompt=self.proj_drop(prompt_output)
            
            
            
            new_prompt = [new_prompt, torch.ones_like(prompt[1])]
            
            self.update_x(new_x=None)
            
            return new_prompt
#         # self.prompt_depth_qkv.weight.data = self.qkv.weight.data.clone()
#         # if self.qkv.bias is not None:
#         #     self.prompt_depth_qkv.bias.data = self.qkv.bias.data.clone()
            
        
#         out_dim = self.qkv.out_features
#         qk_dim = 2 * out_dim // 3

#         self.prompt_depth_qk = nn.Linear(self.qkv.in_features, qk_dim, bias=self.qkv.bias is not None)
