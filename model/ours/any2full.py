import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import time
from model.ours.config import model_configs
from model.ours.logger import Log
import os
from pathlib import Path
from model.ours.depth_anything_v2.dpt import DPTHead
from model.ours.prompt_dinov2 import DINOv2 as PromptDINOv2


class Any2Full(nn.Module):
    

    def __init__(self,
                 encoder='vitl', da_ckpt_path='checkpoints/promptda_vitl.ckpt', args=None):
        super().__init__()
        self.args=args
        self.patch_size = 14  # patch size of the pretrained DINOv2 model
        self.use_bn = False
        self.use_clstoken = False
        
        self.output_act = 'None'

        model_config = model_configs[encoder]

        self.encoder = encoder
        self.model_config = model_config
        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11], 
            'vitl': [4, 11, 17, 23], 
            'vitg': [9, 19, 29, 39]
        }

        self.pretrained = PromptDINOv2(
            model_name=encoder,
            blocks_to_take_list=self.intermediate_layer_idx[encoder],
        )
        
        dim = self.pretrained.blocks[0].attn.qkv.in_features
        self.pretrained_prompt_depth_fusion=nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim*2, dim),
                nn.GELU(),
                nn.Linear(dim, dim)
            ) for i in range(len(self.intermediate_layer_idx[encoder])-1)
        ])
        
        self.pretrained_prompt_depth_scale=nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.Linear(dim, dim)
            ) for i in range(len(self.intermediate_layer_idx[encoder])-1)
        ])
        
        
        self.infer_time=0

                    
                    
        self.depth_head = DPTHead(
                                  in_channels=dim,
                                  features=model_config['features'],
                                  out_channels=model_config['out_channels'],
                                  use_bn=self.use_bn,
                                  use_clstoken=self.use_clstoken)
        
        if da_ckpt_path is not None:
            self.load_pretrainedDA(da_ckpt_path)
            print("Monodcular Depth Model-Encoder FREEZE !!")
            for name, var in self.pretrained.named_parameters():
                if not 'prompt_depth' in name: 
                    var.requires_grad = False
            print("Monodcular Depth Model-Decoder Bias Tuning !!")
            for name, var in self.depth_head.named_parameters():
                var.requires_grad = False
       

    

    def load_pretrainedDA(self, da_ckpt_path):
        if os.path.exists(da_ckpt_path):
            Log.info(f'Loading pretrained DepthAnything checkpoint from {da_ckpt_path}')
            checkpoint = torch.load(da_ckpt_path, map_location='cpu')
            if self.args.stage==1:
                # Report missing keys
                missing_keys = set(dict(self.named_parameters()).keys()) - set(checkpoint.keys())
                print("\nMissing keys:")
                for key in missing_keys:
                    print(key)
                self.load_state_dict(checkpoint, strict=False)
                for name, var in self.pretrained.named_parameters():
                    if 'prompt_depth_' in name:
                        base_name = name.replace('prompt_depth_', '')
                        if hasattr(self.pretrained, base_name):
                            base_var = getattr(self.pretrained, base_name)
                            var.data.copy_(base_var.data)
                            print(f"Copied data from {base_name} to {name}")
                            
            
            del checkpoint
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
           

            
        else:
            Log.warn(f'Checkpoint {da_ckpt_path} not found')
    

    def forward(self, x, prompt_depth=None):
        start_time=time.time()
        resize_mode='resize'
        if prompt_depth is None:
            prompt_depth=x['dep']
            rgb=x['rgb']
        else:
            prompt_depth=prompt_depth
            rgb=x
        if resize_mode =='pad':
            rgb, pad = self.pad_to_multiple(rgb,mode='replicate')
            prompt_depth, pad = self.pad_to_multiple(prompt_depth,mode='constant')
        elif resize_mode == 'resize':
            rgb, diff = self.resize_to_multiple(rgb, mode='bicubic')
            prompt_depth, diff = self.resize_to_multiple(prompt_depth,mode='nearest')
        
        # Treat prompt depth as disparity
        prompt_disparity= self.disparity_to_depth(prompt_depth)
        
        # Normalize input
        bias,scale=self.get_depth_bias_scale(prompt_disparity)
        prompt_disparity=(prompt_disparity-bias.view(-1, 1, 1, 1).detach())/(scale.view(-1, 1, 1, 1).detach())
        
        
        assert torch.isfinite(prompt_disparity).all(), "Input contains nan or inf"


        features= self.pretrained.get_intermediate_layers(rgb,prompt_disparity,self.intermediate_layer_idx[self.encoder], return_class_token=True)
        fused_features=[]
        for i, x in enumerate(features):
            if i==0:
                fused_x=x[0]
            else:
                prompt_v = x[-1][0]
                fused_x = (
                    (self.pretrained_prompt_depth_scale[i-1](prompt_v) + 1) * x[0]
                    + self.pretrained_prompt_depth_fusion[i-1](torch.cat((x[0], prompt_v), dim=-1))
                )
            if self.use_clstoken:
                fused_features.append([fused_x,x[1]])
            else:
                fused_features.append([fused_x])
            
        
        h, w = rgb.shape[-2:]
        patch_h, patch_w = h // self.patch_size, w // self.patch_size
        disparity_ori = None
        disparity_pre= self.depth_head(fused_features, patch_h, patch_w , return_feat=False)
       
        
        self.infer_time=self.infer_time+time.time()-start_time
        if self.args.init_scailing: 
            depth=self.disparity_to_depth(torch.clamp(self.init_scailing(disparity_pre,self.disparity_to_depth(prompt_depth)),min=1/self.args.max_depth))
            depth= torch.clamp(depth, min=self.args.min_depth, max=self.args.max_depth)
        else:
            bias_0,scale_0=self.get_depth_bias_scale(disparity_pre)
            disparity_pre_norm=(disparity_pre-bias_0.view(-1, 1, 1, 1).detach())/(scale_0.view(-1, 1, 1, 1).detach())
            depth = self.disparity_to_depth(torch.clamp(disparity_pre_norm*scale.view(-1, 1, 1, 1)+bias.view(-1, 1, 1, 1),min=1/self.args.max_depth))
            depth= torch.clamp(depth, min=self.args.min_depth, max=self.args.max_depth)
        
       
            
    


        
        if resize_mode =='pad':
            depth = self.unpad(depth, pad)
            disparity_pre = self.unpad(disparity_pre, pad)
        elif  resize_mode == 'resize':
            depth = self.unresize(depth, diff)
            disparity_pre = self.unresize(disparity_pre, diff)
        
        output = {'pred': depth,  'disparity_pre': disparity_pre, 'disparity_ori': None,'prompt_depth_features': None, 'guidance': None, 'confidence': None}
        
        return output
    
    
    def _concat(self, fd, fe, dim=1):
        _, _, Hd, Wd = fd.shape
        _, _, He, We = fe.shape

        if Hd > He:
            h = Hd - He
            fd = fd[:, :, :-h, :]

        if Wd > We:
            w = Wd - We
            fd = fd[:, :, :, :-w]

        f = torch.cat((fd, fe), dim=dim)

        return f
    
        
    def init_scailing(self, pred, sparse, align_points_num=1e10):
        depth=pred.clone().detach()
        for i in range(pred.shape[0]):
                target = sparse[i]
                idx_nnz = torch.nonzero(target.view(-1) >0.00001, as_tuple=False)
                 # Randomly sample up to 100 points
                num_points = idx_nnz.shape[0]
                if num_points > align_points_num:
                    # Sample indices
                    perm = torch.randperm(num_points, device=idx_nnz.device)
                    selected_indices = perm[:align_points_num]
                    idx_nnz = idx_nnz[selected_indices]
                B = target.view(-1)[idx_nnz]
                A = depth[i].view(-1)[idx_nnz]
                A =A+ torch.rand(*A.shape, device=A.device) * 1e-10
                num_dep = A.shape[0]
                A = torch.cat((A,torch.ones(num_dep,1).to(A)),dim=1)
                X = torch.pinverse(A) @ B
                X = X.to(pred)
                depth[i]  = pred[i] * X[0] + X[1]
        
        return depth
    
  

    

    def disparity_to_depth(self, disparity):
        disparity=torch.clamp(disparity, min=0)
        eps = 1e-8
        return torch.where(disparity > 0, 1.0 / (disparity + eps), torch.zeros_like(disparity))

    


    def print_variable_path(self, var, prefix=''):
        """
        Print trainable and non-trainable params that a variable flows through.

        :param var: variable to inspect
        :param prefix: indentation prefix
        """
        if hasattr(var, 'grad_fn'):
            print(f"{prefix}Variable: {var.grad_fn}")
            if hasattr(var.grad_fn, 'next_functions'):
                for next_fn in var.grad_fn.next_functions:
                    if next_fn[0] is not None:
                        self.print_variable_path(next_fn[0], prefix + '  ')
        
        for name, param in self.named_parameters():
            if param.is_leaf and param.grad_fn is var.grad_fn:
                print(f"{prefix}Trainable Parameter: {name}")
        
        for name, buf in self.named_buffers():
            if buf.is_leaf and buf.grad_fn is var.grad_fn:
                print(f"{prefix}Non-trainable Buffer: {name}")


    @torch.no_grad()
    def predict(self,
                image: torch.Tensor,
                prompt_depth: torch.Tensor):
        return self.forward(image, prompt_depth)

    def normalize(self,
                  prompt_depth: torch.Tensor):
        B, C, H, W = prompt_depth.shape
        min_val = torch.quantile(
            prompt_depth.reshape(B, -1), 0., dim=1, keepdim=True)[:, :, None, None]
        max_val = torch.quantile(
            prompt_depth.reshape(B, -1), 1., dim=1, keepdim=True)[:, :, None, None]
        prompt_depth = (prompt_depth - min_val) / (max_val - min_val+1e-8)
        return prompt_depth, min_val, max_val

    def denormalize(self,
                    depth: torch.Tensor,
                    min_val: torch.Tensor,
                    max_val: torch.Tensor):
        return depth * (max_val - min_val+1e-8) + min_val
    
    def pad_to_multiple(self, x, multiple_of=14, mode='constant'):
        _, _, h, w = x.shape  # (B, C, H, W)
        pad_h = (multiple_of - h % multiple_of) % multiple_of
        pad_w = (multiple_of - w % multiple_of) % multiple_of
        
        # Pad to multiple
        padded_x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode)
        
        return padded_x, (pad_h, pad_w)

    def unpad(self, x, pad):
        pad_h, pad_w = pad
        if pad_h == 0 and pad_w == 0:
            return x
        return x[..., :x.size(-2)-pad_h, :x.size(-1)-pad_w]
    
    def resize_to_multiple(self, x, multiple_of=14,mode='bilinear',resize_lower_size=518):
        B, _, h, w = x.shape  # (B, C, H, W)
        scale = max(resize_lower_size / h,resize_lower_size / w)
        # Compute target size
        new_h = int(((h*scale + multiple_of - 1) // multiple_of) * multiple_of)
        new_w = int(((w*scale + multiple_of - 1) // multiple_of) * multiple_of)
        
        # Return early if size unchanged
        if new_h == h and new_w == w:
            return x, (0, 0)
        if B == 1:
            # Resize
            if mode in ['linear', 'bilinear', 'bicubic', 'trilinear']:
                resized_x = F.interpolate(x, size=(new_h, new_w), mode=mode, align_corners=True)
            else:
                resized_x = F.interpolate(x, size=(new_h, new_w), mode=mode)
        else:
            results = []
            for b in range(B):
                single_x = x[b:b+1]
                # Resize
                if mode in ['linear', 'bilinear', 'bicubic', 'trilinear']:
                    single_resized_x = F.interpolate(single_x, size=(new_h, new_w), mode=mode, align_corners=True)
                else:
                    single_resized_x = F.interpolate(single_x, size=(new_h, new_w), mode=mode)
                results.append(single_resized_x)
            resized_x=torch.cat(results, dim=0)
            
        return resized_x, (new_h - h, new_w - w)

    def unresize(self, x, size_diff):
        h_diff, w_diff = size_diff
        if h_diff == 0 and w_diff == 0:
            return x
        
        _, _, h, w = x.shape
        return F.interpolate(x, size=(h - h_diff, w - w_diff), mode='bilinear', align_corners=True)

    def get_depth_bias_scale(self, prompt_depth):
        # prompt_depth shape: B x H x W
        B, C, H, W = prompt_depth.shape
        
        # Mask non-zero elements
        mask = prompt_depth != 0
        
        # Initialize min/max
        means = torch.zeros(B, device=prompt_depth.device, dtype=prompt_depth.dtype)
        stds = torch.zeros(B, device=prompt_depth.device, dtype=prompt_depth.dtype)
        
        for b in range(B):
            # Collect non-zero elements
            nonzero_elements = prompt_depth[b][mask[b]]
            
            if nonzero_elements.numel() > 0:
                # Compute mean and std
                mean = nonzero_elements.mean()
                if nonzero_elements.numel() > 1:
                    std = nonzero_elements.std()
                    if torch.isnan(std) or std==0:
                        print(f"Warning: Std is NaN for sample {b}")
                        std = 1.0
                else:
                    std = 1.0  # fallback for single element
                
                # Store stats
                means[b] = mean
                stds[b] = std
                
            else:
                # No valid elements; use defaults
                means[b] = 0
                stds[b] = 1
        
        return means, stds
