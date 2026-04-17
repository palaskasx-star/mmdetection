import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmcv.cnn.bricks import DropPath
from mmengine.logging import MMLogger
from mmengine.model import BaseModule
from mmengine.runner.checkpoint import CheckpointLoader
from mmdet.registry import MODELS

# Import timm RoPE dependencies (ensure timm is installed)
from timm.layers import create_rope_embed, apply_rot_embed_cat

@MODELS.register_module(force=True)
class LN2d(nn.Module):
    """A LayerNorm variant for (B, C, H, W) shape."""
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

def window_partition(x, window_size):
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)

def window_unpartition(windows, window_size, pad_hw, hw):
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)
    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, rope=None):
        B, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        # Apply DINOv3 Rotary Position Embeddings (RoPE) if provided
        if rope is not None:
            # Reformat to match timm's apply_rot_embed_cat expectations: (B, num_heads, N, head_dim)
            q = q.view(-1, self.num_heads, H * W, self.head_dim)
            k = k.view(-1, self.num_heads, H * W, self.head_dim)

            # DINOv3 uses half-rotation layout
            q = apply_rot_embed_cat(q, rope, half=True)
            k = apply_rot_embed_cat(k, rope, half=True)

            # Revert shape back to (B * num_heads, N, head_dim)
            q = q.view(-1, H * W, self.head_dim)
            k = k.view(-1, H * W, self.head_dim)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_cfg=dict(type='GELU'), bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = build_activation_layer(act_cfg)
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class PatchEmbed(nn.Module):
    def __init__(self, kernel_size=(16, 16), stride=(16, 16), padding=(0, 0), in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = self.proj(x)
        return x.permute(0, 2, 3, 1)

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, drop_path=0.0, 
                 norm_cfg=dict(type='LN', eps=1e-6), act_cfg=dict(type='GELU'), 
                 window_size=0, init_values=None):
        super().__init__()
        self.norm1 = build_norm_layer(norm_cfg, dim)[1]
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = build_norm_layer(norm_cfg, dim)[1]
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_cfg=act_cfg)
        self.window_size = window_size

        # LayerScale implementation
        if init_values is not None and init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)))
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)))
        else:
            self.gamma_1 = None
            self.gamma_2 = None

    def forward(self, x, rope_2d=None):
        shortcut = x
        x = self.norm1(x)
        curr_rope = rope_2d

        # Window partition for both spatial features AND Rotary Positional Embeddings
        if self.window_size > 0:
            B_img, H, W, _ = x.shape  # Extract original batch size (e.g., 4)
            x, pad_hw = window_partition(x, self.window_size)
            
            if curr_rope is not None:
                # Partition the 2D RoPE grid: [1, H, W, head_dim] -> [num_windows, window_size, window_size, head_dim]
                curr_rope, _ = window_partition(curr_rope, self.window_size)
                
                # Expand/repeat along the batch dimension so it aligns perfectly with x's flattened windows
                # curr_rope shape becomes [B_img * num_windows, window_size, window_size, head_dim]
                curr_rope = curr_rope.repeat(B_img, 1, 1, 1)
                
                # Reshape to timm's expected (Batch, 1, seq_len, head_dim)
                curr_rope = curr_rope.view(-1, 1, self.window_size * self.window_size, curr_rope.shape[-1])
                
        elif curr_rope is not None:
            # If no windowing, PyTorch can broadcast (1, 1, H*W, D) with (B, num_heads, H*W, D) natively
            H, W = x.shape[1], x.shape[2]
            curr_rope = curr_rope.view(1, 1, H * W, curr_rope.shape[-1])

        x = self.attn(x, rope=curr_rope)

        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        # Apply LayerScale 1 (gamma_1) if it exists
        if self.gamma_1 is not None:
            x = shortcut + self.drop_path(self.gamma_1 * x)
        else:
            x = shortcut + self.drop_path(x)

        shortcut = x
        x = self.mlp(self.norm2(x))
        
        # Apply LayerScale 2 (gamma_2) if it exists
        if self.gamma_2 is not None:
            x = shortcut + self.drop_path(self.gamma_2 * x)
        else:
            x = shortcut + self.drop_path(x)

        return x
        
@MODELS.register_module()
class EVA(BaseModule):
    """DINOv3 backbone retaining original ViT arguments for drop-in compatibility."""
    def __init__(self, img_size=1024, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12,
                 mlp_ratio=4.0, qkv_bias=True, drop_path_rate=0.0, norm_cfg=dict(type='LN', eps=1e-6),
                 act_cfg=dict(type='GELU'), use_abs_pos=False, use_rel_pos=False, rel_pos_zero_init=True,
                 window_size=0, window_block_indexes=(0, 1, 3, 4, 6, 7, 9, 10),
                 pretrain_img_size=224, pretrain_use_cls_token=True, init_values=None, init_cfg=None, 
                 **kwargs): # Added init_values and **kwargs to absorb extra arguments
        super().__init__(init_cfg)

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size),
            in_chans=in_chans, embed_dim=embed_dim)

        # DINOv3 uses RoPE (Rotary Position Embeddings) instead of absolute/relative embeddings
        self.rope = create_rope_embed(
            rope_type='dinov3', 
            dim=embed_dim, 
            num_heads=num_heads, 
            temperature=100
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop_path=dpr[i],
                norm_cfg=norm_cfg, act_cfg=act_cfg, window_size=window_size if i in window_block_indexes else 0,
                init_values=init_values  # Passed to the blocks
            ) for i in range(depth)
        ])

    def init_weights(self):
        logger = MMLogger.get_current_instance()
        if self.init_cfg is None:
            logger.warning(f'No pre-trained weights for {self.__class__.__name__}, training from scratch')
        else:
            ckpt = CheckpointLoader.load_checkpoint(self.init_cfg.checkpoint, logger=logger, map_location='cpu')
            _state_dict = ckpt['model'] if 'model' in ckpt else ckpt
            self.load_state_dict(_state_dict, strict=False)

    def forward(self, x):
        x = self.patch_embed(x)
        B, H, W, C = x.shape

        # Generate absolute RoPE grid for the current dynamic H and W
        rot_pos_embed = self.rope.get_embed(shape=(H, W)) 
        
        # Reshape to a 2D spatial grid so we can window_partition it easily inside the blocks
        head_dim = rot_pos_embed.shape[-1]
        rope_2d = rot_pos_embed.view(1, H, W, head_dim)

        for blk in self.blocks:
            x = blk(x, rope_2d=rope_2d)

        return x.permute(0, 3, 1, 2)
