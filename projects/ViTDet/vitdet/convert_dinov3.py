import torch
import argparse

def convert_dinov3_to_mmdet(timm_ckpt_path, out_path):
    print(f"Loading timm DINOv3 checkpoint from: {timm_ckpt_path}")
    timm_state = torch.load(timm_ckpt_path, map_location='cpu')
    
    # Extract actual weights depending on how the checkpoint was saved
    if 'model_ema' in timm_state:
        timm_state = timm_state['model_ema']
    elif 'model' in timm_state:
        timm_state = timm_state['model']
    elif 'state_dict' in timm_state:
        timm_state = timm_state['state_dict']

    mmdet_state = {}
    
    # Track which blocks we've fused QKV biases for
    fused_blocks = set()

    for k, v in timm_state.items():
        # 1. Drop tokens unused in dense prediction tasks (ViTDet style)
        if any(token in k for token in ['cls_token', 'reg_token', 'mask_token']):
            continue
        
        # Drop the final global norm since detection heads use their own normalization
        if k.startswith('norm.'): 
            continue

        # 2. Re-fuse QKV biases. timm splits q_bias and v_bias (with k_bias implicitly 0)
        if 'q_bias' in k or 'v_bias' in k:
            block_idx = k.split('.')[1]
            if block_idx not in fused_blocks:
                q_bias_key = f'blocks.{block_idx}.attn.q_bias'
                v_bias_key = f'blocks.{block_idx}.attn.v_bias'
                
                if q_bias_key in timm_state and v_bias_key in timm_state:
                    q_bias = timm_state[q_bias_key]
                    v_bias = timm_state[v_bias_key]
                    k_bias = torch.zeros_like(q_bias)  # DINOv3 enforces zero k_bias
                    
                    # Fuse them into one bias tensor for the MMDetection qkv nn.Linear layer
                    fused_bias = torch.cat([q_bias, k_bias, v_bias])
                    mmdet_state[f'blocks.{block_idx}.attn.qkv.bias'] = fused_bias
                    fused_blocks.add(block_idx)
            continue

        # Map all other matching parameters strictly 1-to-1
        mmdet_state[k] = v

    print(f"Fused QKV Biases for {len(fused_blocks)} blocks.")
    
    # Save formatted for MMDetection's CheckpointLoader
    torch.save({'model': mmdet_state}, out_path)
    print(f"Successfully converted DINOv3 weights and saved to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert timm DINOv3 weights to MMDetection backbone format.")
    parser.add_argument("--src", type=str, required=True, help="Path to original timm .pth file")
    parser.add_argument("--dest", type=str, required=True, help="Path to output the converted .pth file")
    args = parser.parse_args()

    convert_dinov3_to_mmdet(args.src, args.dest)
