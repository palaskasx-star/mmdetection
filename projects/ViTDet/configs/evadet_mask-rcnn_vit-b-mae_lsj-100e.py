_base_ = [
    '../../../configs/_base_/models/mask-rcnn_r50_fpn.py',
    './lsj-100e_coco-instance.py',
]

custom_imports = dict(
    imports=['projects.ViTDet.vitdet.eva'], 
    allow_failed_imports=False
)

backbone_norm_cfg = dict(type='LN', requires_grad=True)
norm_cfg = dict(type='LN2d', requires_grad=True)
image_size = (1024, 1024)
batch_augments = [
    dict(type='BatchFixedSizePad', size=image_size, pad_mask=True)
]

# model settings
model = dict(
    data_preprocessor=dict(pad_size_divisor=32, batch_augments=batch_augments),
    backbone=dict(
        _delete_=True,
        type='EVA',
        img_size=1024,
        patch_size=16,
        
        # --- ARCHITECTURE DIMENSIONS (DINOv3 SMALL) ---
        embed_dim=384, 
        depth=12,
        num_heads=6,
        # ----------------------------------------------
        
        drop_path_rate=0.1,
        window_size=14,
        mlp_ratio=4,
        qkv_bias=True, 
        norm_cfg=backbone_norm_cfg,
        window_block_indexes=[0, 1, 3, 4, 6, 7, 9, 10],
        
        # --- DINOv3 SPECIFIC ---
        # REMOVED: use_rel_pos (RoPE is internal now, so this arg causes TypeError)
        init_values=1.0e-5, 
        
        init_cfg=dict(
            type='Pretrained', 
            checkpoint='/home/cpalaskas/diploma_project/mmdetection/projects/ViTDet/vitdet/small_dinov3_temp.pth'
        )),
    neck=dict(
        _delete_=True,
        type='SimpleFPN',
        
        # --- NECK DIMENSIONS (Updated for Small 384) ---
        # backbone_channel must match embed_dim
        backbone_channel=384, 
        # in_channels are the feature scales [1/4, 1/2, 1, 1] of backbone_channel
        in_channels=[96, 192, 384, 384], 
        # -----------------------------------------------
        
        out_channels=256,
        num_outs=5,
        norm_cfg=norm_cfg),
    rpn_head=dict(num_convs=2),
    roi_head=dict(
        bbox_head=dict(
            type='Shared4Conv1FCBBoxHead',
            conv_out_channels=256,
            norm_cfg=norm_cfg),
        mask_head=dict(norm_cfg=norm_cfg)))

custom_hooks = [dict(type='Fp16CompresssionHook')]
