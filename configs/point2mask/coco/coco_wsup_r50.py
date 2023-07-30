
_base_ = [
    '../../_base_/datasets/coco_points.py',
    '../../_base_/default_runtime.py'
]
_dim_ = 256
_dim_half_ = _dim_//2
_feed_ratio_ = 4
_feed_dim_ = _feed_ratio_*_dim_
_num_levels_ = 4
model = dict(
    type='PanSeg',
    pretrained='https://download.pytorch.org/models/resnet50-11ad3fa6.pth',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='ChannelMapper',
        in_channels=[512, 1024, 2048],
        kernel_size=1,
        out_channels=_dim_,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=_num_levels_),
    bbox_head=dict(
        type='WsupPanformerHead',
        WARMUP_ITER=14659,
        OT_warmup_iters=60000,
        lambda_color_prior=3,
        lambda_diff_prob=1,
        lambda_diff_bond=0.1,
        lambda_diff_feat=0.1,
        expand_size=17,
        num_query=300,
        num_classes=133,  # 80+53
        num_things_classes=80,
        num_stuff_classes=53,
        in_channels=2048,
        sync_cls_avg_factor=True,
        as_two_stage=False,
        with_box_refine=True,
        use_low_level_edge=False,
        edge_model_path = '/path/to/model.yml.gz',
        transformer=dict(
            type='Deformable_Transformer',
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention',
                        embed_dims=_dim_,
                        num_levels=_num_levels_,
                         ),
                    feedforward_channels=_feed_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='DeformableDetrTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=_dim_,
                            num_levels=_num_levels_,
                        )
                    ],
                    feedforward_channels=_feed_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=_dim_half_,
            normalize=True,
            offset=-0.5),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0),
        loss_mask=dict(type='DiceLoss', loss_weight=2.0, activate=False),
        thing_transformer_head=dict(type='MaskHead',d_model=_dim_,nhead=8,num_decoder_layers=4),
        stuff_transformer_head=dict(type='MaskHead',d_model=_dim_,nhead=8,num_decoder_layers=6,self_attn=True),
        ),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0),
            ),
        assigner_with_mask=dict(
            type='HungarianAssigner_multi_info',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0),
            mask_cost=dict(type='DiceCost', weight=2.0),
            ),
        sampler =dict(type='PseudoSampler'),    
        sampler_with_mask =dict(type='PseudoSampler_segformer'),    
        ),
    test_cfg=dict(max_per_img=100))
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# train_pipeline
_img_scale = [(480, 1333), (512, 1333), (544, 1333),
    (576, 1333), (608, 1333), (640, 1333),
    (672, 1333), (704, 1333), (736, 1333),
    (768, 1333), (800, 1333)]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadPanopticAnnotations',
         with_bbox=True,
         with_mask=True,
         with_seg=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Resize',
         img_scale=_img_scale,
         multiscale_mode='value',
         keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=1),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_semantic_seg'])
]

# test_pipeline
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=_img_scale[-1][::-1],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=1),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
# optimizer
optimizer = dict(
    type='AdamW',
    lr=0.0002,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1)
        }))
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[18])
runner = dict(type='EpochBasedRunner', max_epochs=24)


custom_imports = dict(
    imports=["datasets", "easymd"],
    allow_failed_imports=False)

