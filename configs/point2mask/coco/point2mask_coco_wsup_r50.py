
_base_ = './coco_wsup_r50.py'
lr_config = dict(policy='step', step=[8, 12])
runner = dict(type='EpochBasedRunner', max_epochs=15)

work_dir = './work_dirs/point2mask_coco_r50'
