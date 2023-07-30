
_base_ = './voc_wsup_r101.py'
lr_config = dict(policy='step', step=[15, 20])
runner = dict(type='EpochBasedRunner', max_epochs=24)

work_dir = './work_dirs/point2mask_voc_r101'


