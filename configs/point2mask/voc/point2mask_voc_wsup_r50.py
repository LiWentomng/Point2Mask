
_base_ = './voc_wsup_r50.py'
lr_config = dict(policy='step', step=[15, 20])
runner = dict(type='EpochBasedRunner', max_epochs=24)
work_dir = './work_dirs/voc_r50'


