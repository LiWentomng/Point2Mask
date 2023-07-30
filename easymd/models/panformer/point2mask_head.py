import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear, bias_init_with_prob, constant_init
from mmcv.runner import force_fp32
import mmcv
from mmdet.core import multi_apply
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models.builder import HEADS, build_loss
from mmdet.core import (bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh,
                        build_assigner, build_sampler, multi_apply,
                        reduce_mean)
from mmdet.datasets.coco_panoptic import INSTANCE_OFFSET
from mmdet.models.utils import build_transformer
from easymd.models.panformer import DETRHeadv2
from easymd.models.panformer.utils import _unfold_wo_center
from pydijkstra import dijkstra2d, dijkstra_image
from .utils import *
from scipy import ndimage


class _MaskTransformerWrapper(nn.Module):
    def __init__(self, num_layers, self_attn=False):
        super().__init__()
        cfg = dict(
            type='PanformerMaskDecoder',
            d_model=256,
            num_heads=8,
            num_layers=num_layers,
            self_attn=self_attn)
        self.model = build_transformer(cfg)

    def forward(self, memory, memory_mask, placeholder1, query, placeholder2, query_pos, hw_lvl):
        assert memory_mask.shape[0] == memory.shape[0], (memory_mask.shape, memory.shap)

        memory = memory.transpose(0, 1)
        query = query.transpose(0, 1)
        if query_pos is not None:
            query_pos = query_pos.transpose(0, 1)
        assert len(hw_lvl) == 4, (hw_lvl)
        hw_lvl = hw_lvl[:3]


        all_query, all_mask = self.model(
            memory=memory,
            # memory_pos=memory_pos,
            memory_mask=memory_mask,
            query=query,
            query_pos=query_pos,
            hw_lvl=hw_lvl)

        assert all_query[0].shape[1] == memory.shape[1], ([x.shape for x in all_query], memory.shape)
        all_query = [x.transpose(0, 1) for x in all_query]
        mask_final = all_mask[-1]
        inter_masks = all_mask[:-1]
        return mask_final, inter_masks, all_query


class _MemoryProj(nn.Module):
    def __init__(self, in_channels, out_channels, norm='l2'):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, 1))
        self.norm = norm

    def forward(self, memory, memory_pos, memory_mask, hw_lvl):
        bsz, nMem, embed_dims = memory.shape
        memory_with_pos = memory + memory_pos
        memory_with_pos = memory_with_pos.detach()

        hw_lvl = hw_lvl[:3]
        assert nMem == sum(h * w for h, w in hw_lvl), (memory.shape, hw_lvl)

        begin = 0
        memory_out = []
        for i, (h, w) in enumerate(hw_lvl):
            m2d = memory_with_pos[:, begin:begin + h * w, :].view(bsz, h, w, embed_dims)
            m2d = m2d.permute(0, 3, 1, 2).contiguous()
            begin = begin + h * w

            m2d = F.interpolate(m2d, hw_lvl[0], mode='bilinear', align_corners=False)
            memory_out.append(m2d)

        memory_out = sum(memory_out)
        out = self.layer(memory_out)
        if self.norm == 'l2':
            out = F.normalize(out, p=2, dim=1)
        elif self.norm is None:
            pass
        elif self.norm == 'tanh':
            out = F.tanh(out)
        else:
            raise ValueError(self.norm)
        return out


class SinkhornDistance(torch.nn.Module):
    """
        Sinkhorn Knopp Algorithm
        Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
    """

    def __init__(self, eps=1e-3, max_iter=80):
        super(SinkhornDistance, self).__init__()
        self.epsilon = eps
        self.max_iter = max_iter

    def forward(self, mu, nu, cost):
        u = torch.ones_like(mu)
        v = torch.ones_like(nu)

        # Sinkhorn iterations
        for i in range(self.max_iter):
            v = self.epsilon * (torch.log(nu + 1e-8) - torch.logsumexp(self.log_mul(cost, u, v).transpose(-2, -1), dim=-1)) + v
            u = self.epsilon * (torch.log(mu + 1e-8) - torch.logsumexp(self.log_mul(cost, u, v), dim=-1)) + u

        # Transport plan: pi = diag(u)*K*diag(v)
        pi = torch.exp(self.log_mul(cost, u, v)).detach()

        # Sinkhorn distance
        cost = torch.sum(pi * cost, dim=(-2, -1))
        return cost, pi


    def log_mul(self, C, u, v):
        '''
        "M_ij = (-c_ij + u_i + v_j) / epsilon"
        '''
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.epsilon


@HEADS.register_module()
class WsupPanformerHead(DETRHeadv2):
    """
    Head of Panoptic SegFormer

    Code is modified from the `official github repo
    <https://github.com/open-mmlab/mmdetection>`_.

    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
    """

    def __init__(
            self,
            *args,
            with_box_refine=False,
            as_two_stage=False,
            transformer=None,
            quality_threshold_things=0.25,
            quality_threshold_stuff=0.25,
            overlap_threshold_things=0.4,
            overlap_threshold_stuff=0.2,
            use_argmax=False,
            datasets='coco',  # MDS
            thing_transformer_head=dict(
                type='TransformerHead',  # mask decoder for things
                d_model=256,
                nhead=8,
                num_decoder_layers=6),
            stuff_transformer_head=dict(
                type='TransformerHead',  # mask decoder for stuff
                d_model=256,
                nhead=8,
                num_decoder_layers=6),
            loss_mask=dict(type='DiceLoss', weight=2.0),
            train_cfg=dict(
                assigner=dict(type='HungarianAssigner',
                              cls_cost=dict(type='ClassificationCost',
                                            weight=1.),
                              reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                              iou_cost=dict(type='IoUCost',
                                            iou_mode='giou',
                                            weight=2.0)),
                sampler=dict(type='PseudoSampler'),
            ),
            **kwargs):
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        self.quality_threshold_things = quality_threshold_things
        self.quality_threshold_stuff = quality_threshold_stuff
        self.overlap_threshold_things = overlap_threshold_things
        self.overlap_threshold_stuff = overlap_threshold_stuff
        self.use_argmax = use_argmax
        self.datasets = datasets
        self.fp16_enabled = False

        self.WARMUP_ITER = kwargs['WARMUP_ITER']
        self.lambda_diff_prob = kwargs['lambda_diff_prob']
        self.lambda_diff_bond = kwargs['lambda_diff_bond']
        self.lambda_diff_feat = kwargs['lambda_diff_feat']
        self.lambda_color_prior = kwargs['lambda_color_prior']
        self.EXPAND_SIZE = kwargs['expand_size']
        self._low_level_edge = kwargs['use_low_level_edge']
        if self._low_level_edge:
            self.edge_model_path = kwargs['edge_model_path']

        # MDS: id_and_category_maps is the category_dict
        if datasets == 'coco':
            from easymd.datasets.coco_panoptic import id_and_category_maps
            self.cat_dict = id_and_category_maps
        else:
            self.cat_dict = None
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        self.num_dec_things = thing_transformer_head['num_decoder_layers']
        self.num_dec_stuff = stuff_transformer_head['num_decoder_layers']
        super(WsupPanformerHead, self).__init__(*args,
                                                transformer=transformer,
                                                train_cfg=train_cfg,
                                                **kwargs)
        if train_cfg:
            sampler_cfg = train_cfg['sampler_with_mask']
            self.sampler_with_mask = build_sampler(sampler_cfg, context=self)
            assigner_cfg = train_cfg['assigner_with_mask']
            self.assigner_with_mask = build_assigner(assigner_cfg)
            self.assigner_filter = build_assigner(
                dict(
                    type='HungarianAssigner_filter',
                    cls_cost=dict(type='FocalLossCost', weight=2.0),
                    reg_cost=dict(type='BBoxL1Cost',
                                  weight=5.0,
                                  box_format='xywh'),
                    iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0),
                    max_pos=3), )

        self.loss_mask = build_loss(loss_mask)
        self.things_mask_head = _MaskTransformerWrapper(4, False)
        self.stuff_mask_head = _MaskTransformerWrapper(6, True)
        self.semantic_mask_head = _MaskTransformerWrapper(6, True)
        num_classes = self.num_things_classes + self.num_stuff_classes
        self.num_classes = num_classes
        self.semantic_proj = nn.Conv2d(num_classes, num_classes, 1, groups=num_classes)
        self.count = 0
        self.tmp_state = {}
        self.warmup_niter = 0

        self.memory_proj = _MemoryProj(self.embed_dims, 128, 'l2')
        self.boundary_proj = _MemoryProj(self.embed_dims, 1, None)

        self.register_buffer("_iter", torch.zeros([1]))
        self.ot_warmup_iters = kwargs['OT_warmup_iters']

        self.sinkhorn = SinkhornDistance()


    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""

        fc_cls = Linear(self.embed_dims, self.cls_out_channels)
        fc_cls_stuff = Linear(self.embed_dims, 1)
        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, 4))
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_pred = (self.transformer.decoder.num_layers + 1) if \
            self.as_two_stage else self.transformer.decoder.num_layers

        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])
        if not self.as_two_stage:
            self.query_embedding = nn.Embedding(self.num_query,
                                                self.embed_dims * 2)
        self.stuff_query = nn.Embedding(self.num_stuff_classes,
                                        self.embed_dims * 2)
        self.semantic_query = nn.Embedding(self.num_things_classes + self.num_stuff_classes,
                                           self.embed_dims * 2)
        self.cls_stuff_branches = _get_clones(fc_cls_stuff, self.num_dec_stuff)  # used in mask deocder
        self.cls_semantic_branches = _get_clones(fc_cls_stuff, self.num_dec_stuff)


    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m.bias, bias_init)
            for m in self.cls_stuff_branches:
                nn.init.constant_(m.bias, bias_init)
        for m in self.reg_branches:
            constant_init(m[-1], 0, bias=0)
        nn.init.constant_(self.reg_branches[0][-1].bias.data[2:], -2.0)

        if self.as_two_stage:
            for m in self.reg_branches:
                nn.init.constant_(m[-1].bias.data[2:], 0.0)

        nn.init.constant_(self.semantic_proj.weight, 1.0)
        nn.init.constant_(self.semantic_proj.bias, 0.0)


    @force_fp32(apply_to=('mlvl_feats',))
    def forward(self, mlvl_feats, img_metas=None):
        """Forward function.

        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 4D-tensor with shape
                (N, C, H, W).
            img_metas (list[dict]): List of image information.

        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, h). \
                Shape [nb_dec, bs, num_query, 4].
            enc_outputs_class (Tensor): The score of each point on encode \
                feature map, has shape (N, h*w, num_class). Only when \
                as_two_stage is True it would be returned, otherwise \
                `None` would be returned.
            enc_outputs_coord (Tensor): The proposal generate from the \
                encode feature map, has shape (N, h*w, 4). Only when \
                as_two_stage is True it would be returned, otherwise \
                `None` would be returned.
        """

        batch_size = mlvl_feats[0].size(0)

        input_img_h, input_img_w = img_metas[0]['batch_input_shape']

        img_masks = mlvl_feats[0].new_ones(
            (batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]['img_shape']
            img_masks[img_id, :img_h, :img_w] = 0

        hw_lvl = [feat_lvl.shape[-2:] for feat_lvl in mlvl_feats]
        mlvl_masks = []
        mlvl_positional_encodings = []
        for feat in mlvl_feats:
            mlvl_masks.append(
                F.interpolate(img_masks[None],
                              size=feat.shape[-2:]).to(torch.bool).squeeze(0))
            mlvl_positional_encodings.append(
                self.positional_encoding(mlvl_masks[-1]))

        query_embeds = None
        if not self.as_two_stage:
            query_embeds = self.query_embedding.weight
        (memory, memory_pos, memory_mask, query_pos), hs, init_reference, inter_references, \
        enc_outputs_class, enc_outputs_coord = self.transformer(
            mlvl_feats,
            mlvl_masks,
            query_embeds,
            mlvl_positional_encodings,
            reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
            cls_branches=self.cls_branches if self.as_two_stage else None  # noqa:E501
        )

        memory = memory.permute(1, 0, 2)
        query = hs[-1].permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        memory_pos = memory_pos.permute(1, 0, 2)

        len_last_feat = hw_lvl[-1][0] * hw_lvl[-1][1]

        # we should feed these to mask deocder.
        args_tuple = (memory[:, :-len_last_feat, :],
                      memory_mask[:, :-len_last_feat],
                      memory_pos[:, :-len_last_feat, :], query, None,
                      query_pos, hw_lvl)

        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])

            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)

        if self.as_two_stage:
            return outputs_classes, outputs_coords, \
                   enc_outputs_class, \
                   enc_outputs_coord.sigmoid(), args_tuple, reference
        else:
            return outputs_classes, outputs_coords, \
                   None, None, args_tuple, reference

    @force_fp32(apply_to=('all_cls_scores_list', 'all_bbox_preds_list',
                          'args_tuple', 'reference'))
    def loss(
            self,
            all_cls_scores,
            all_bbox_preds,
            enc_cls_scores,
            enc_bbox_preds,
            args_tuple,
            reference,
            gt_bboxes_list,
            gt_labels_list,
            gt_masks_list=None,
            gt_semantics_list=None,
            img_metas=None,
            gt_bboxes_ignore=None,
    ):
        """Loss function.

        Args:
            all_cls_scores (Tensor): Classification score of all
                decoder layers, has shape
                [nb_dec, bs, num_query, cls_out_channels].
            all_bbox_preds (Tensor): Sigmoid regression
                outputs of all decode layers. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and shape
                [nb_dec, bs, num_query, 4].
            enc_cls_scores (Tensor): Classification scores of
                points on encode feature map , has shape
                (N, h*w, num_classes). Only be passed when as_two_stage is
                True, otherwise is None.
            enc_bbox_preds (Tensor): Regression results of each points
                on the encode feature map, has shape (N, h*w, 4). Only be
                passed when as_two_stage is True, otherwise is None.
            args_tuple (Tuple) several args
            reference (Tensor) reference from location decoder
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        self._iter += 1

        loss_dict = {}

        # batch-size first
        memory, memory_mask, memory_pos, query, _, query_pos, hw_lvl = args_tuple
        bsz, _, embed_dims = memory_pos.shape

        loss_semantic, semantic_info = self.loss_semantic(args_tuple, gt_semantics_list)
        loss_dict.update(loss_semantic)

        memory_proj = self.memory_proj(memory, memory_pos, memory_mask, hw_lvl)  # [bsz, dim, h, w]
        boundary_proj = self.boundary_proj(memory, memory_pos, memory_mask, hw_lvl).squeeze(1)


        pl_labels_list, pl_bboxes_list, pl_masks_list, pl_semantics_list = \
            self.get_pseudo_mask_label(*semantic_info, gt_labels_list, gt_masks_list, boundary_proj,
                                           memory_proj, img_metas, self._iter)

        # location decoder loss
        for i, (cls_scores, bbox_preds) in enumerate(zip(all_cls_scores, all_bbox_preds)):
            if i == len(all_cls_scores) - 1:
                continue
            loss_cls, loss_bbox, loss_iou = self.loss_single(
                cls_scores, bbox_preds,
                pl_bboxes_list, pl_labels_list,
                img_metas, gt_bboxes_ignore, )
            loss_dict.update({
                f'd{i}.loss_cls': loss_cls,
                f'd{i}.loss_bbox': loss_bbox,
                f'd{i}.loss_iou': loss_iou, })

        if enc_cls_scores is not None:
            binary_labels_list = [torch.zeros_like(x) for x in pl_labels_list]
            enc_losses_cls, enc_losses_bbox, enc_losses_iou = self.loss_single(
                enc_cls_scores, enc_bbox_preds,
                pl_bboxes_list, binary_labels_list,
                img_metas, gt_bboxes_ignore)
            loss_dict.update({
                'enc_loss_cls': enc_losses_cls,
                'enc_loss_bbox': enc_losses_bbox,
                'enc_loss_iou': enc_losses_iou})

        # mask decoder loss
        loss_mask_decoder, thing_ratio, stuff_ratio, all_th_masks, all_st_masks = self.loss_single_panoptic_simplified(
            all_cls_scores[-1],
            all_bbox_preds[-1],
            args_tuple,
            pl_bboxes_list,
            pl_labels_list,
            pl_masks_list,
            pl_semantics_list,
            img_metas,
            gt_bboxes_ignore)
        loss_dict.update(loss_mask_decoder)

        for k in loss_dict.keys():
            ratio = stuff_ratio if 'st' in k else thing_ratio
            loss_dict[k] = loss_dict[k] * ratio

        # feature metric distance loss
        pl_stuff_masks = F.one_hot(pl_semantics_list.long(), 256)[...,
                         self.num_things_classes:self.num_things_classes + self.num_stuff_classes].permute(0, 3, 1, 2).to(pl_masks_list[0])

        loss_cl = self.loss_metric(memory_proj, gt_labels_list, gt_masks_list, gt_semantics_list, pl_masks_list,
                                   pl_stuff_masks, img_metas)
        loss_dict.update(loss_cl)

        loss_bond = self.loss_boundary(boundary_proj, pl_masks_list, pl_semantics_list, img_metas)
        loss_dict.update(loss_bond)

        # tmp states
        self.tmp_state.update({
            'gt_bboxes_list': pl_bboxes_list,
            'gt_labels_list': pl_labels_list,
            'gt_masks_list': pl_masks_list,
            'gt_semantics_list': pl_semantics_list,
            'th_cls_scores': all_cls_scores[-1],
            'th_bbox_preds': all_bbox_preds[-1],
        })

        # TODO: hard code for pseudo-label warmup
        warmup = max(min(float(self.warmup_niter) / self.WARMUP_ITER, 1), 0)
        self.warmup_niter += 1
        k_contains = lambda k: any([x in k for x in ['sem', 'loss_cl']])
        loss_dict = {k: v * (1 if k_contains(k) else warmup) for k, v in loss_dict.items()}

        return loss_dict


    "loss for categoty-wise sematic map"
    def loss_semantic(self, args_tuple, gt_semantics_list):

        memory, memory_mask, memory_pos, query, _, query_pos, hw_lvl = args_tuple
        bsz, _, embed_dims = query.shape

        # get predictions of semantic masks
        sem_query, sem_query_pos = torch.split(
            self.semantic_query.weight[None].expand(bsz, -1, -1),
            embed_dims, -1)
        mask_sem, mask_inter_sem, query_inter_sem = self.semantic_mask_head(
            memory, memory_mask, None, sem_query, None, sem_query_pos, hw_lvl=hw_lvl)
        all_sem_masks = []
        for sem_masks in mask_inter_sem + [mask_sem]:
            sem_masks = sem_masks.squeeze(-1)
            sem_masks = sem_masks.reshape(-1, *hw_lvl[0])
            all_sem_masks.append(sem_masks)
        all_sem_masks = torch.stack(all_sem_masks)  # [n_dec, bsz*cls, h, w]

        n_dec, _, h, w = all_sem_masks.shape
        all_sem_masks_proj = self.semantic_proj(all_sem_masks.view(n_dec * bsz, -1, h, w)).view(n_dec, -1, h, w)

        # get predictions of semantic cls
        all_sem_cls = []
        for i, query_sem in enumerate(query_inter_sem):
            sem_cls = self.cls_semantic_branches[i](query_sem).view(-1, 1)
            all_sem_cls.append(sem_cls)
        all_sem_cls = torch.stack(all_sem_cls)  # [n_dec, bsz*cls, 1]

        # target mask
        target_mask = F.one_hot(gt_semantics_list.long(), max(256, gt_semantics_list.max() + 1))
        num_classes = self.num_things_classes + self.num_stuff_classes
        target_mask = target_mask[..., :num_classes]
        target_mask = target_mask.permute(0, 3, 1, 2).float()  # [bsz, cls, h, w]
        target_mask = expand_target_masks(target_mask, self.EXPAND_SIZE)
        target_mask = target_mask.flatten(0, 1)  # [bsz*cls, h, w]
        target_mask_weight = target_mask.max(2)[0].max(1)[0]  # [bsz*cls]
        target_cls = (1 - target_mask_weight).long()

        # NOTE: dice loss is unavailable because of partial annotations
        loss_dict = {}
        for i, sem_cls in enumerate(all_sem_cls):
            loss_cls = self.loss_cls(
                sem_cls,
                target_cls,
                avg_factor=target_mask_weight.sum())
            loss_dict[f'd{i}.loss_sem_cls'] = loss_cls

        all_sem_cls_prob = all_sem_cls[..., None].sigmoid()  # [n_dec, bsz*cls, 1, 1]
        all_sem_masks_logit_scaled = all_sem_masks_proj * all_sem_cls_prob
        all_sem_masks_logit = F.interpolate(all_sem_masks_logit_scaled, target_mask.shape[-2:],
                                            mode='bilinear', align_corners=False)

        images = self.tmp_state['img']

        H, W = all_sem_masks_logit_scaled.shape[-2:]
        images_ = F.interpolate(images, (H, W), mode='bilinear', align_corners=False)
        target_mask_small = F.interpolate(target_mask[None], (H, W), mode='nearest').view(bsz, -1, H, W)

        for i, sem_mask_logit in enumerate(all_sem_masks_logit_scaled):
            loss_mask = partial_cross_entropy_loss(
                sem_mask_logit.view(bsz, -1, H, W),
                target_mask_small)
            loss_sem_lab_color = color_prior_loss(
                sem_mask_logit.view(bsz, -1, H, W),
                images_)

            loss_sem_rgb_color = rgd_semantic_loss(
                sem_mask_logit.view(bsz, -1, H, W),
                target_mask_small,
                images_)

            loss_dict[f'd{i}.loss_sem_mask'] = loss_mask
            loss_dict[f'd{i}.loss_sem_color'] = loss_sem_lab_color * self.lambda_color_prior
            loss_dict[f'd{i}.loss_tree_color'] = loss_sem_rgb_color * self.lambda_color_prior


        H2, W2 = all_sem_masks_logit.shape[-2:]
        self.tmp_state.update({
            'semantic_pred_logit': all_sem_masks_logit[-1].view(bsz, -1, H2, W2),
            'semantic_pred_target': target_mask.view(bsz, -1, H2, W2)
        })

        semantic_info = (
            all_sem_masks_logit[-1].view(bsz, -1, H2, W2),
            target_mask.view(bsz, -1, H2, W2),
            target_mask_weight.view(bsz, -1),
        )
        return loss_dict, semantic_info

    "loss for manifold features"
    def loss_metric(self,
                    memory_proj,
                    gt_labels_list,
                    gt_masks_list,
                    gt_semantics_list,
                    pred_masks_list,
                    pred_stuff_masks_list,
                    img_metas,
                    use_stuff=True):

        # points' masks, including stuff classes
        if use_stuff:
            gt_stuff_masks = F.one_hot(gt_semantics_list.long(), 256)[...,
                             self.num_things_classes:self.num_things_classes + self.num_stuff_classes]\
                                .permute(0, 3, 1, 2).to(gt_masks_list[0])
            gt_masks_list = [torch.cat([th_masks, st_masks]) for th_masks, st_masks in
                             zip(gt_masks_list, gt_stuff_masks)]
            pred_masks_list = [torch.cat([th_masks, st_masks]) for th_masks, st_masks in
                               zip(pred_masks_list, pred_stuff_masks_list)]

        # two sizes
        h, w = memory_proj.shape[-2:]
        H, W = gt_masks_list[0].shape[-2:]
        coord_factor = memory_proj.new_tensor([1, h / H, w / W]).view(1, -1)

        # scale = 1. / memory_proj.shape[1]**.5
        scale = 0.07

        # compute loss
        loss_cl = []
        for i, gt_masks in enumerate(gt_masks_list):
            coords = torch.nonzero(gt_masks)  # [N, 3], {insID, ih, iw}
            coords = (coords * coord_factor).long()

            query = memory_proj[i][:, coords[:, 1], coords[:, 2]]  # [dim, N]
            pred_masks = \
            F.interpolate(pred_masks_list[i][None].detach().float(), (h, w), mode='bilinear', align_corners=False)[0]
            reference = torch.einsum('dhw,nhw->dn', memory_proj[i], pred_masks) / pred_masks.sum((1, 2)).clip(min=1)[None]  # [dim, n]

            dot = (query.T @ reference) / scale  # [N, n]

            ins_indices = coords[:, 0]  # [N,], value in {0, 1, ..., n-1}
            pos_mask = torch.zeros_like(dot)
            pos_mask = torch.zeros_like(dot)
            pos_mask[torch.arange(dot.shape[0], device=dot.device), ins_indices] = 1
            loss_cl_i = - (torch.log_softmax(dot, 1) * pos_mask).sum(1)
            pos_mask_valid = (pos_mask.sum(1) < pos_mask.shape[1]).float()
            loss_cl.append((loss_cl_i * pos_mask_valid).sum() / pos_mask_valid.sum().clip(min=1))
        loss_cl = sum(loss_cl) / len(loss_cl)
        loss_dict = dict(
            loss_cl=loss_cl
        )
        self.tmp_state.update(dict(memory_proj=memory_proj))
        return loss_dict

    "The loss for high-level boundary map"
    def loss_boundary(self, boundary_proj, pl_masks_list, pl_stuff_masks, img_metas):

        bs = len(pl_masks_list)
        images = self.tmp_state['img']
        H, W = images.shape[-2:]
        boundary_proj = F.interpolate(boundary_proj[None], (H, W), mode='bilinear', align_corners=False)[0]
        boundary_proj = torch.sigmoid(boundary_proj)

        losses = {}
        loss_bond = 0

        for k in range(bs):
            t_msk = pl_masks_list[k]
            s_msk = pl_stuff_masks[k]

            # thing
            neighbor_t_msk = _unfold_wo_center(t_msk.float().unsqueeze(0), 3, 1).squeeze(0).permute(1, 0, 2, 3)
            t_valid_label = torch.logical_and(torch.gt(t_msk, 0), torch.gt(neighbor_t_msk, 0))
            t_equal_label = torch.eq(t_msk, neighbor_t_msk)
            t_pos_affinity_label_all = torch.logical_and(t_equal_label, t_valid_label).float()
            t_pos_affinity_label = torch.sum(t_pos_affinity_label_all, dim=1)  # [8,H,W]

            t_neg_affinity_label_all = torch.logical_not(t_equal_label)
            t_neg_affinity_label = torch.sum(t_neg_affinity_label_all, dim=1)  # [8,H,W]
            t_neg_affinity_label = torch.clamp(t_neg_affinity_label, 0, 1)

            # stuff
            neighbor_s_msk = _unfold_wo_center(s_msk.float().unsqueeze(0).unsqueeze(0), 3, 1).squeeze(0).squeeze(0) 
            s_valid_label_1 = torch.logical_and(torch.ge(s_msk, self.num_things_classes),
                                                torch.ge(neighbor_s_msk, self.num_things_classes))
            s_valid_label_2 = torch.logical_and(torch.lt(s_msk, 255), torch.lt(neighbor_s_msk, 255))
            s_valid_label = torch.logical_and(s_valid_label_1, s_valid_label_2)
            s_equal_label = torch.eq(s_msk, neighbor_s_msk)
            s_pos_affinity_label = torch.logical_and(s_valid_label, s_equal_label).float()

            s_neg_affinity_label = torch.logical_and(torch.logical_not(s_equal_label), s_valid_label_2)

            neg_affinity_label = torch.logical_or(t_neg_affinity_label, s_neg_affinity_label).float()

            # affinity
            neighbor_boundary_proj = _unfold_wo_center(boundary_proj[k].unsqueeze(0).unsqueeze(0), 3, 1).squeeze()  # [8,H,W]
            max_boundary = torch.maximum(neighbor_boundary_proj, boundary_proj[k])
            aff = 1 - max_boundary

            pos_aff_loss = (-1) * torch.log(aff + 1e-5)
            neg_aff_loss = (-1) * torch.log(1. + 1e-5 - aff)

            t_pos_aff_loss = torch.sum(t_pos_affinity_label * pos_aff_loss) / (torch.sum(t_pos_affinity_label) + 1e-5)
            s_pos_aff_loss = torch.sum(s_pos_affinity_label * pos_aff_loss) / (torch.sum(s_pos_affinity_label) + 1e-5)

            pos_aff_loss = t_pos_aff_loss / 2 + s_pos_aff_loss / 2
            neg_aff_loss = torch.sum(neg_affinity_label * neg_aff_loss) / (torch.sum(neg_affinity_label) + 1e-5)

            loss_bond += (pos_aff_loss + neg_aff_loss) / 2

        losses[f'loss_bnd'] = loss_bond / bs

        return losses


    "Obtain the pseudo mask label by Optimal Transport"
    @torch.no_grad()
    def get_pseudo_mask_label(self,
                              pred_semantic_masks,
                              sup_semantic_masks,
                              cls_labels,
                              gt_labels_list,
                              gt_masks_list,
                              boundary_proj,
                              memory_proj,
                              img_metas,
                              train_iter):
        """
        Input:
            pred_semantic_masks:     [bsz, c_th+c_st, h, w], logits before softmax.
            sup_semantic_masks:       [bsz, c_th+c_st, h, w], in value {0, 1}, semantic gts, typically point-level labels.
            cls_labels:     [bsz, c_th+c_st], in value {0, 1}, indicating whether the class exits.
            gt_labels_list: List<bsz>[ [n,] ], in value {0, 1, ..., C-1}, class label of each thing class, 'n' is the number of instances in the sample.
            gt_masks_list:  List<bsz>[ [n, h, w] ], in value {0, 1}, 'n' is the number of instances in the sample.
        """

        # semantic results
        pred_mask_probs = torch.maximum(pred_semantic_masks.softmax(1) * cls_labels[:, :, None, None],
                                        sup_semantic_masks)
        out_semantic_probs, out_semantics_list = pred_mask_probs.max(1)
        out_semantics_list[out_semantic_probs < 0.5] = 255
        pl_semantics_list = out_semantics_list

        H, W = gt_masks_list[0].shape[-2:]
        downsample = 16
        h, w = H // downsample, W // downsample
        dfactor = pred_mask_probs.new_tensor([1, 1. / downsample, 1. / downsample])
        downrate = 1. / downsample

        resize = lambda x: F.interpolate(x, (h, w), mode='bilinear', align_corners=False)

        # class prob
        diff_prob = neighbour_diff(resize(pred_mask_probs), 'l1')  # [n, 8, h, w]

        # high-level boundary
        boundary_proj = F.interpolate(boundary_proj[None], (h, w), mode='bilinear', align_corners=False)[0]
        boundary_proj = torch.sigmoid(boundary_proj)  
        neighbor_boundary_proj = _unfold_wo_center(boundary_proj.unsqueeze(1), 3, 1).squeeze()  
        diff_bond_high_level = torch.maximum(neighbor_boundary_proj, boundary_proj.unsqueeze(dim=1))

        # embed feature
        memory_proj = resize(memory_proj)
        diff_feat = neighbour_diff(memory_proj, 'dot')
        diff_feat = diff_feat.clip(min=0)

        # low-level boundary
        if self._low_level_edge:
            images = self.tmp_state['img']
            img_edge = image_to_edgebox(images, self.edge_model_path)
            img_edge = F.interpolate(img_edge[None], (h, w), mode='bilinear', align_corners=False)[0]
            diff_bond_low_level = neighbour_diff(img_edge[:, None], 'max')
            diff_all = diff_prob * 1 + (diff_bond_high_level + diff_bond_low_level) * self.lambda_diff_bond + diff_feat * self.lambda_diff_feat

        else:
            diff_all = diff_prob * 1 + diff_bond_high_level * self.lambda_diff_bond + diff_feat * self.lambda_diff_feat

        # [n, h, w, 8]
        diff_np = diff_all.permute(0, 2, 3, 1).data.cpu().numpy()
        coords_raw = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W)), -1).to(pred_mask_probs)  # [H, W, 2]

        out_bboxes_list, out_masks_list = [], []

        for i, gt_masks in enumerate(gt_masks_list):
            # query point coordinates
            pnt_coords = torch.nonzero(gt_masks)
            if len(pnt_coords) == 0:
                pseudo_gt = torch.zeros_like(gt_masks)
            else:
                pnt_coords_raw = pnt_coords
                pnt_coords = (pnt_coords * dfactor.view(1, 3)).long()
                pnt_coords[:, 1].clip_(min=0, max=h - 1)
                pnt_coords[:, 2].clip_(min=0, max=w - 1)
                pnt_coords_np = pnt_coords.data.cpu().numpy()
                pnt_indices = [pnt_coords_np[:, 0] == ii for ii in range(len(gt_masks))]

                # min distance
                raw_mindist = dijkstra_image(diff_np[i], pnt_coords_np[:, 1:])
                raw_mindist_max = 10
                raw_mindist_list = []
                for ii in range(len(gt_masks)):
                    raw_mindist_i = raw_mindist[pnt_indices[ii]]
                    if len(raw_mindist_i) > 0:
                        raw_mindist_list.append(raw_mindist_i.min(0))
                    else:
                        raw_mindist_list.append(np.full((h, w), raw_mindist_max, np.float32))
                raw_mindist = np.array(raw_mindist_list)
                raw_mindist = torch.as_tensor(raw_mindist, dtype=torch.float32, device=pnt_coords.device)
                raw_mindist = F.interpolate(raw_mindist[None], (H, W), mode='bilinear', align_corners=False)[0]
                raw_mindist_1 = raw_mindist.clone()

                # not use OT at the beginning
                if train_iter<= self.ot_warmup_iters:
                    k_num_matrix = raw_mindist.min(dim=0)[1]
                    k_num_list = []
                    total_pixel = H * W

                    for j in range(len(raw_mindist)):
                        k_j = (k_num_matrix == j).sum()
                        k_num_list.append(k_j)

                    k_num_list[-1] = total_pixel - sum(k_num_list[:-1])

                    mu = torch.IntTensor(k_num_list).to(raw_mindist.device)

                    m = H * W
                    n = len(raw_mindist)
                    cost = raw_mindist.view(n, -1)

                    nu = raw_mindist.new_ones(m)
                    _, pi = self.sinkhorn(mu, nu, cost)
                    pi = pi.view(n, H, W)

                raw_mindist /= raw_mindist.max() + 1e-5
                dist_likeli = 1 - raw_mindist  # [n, H, W]

                # class comp
                gt_labels = F.one_hot(gt_labels_list[i], 256)[..., :self.num_things_classes].float()  # [n, c]
                gt_labels = torch.cat(
                    [gt_labels, cls_labels[i:i + 1, self.num_things_classes:].expand(len(gt_labels), -1).float()],
                    -1)  # [n, C]
                clas_likeli = torch.einsum('nc,chw->nhw', gt_labels, pred_mask_probs[i])

                # instance masks
                likeli = dist_likeli * clas_likeli

                pseudo_gt = likeli.argmax(0)
                pseudo_gt[pred_mask_probs[i].argmax(0) >= self.num_things_classes] = len(likeli)
                pseudo_gt = F.one_hot(pseudo_gt, len(likeli) + 1)[..., :len(likeli)].permute(2, 0, 1).contiguous()  # [n, h, w]

                # use OT
                if train_iter > self.ot_warmup_iters:
                    # Centroid seeking
                    center_ids = np.zeros((len(pseudo_gt), 2))
                    for kk in np.arange(len(pseudo_gt)):
                        center_h, center_w = ndimage.measurements.center_of_mass(pseudo_gt[kk].cpu().numpy())
                        try:
                            if center_h >= 0 and center_w >= 0:
                                center_h = center_h.astype(int)
                                center_w = center_w.astype(int)
                                raw_h = (pnt_coords_np[:, 1:][pnt_indices[kk]][0][0] / downrate).astype(int)
                                raw_w = (pnt_coords_np[:, 1:][pnt_indices[kk]][0][1] / downrate).astype(int)
                                if pseudo_gt[kk, center_h, center_w] > 0 and pseudo_gt[kk, raw_h, raw_w] > 0:
                                    center_ids[kk, 0] = (center_h * downrate).astype(int)
                                    center_ids[kk, 1] = (center_w * downrate).astype(int)
                                else:
                                    center_ids[kk] = pnt_coords_np[:, 1:][pnt_indices[kk]][0]
                            else:
                                center_ids[kk] = pnt_coords_np[:, 1:][pnt_indices[kk]][0]
                        except:
                            center_ids[kk][0] = 0
                            center_ids[kk][1] = 0
                            error_ids.append(kk)                    

                    # min distance
                    mindist = dijkstra_image(diff_np[i], center_ids)

                    mindist = torch.as_tensor(mindist, dtype=torch.float32, device=pnt_coords.device)
                    mindist /= mindist.max() + 1e-5
                    mindist = F.interpolate(mindist[None], (H, W), mode='bilinear', align_corners=False)[0]
                    dist_likeli = 1 - mindist  # [n, H, W]

                    # instance masks
                    likeli = dist_likeli * clas_likeli

                    pseudo_gt = likeli.argmax(0)
                    pseudo_gt[pred_mask_probs[i].argmax(0) >= self.num_things_classes] = len(gt_masks)
                    pseudo_gt_oh = F.one_hot(pseudo_gt, len(likeli) + 1)[..., :len(likeli)].permute(2, 0, 1).contiguous()  # [n, h, w]

                    # *********OT solver***********
                    mu = pseudo_gt_oh.sum(dim=(1,2))
                    n = len(raw_mindist)
                    cost = raw_mindist_1.view(n, -1)

                    nu = mindist.new_ones((H,W))
                    nu = torch.where(pseudo_gt==len(gt_masks),torch.tensor(0,dtype=nu.dtype,device=nu.device),nu).reshape(-1)
                    _, pi = self.sinkhorn(mu, nu, cost)

                    pi = pi.view(n, H, W)

                likeli = pi
                # likeli = pi * clas_likeli
                pseudo_gt = likeli.argmax(0)
                pseudo_gt[pred_mask_probs[i].argmax(0) >= self.num_things_classes] = len(gt_masks)
                pseudo_gt = F.one_hot(pseudo_gt, len(likeli) + 1)[..., :len(likeli)].permute(2, 0, 1).contiguous()  # [n, h, w]

            # instance bboxes
            pseudo_bboxes = []
            img_h, img_w = img_metas[i]['img_shape'][:2]
            mask_sizes = []
            for pgt in pseudo_gt:
                mask_coords = coords_raw[pgt > 0]
                mask_sizes.append(len(mask_coords))
                if len(mask_coords) == 0:
                    bboxes = [img_w // 4, img_h // 4, img_w * 3 // 4, img_h * 3 // 4]
                else:
                    y0, x0 = mask_coords.min(0)[0]
                    y1, x1 = mask_coords.max(0)[0]
                    bboxes = [x0, y0, x1 + 1, y1 + 1]
                pseudo_bboxes.append(bboxes)
            pseudo_bboxes = coords_raw.new_tensor(pseudo_bboxes)
            pseudo_bboxes[:, 0::2].clip_(min=0, max=img_w)
            pseudo_bboxes[:, 1::2].clip_(min=0, max=img_h)

            out_bboxes_list.append(pseudo_bboxes)
            out_masks_list.append(pseudo_gt)

        pl_labels_list = gt_labels_list
        pl_bboxes_list = out_bboxes_list
        pl_masks_list = out_masks_list
        return pl_labels_list, pl_bboxes_list, pl_masks_list, pl_semantics_list


    def simplified_filter_and_loss(self,
                                   cls_scores,
                                   bbox_preds,
                                   gt_bboxes_list,
                                   gt_labels_list,
                                   gt_masks_list,
                                   img_metas,
                                   gt_bboxes_ignore_list):
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list,
                                           img_metas, gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg, pos_inds_list, gt_inds_list) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
                         num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(cls_scores,
                                 labels,
                                 label_weights,
                                 avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(img_metas, bbox_preds):
            img_h, img_w, _ = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)

        bbox_preds = bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(bboxes,
                                 bboxes_gt,
                                 bbox_weights,
                                 avg_factor=num_total_pos)

        # regression L1 loss
        loss_bbox = self.loss_bbox(bbox_preds,
                                   bbox_targets,
                                   bbox_weights,
                                   avg_factor=num_total_pos)

        assert len(gt_inds_list) == len(gt_masks_list) == num_imgs, (len(gt_inds_list), len(gt_masks_list), num_imgs)
        mask_targets = [mask[inds] for mask, inds in zip(gt_masks_list, gt_inds_list)]
        mask_weights = [mask.new_ones(mask.shape[0]) for mask in mask_targets]

        return loss_cls, loss_bbox, loss_iou, pos_inds_list, num_total_pos, mask_targets, mask_weights


    def loss_single_panoptic_simplified(self,
                                        cls_scores,
                                        bbox_preds,
                                        args_tuple,
                                        gt_bboxes_list,
                                        gt_labels_list,
                                        gt_masks_list,
                                        gt_semantics_list,
                                        img_metas,
                                        gt_bboxes_ignore_list=None):
        loss_dict = {}

        (loss_cls, loss_iou, loss_bbox,
         pos_inds_mask_list, num_total_pos_thing,
         mask_targets_list, mask_weights_list) = self.simplified_filter_and_loss(
            cls_scores, bbox_preds,
            gt_bboxes_list, gt_labels_list, gt_masks_list,
            img_metas, gt_bboxes_ignore_list)
        loss_dict.update({
            'loss_cls': loss_cls,
            'loss_iou': loss_iou,
            'loss_bbox': loss_bbox})

        # batch first args
        memory, memory_mask, memory_pos, query, _, query_pos, hw_lvl = args_tuple
        bsz, _, embed_dims = query.shape

        # thing masks & loss
        max_query_num = max(len(pos_inds) for pos_inds in pos_inds_mask_list)
        thing_query = query.new_zeros([bsz, max_query_num, embed_dims])
        for i, pos_inds in enumerate(pos_inds_mask_list):
            thing_query[i, :len(pos_inds)] = query[i, pos_inds]

        mask_things, mask_inter_things, query_inter_things = self.things_mask_head(
            memory, memory_mask, None, thing_query, None, None, hw_lvl=hw_lvl)
        # dummy loss
        loss_dict['loss_cls'] = loss_dict['loss_cls'] + sum(x.sum() * 0 for x in query_inter_things)

        all_th_masks = []
        for th_masks in mask_inter_things + [mask_things]:
            th_masks = th_masks.squeeze(-1)
            th_masks = [mask[:len(pos_inds)].view(-1, *hw_lvl[0]) for mask, pos_inds in \
                        zip(th_masks, pos_inds_mask_list)]
            all_th_masks.append(th_masks)

        self.tmp_state.update({
            'th_masks': all_th_masks[-1],
            'th_pos_inds_list': pos_inds_mask_list, })

        th_mask_targets = torch.cat(mask_targets_list).float()
        th_mask_weights = torch.cat(mask_weights_list)

        all_th_masks = [torch.cat(th_masks) for th_masks in all_th_masks]
        all_th_masks = F.interpolate(torch.stack(all_th_masks), th_mask_targets.shape[-2:],
                                     mode='bilinear', align_corners=False)
        for i, th_masks in enumerate(all_th_masks):
            loss_mask = self.loss_mask(th_masks,
                                       th_mask_targets,
                                       th_mask_weights,
                                       avg_factor=num_total_pos_thing)
            loss_dict.update({f'd{i}.loss_mask': loss_mask})

        # stuff masks & loss
        stuff_query, stuff_query_pos = torch.split(
            self.stuff_query.weight[None].expand(bsz, -1, -1),
            embed_dims, -1)

        mask_stuff, mask_inter_stuff, query_inter_stuff = self.stuff_mask_head(
            memory, memory_mask, None, stuff_query, None, stuff_query_pos, hw_lvl=hw_lvl)

        all_st_masks = []
        for st_masks in mask_inter_stuff + [mask_stuff]:
            st_masks = st_masks.squeeze(-1)
            st_masks = st_masks.reshape(-1, *hw_lvl[0])
            all_st_masks.append(st_masks)

        all_st_cls = []
        for i, query_st in enumerate(query_inter_stuff):
            st_cls = self.cls_stuff_branches[i](query_st).view(-1, 1)
            all_st_cls.append(st_cls)

        self.tmp_state.update({
            'st_masks': all_st_masks[-1].view(bsz, -1, *hw_lvl[0]),
            'st_cls': all_st_cls[-1].view(bsz, -1), })

        if isinstance(gt_semantics_list, list):
            gt_semantics_list = torch.stack(gt_semantics_list)
        target_st = F.one_hot(gt_semantics_list.long(), max(256, gt_semantics_list.max() + 1))
        target_st = target_st[..., self.num_things_classes:self.num_things_classes + self.num_stuff_classes]
        target_st = target_st.permute(0, 3, 1, 2).float().flatten(0, 1)

        all_st_masks = F.interpolate(torch.stack(all_st_masks), target_st.shape[-2:],
                                     mode='bilinear', align_corners=False)

        target_st_weight = target_st.max(2)[0].max(1)[0]
        num_total_pos_stuff = target_st_weight.sum()
        if num_total_pos_stuff > 0:
            for i, st_masks in enumerate(all_st_masks):
                loss_mask = self.loss_mask(st_masks,
                                           target_st,
                                           target_st_weight,
                                           avg_factor=num_total_pos_stuff)
                loss_dict.update({f'd{i}.loss_st_mask': loss_mask})

            target_st_label = (1 - target_st_weight).long()
            for i, st_cls in enumerate(all_st_cls):
                loss_cls = self.loss_cls(
                    st_cls,
                    target_st_label,
                    avg_factor=num_total_pos_stuff) * 2
                loss_dict.update({f'd{i}.loss_st_cls': loss_cls})
        else:
            loss_dict.update({f'd{i}.loss_st_mask': (st_masks * 0).sum() for i, st_masks in enumerate(all_st_masks)})
            loss_dict.update({f'd{i}.loss_st_cls': (st_cls * 0).sum() for i, st_cls in enumerate(all_st_cls)})

        thing_ratio = num_total_pos_thing / (num_total_pos_thing + num_total_pos_stuff)
        stuff_ratio = 1 - thing_ratio
        return loss_dict, thing_ratio, stuff_ratio, all_th_masks, all_st_masks


    @force_fp32(apply_to=('x',))
    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_masks=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      gt_semantic_seg=None,
                      **kwargs):

        assert proposal_cfg is None, '"proposal_cfg" must be None'
        outs = self(x, img_metas)

        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            if gt_masks is None:
                loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
            else:
                loss_inputs = outs + (gt_bboxes, gt_labels, gt_masks, gt_semantic_seg,
                                      img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses


    def _get_bboxes_single(self,
                           cls_score,
                           bbox_pred,
                           img_shape,
                           scale_factor,
                           rescale=False):

        assert len(cls_score) == len(bbox_pred)
        max_per_img = self.test_cfg.get('max_per_img', self.num_query)

        # exclude background
        if self.loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()
            scores, indexes = cls_score.view(-1).topk(max_per_img)
            det_labels = indexes % self.num_things_classes
            bbox_index = indexes // self.num_things_classes
            bbox_pred = bbox_pred[bbox_index]
        else:
            scores, det_labels = F.softmax(cls_score, dim=-1)[..., :-1].max(-1)
            scores, bbox_index = scores.topk(max_per_img)
            bbox_pred = bbox_pred[bbox_index]
            det_labels = det_labels[bbox_index]

        det_bboxes = bbox_cxcywh_to_xyxy(bbox_pred)
        det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
        det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
        det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
        if rescale:
            det_bboxes /= det_bboxes.new_tensor(scale_factor)
        det_bboxes = torch.cat((det_bboxes, scores.unsqueeze(1)), -1)

        return bbox_index, det_bboxes, det_labels

    @force_fp32(apply_to=('all_cls_scores_list', 'all_bbox_preds_list',
                          'args_tuple'))
    @torch.no_grad()
    def get_bboxes(
            self,
            all_cls_scores,
            all_bbox_preds,
            enc_cls_scores,
            enc_bbox_preds,
            args_tuple,
            reference,
            img_metas,
            rescale=False,
    ):
        """
        """
        cls_scores = all_cls_scores[-1]
        bbox_preds = all_bbox_preds[-1]
        memory, memory_mask, memory_pos, query, _, query_pos, hw_lvl = args_tuple

        det_results = []
        pan_results = []
        for img_id in range(len(img_metas)):
            cls_score = cls_scores[img_id][..., :self.num_things_classes]
            bbox_pred = bbox_preds[img_id]
            img_shape = img_metas[img_id]['img_shape'][:2]
            ori_shape = img_metas[img_id]['ori_shape'][:2]
            bch_shape = img_metas[img_id]['batch_input_shape']
            scale_factor = img_metas[img_id]['scale_factor']

            proposals = self._get_bboxes_single(
                cls_score, bbox_pred, img_shape, scale_factor, rescale)
            det_results.append(proposals)

            bbox_index, det_bboxes, det_labels = proposals

            thing_query = query[img_id:img_id + 1, bbox_index]
            mask_things, mask_inter_things, query_inter_things = self.things_mask_head(
                memory[img_id:img_id + 1],
                memory_mask[img_id:img_id + 1],
                None,
                thing_query,
                None,
                None,
                hw_lvl=hw_lvl)

            stuff_query, stuff_query_pos = self.stuff_query.weight[None].split(self.embed_dims, -1)
            mask_stuff, mask_inter_stuff, query_inter_stuff = self.stuff_mask_head(
                memory[img_id:img_id + 1],
                memory_mask[img_id:img_id + 1],
                None,
                stuff_query,
                None,
                stuff_query_pos,
                hw_lvl=hw_lvl)

            mask_pred = torch.cat([mask_things, mask_stuff], 1).view(-1, *hw_lvl[0])
            mask_pred = F.interpolate(mask_pred[None], bch_shape,
                                      mode='bilinear', align_corners=False)[0]
            mask_pred = mask_pred[..., :img_shape[0], :img_shape[1]]
            mask_pred = F.interpolate(mask_pred[None], ori_shape,
                                      mode='bilinear', align_corners=False)[0]

            stuff_query = query_inter_stuff[-1]
            scores_stuff = self.cls_stuff_branches[-1](stuff_query).sigmoid().view(-1)

            binary_masks = mask_pred > 0.5
            mask_sizes = binary_masks.sum((1, 2)).float()

            scores_cls = torch.cat([det_bboxes[..., -1], scores_stuff])
            scores_msk = (mask_pred * binary_masks).sum((1, 2)) / (mask_sizes + 1)
            scores_all = scores_cls * (scores_msk ** 2)
            labels_all = torch.cat([det_labels,
                                    torch.arange(self.num_stuff_classes).to(det_labels) + self.num_things_classes])

            scores_all_, index = torch.sort(scores_all, descending=True)
            filled = binary_masks.new_zeros(mask_pred.shape[-2:]).bool()
            pan_result = torch.full(mask_pred.shape[-2:],
                                    self.num_things_classes + self.num_stuff_classes, device=mask_pred.device).long()
            pan_id = 1

            for i, score in zip(index, scores_all_):
                L = labels_all[i]
                isthing = L < self.num_things_classes

                score_threshold = self.quality_threshold_things \
                    if isthing else self.quality_threshold_stuff
                if score < score_threshold:
                    continue

                area = mask_sizes[i]
                if area == 0:
                    continue

                intersect_area = (binary_masks[i] & filled).sum()
                inter_threshold = self.overlap_threshold_things \
                    if isthing else self.overlap_threshold_stuff
                if (intersect_area / area) > inter_threshold:
                    continue

                mask = binary_masks[i] & (~filled)
                filled[mask] = True
                pan_result[mask] = pan_id * INSTANCE_OFFSET + L
                pan_id += 1

            # semantic prediction
            sem_query, sem_query_pos = self.semantic_query.weight[None].split(self.embed_dims, -1)
            mask_sem, mask_inter_sem, query_inter_sem = self.semantic_mask_head(
                memory[img_id:img_id + 1],
                memory_mask[img_id:img_id + 1],
                None,
                sem_query,
                None,
                sem_query_pos,
                hw_lvl=hw_lvl)
            mask_sem = mask_sem.reshape(1, -1, *hw_lvl[0])
            mask_sem = self.semantic_proj(mask_sem).squeeze(0)
            sem_cls = self.cls_semantic_branches[-1](query_inter_sem[-1]).view(-1, 1, 1)
            sem_mask_pred = mask_sem * sem_cls.sigmoid()
            sem_mask_pred = F.interpolate(sem_mask_pred[None], bch_shape,
                                          mode='bilinear', align_corners=False)[0]
            sem_mask_pred = sem_mask_pred[..., :img_shape[0], :img_shape[1]]
            sem_mask_pred = F.interpolate(sem_mask_pred[None], ori_shape,
                                          mode='bilinear', align_corners=False)[0]
            sem_result = sem_mask_pred.argmax(0)

            pan_results.append(dict(
                pan_results=pan_result.data.cpu().numpy(),
                semantic=sem_result.data.cpu().numpy()))

        return pan_results
