import torch
from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.detr import DETR
from mmdet.models.detectors.single_stage import SingleStageDetector
from mmdet.core import bbox2result

@DETECTORS.register_module()
class Panformer(SingleStageDetector):
    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super().__init__(backbone, neck, bbox_head, train_cfg, test_cfg, pretrained, init_cfg)
        self.first_forward = True

    def forward_train(self,
            img,
            img_metas,
            gt_bboxes,
            gt_labels,
            gt_masks=None,
            gt_bboxes_ignore=None,
            gt_semantic_seg=None,
            ):
        """Overwrite SingleStageDetector.forward_train to support masks.
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        if self.first_forward:
            pan_results = self.simple_test(img, img_metas)
            print([x['pan_results'].shape for x in pan_results])
            self.first_forward = False

        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        batch_input_shape = img.shape[-2:]
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape

        # pad masks
        if gt_masks is not None:
            gt_masks_ = []
            for mask in gt_masks:
                mask = mask.to_tensor(dtype=img.dtype, device=img.device)
                assert mask.ndim == 3, mask.shape
                mask_ = mask.new_zeros((mask.shape[0], *batch_input_shape))
                mask_[:, :mask.shape[1], :mask.shape[2]] = mask
                gt_masks_.append(mask_)
            gt_masks = gt_masks_

        if gt_semantic_seg is not None:
            gt_semantic_seg_ = []
            for mask in gt_semantic_seg:
                assert mask.ndim == 3 and mask.shape[0] == 1, mask.shape
                mask_ = mask.new_full(batch_input_shape, 255)
                mask_[:mask.shape[1], :mask.shape[2]] = mask[0]
                gt_semantic_seg_.append(mask_)
            gt_semantic_seg = torch.stack(gt_semantic_seg_)

        # forward
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels,
                                              gt_masks=gt_masks,
                                              gt_semantic_seg=gt_semantic_seg,
                                              gt_bboxes_ignore=gt_bboxes_ignore)

        # hack for visualization
        self.bbox_head._visualization_stats.update({
                'img': img[0],
                'img_metas': img_metas[0], })
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        batch_input_shape = img.shape[-2:]
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape
        feat = self.extract_feat(img)
        results_list = self.bbox_head.simple_test(
            feat, img_metas, rescale=rescale)
        return results_list # pan_results
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    @torch.no_grad()
    def get_visualization(self):
        return self.bbox_head.get_visualization()
