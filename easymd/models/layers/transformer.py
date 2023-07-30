import copy
import torch
import torch.nn as nn

from mmdet.models.utils.transformer import DeformableDetrTransformer
from mmdet.models.utils.builder import TRANSFORMER
from mmcv.runner.base_module import BaseModule
from .attention import MultiheadAttentionWithMask


class _MLP(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, act_layer=nn.ReLU, dropout=0.):
        super().__init__()
        self.layers = nn.Sequential(
                nn.Linear(in_channels, mid_channels),
                act_layer(),
                nn.Dropout(dropout),
                nn.Linear(mid_channels, out_channels),
                nn.Dropout(dropout))

    def forward(self, x):
        return self.layers(x)


class _DropPath(nn.Module):
    def __init__(self, p=0.):
        super().__init__()
        assert p >= 0 and p <= 1, p
        self.p = p

    def forward(self, x):
        if self.p == 0 or not self.training:
            return x
        kp = 1 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = (kp + torch.rand(shape, dtype=x.dtype, device=x.device)).floor_()
        output = x.div(kp) * mask
        return output


class MaskTransormerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, num_levels=3,
            self_attn=False, drop_path=0., norm_layer=nn.LayerNorm, act_layer=nn.GELU,
            tail=False):
        super().__init__()
        self.mask_attn = MultiheadAttentionWithMask(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                bias=bias,
                num_levels=num_levels)
        self.norm1 = norm_layer(embed_dim)

        if self_attn:
            self.self_attn = nn.MultiheadAttention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    bias=bias)
            self.norm2 = norm_layer(embed_dim)
        else:
            self.self_attn = None

        self.tail = tail
        if not tail:
            self.mlp = _MLP(
                    in_channels=embed_dim,
                    mid_channels=embed_dim * 4,
                    out_channels=embed_dim,
                    act_layer=act_layer,
                    dropout=dropout)
            self.norm3 = norm_layer(embed_dim)

        self.drop_path = _DropPath(drop_path)


    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None, hw_lvl=None):
        if self.self_attn is not None:
            self_attn, _ = self.self_attn(query, query, query, need_weights=False)
            query = query + self.drop_path(self_attn)
            query = self.norm2(query)

        cross_attn, mask_output = self.mask_attn(
                query, key, value,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
                hw_lvl=hw_lvl)

        if self.tail:
            return None, mask_output

        query = query + self.drop_path(cross_attn)
        query = self.norm1(query)

        query = query + self.drop_path(self.mlp(query))
        query = self.norm3(query)
        return query, mask_output


@TRANSFORMER.register_module()
class PanformerMaskDecoder(BaseModule):
    def __init__(self,
            d_model,
            num_heads,
            num_layers,
            norm_layer=nn.LayerNorm,
            act_layer=nn.GELU,
            dropout=0,
            drop_path=0,
            self_attn=False,
            num_levels=3):
        super().__init__()

        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_levels = num_levels

        #layer_cfg = dict(
        layer = MaskTransormerLayer(
                embed_dim=d_model,
                num_heads=num_heads,
                dropout=dropout,
                bias=True,
                num_levels=num_levels,
                self_attn=self_attn,
                drop_path=drop_path,
                norm_layer=norm_layer,
                act_layer=act_layer)
        #layer_cfgs = [copy.deepcopy(layer_cfg) for _ in range(num_layers + 1)]
        #layer_cfgs[-1].update({'tail': True})
        #self.layers = nn.ModuleList([MaskTransormerLayer(**cfg) for cfg in layer_cfgs])

        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        #self.layers = nn.ModuleList([MaskTransormerLayer(**layer_cfg) for _ in range(num_layers-1)])
        #self.layer_tail = MaskTransormerLayer(**layer_cfg, tail=True)

    @staticmethod
    def with_pos(query, query_pos):
        return query if query_pos is None else query + query_pos

    #def forward(self, query, key, value, key_padding_mask=None, attn_mask=None, hw_lvl=None):
    def forward(self, memory, query, memory_pos=None, query_pos=None,
            query_mask=None, memory_mask=None, hw_lvl=None):
        """
        Args:
            query: Tensor of Shape [nQuery, bs, dim]
            query_pos: position embedding of 'query'
            query_mask: [Tensor | None], Tensor of shape [bs, nQuery], indicating which query should NOT be involved in the attention computation.
            memory: Tensor of shape [sum(H_i*W_i), bs, dim]
            memory_pos: position embedding of 'memory'
            memory_mask: [Tensor | None], Tensor of shape [bs, sum(H_i*W_i)], indicating which elements in the memory should NOT be involved in the attention computation.

        Returns:
            inter_query: List[Tensor] of 'num_layers' query outputs, each of shape [nQuery, bs, dim]
            inter_mask: List[Tensor] of 'num_layers' mask outputs, each of shape [bs, nQuery, h,  w]
        """
        sizes = [h * w for h, w in hw_lvl]
        assert memory.shape[0] == sum(sizes), (memory.shape, sum(sizes))

        if len(hw_lvl) > self.num_levels:
            sizes = sizes[:self.num_levels]
            hw_lvl = hw_lvl[:self.num_levels]
            nMem = sum(sizes)
            memory = memory[:nMem]
            memory_pos = memory_pos[:nMem] if memory_pos is not None else None
            memory_mask = memory_mask[:, :nMem] if memory_mask is not None else None

        if query_mask is not None:
            nQuery, bsz, dim = query.shape
            assert query_mask.dtype == torch.bool, query_mask.dtype
            assert query_mask.shape == (bsz, nQuery), (query_mask.shape, query.shape)
            nMem = memory.shape[0]
            attn_mask = query_mask[:, None, :, None].expand(-1, self.num_heads, -1, nMem)
            attn_mask = attn_mask.flatten(0, 1)
        else:
            attn_mask = None

        inter_query, inter_mask = [], []
        for layer in self.layers:
            query, mask = layer(
                    self.with_pos(query, query_pos),
                    self.with_pos(memory, memory_pos),
                    memory,
                    key_padding_mask=memory_mask,
                    attn_mask=attn_mask,
                    hw_lvl=hw_lvl)
            inter_query.append(query)
            inter_mask.append(mask)
        #assert inter_query[-1] is None, inter_query[-1].shape
        #inter_query.pop()
        return inter_query, inter_mask


@TRANSFORMER.register_module()
class DeformableDetrTransformer2(DeformableDetrTransformer):
    """Copy-past from DeformableDetrTransformer, except that provides additional 
    memory info for predicting masks by panformer.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self,
                mlvl_feats,
                mlvl_masks,
                query_embed,
                mlvl_pos_embeds,
                reg_branches=None,
                cls_branches=None,
                **kwargs):
        """Forward function for `Transformer`.

        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, embed_dims, h, w].
            mlvl_masks (list(Tensor)): The key_padding_mask from
                different level used for encoder and decoder,
                each element has shape  [bs, h, w].
            query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            mlvl_pos_embeds (list(Tensor)): The positional encoding
                of feats from different level, has the shape
                 [bs, embed_dims, h, w].
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when
                `with_box_refine` is True. Default to None.
            cls_branches (obj:`nn.ModuleList`): Classification heads
                for feature maps from each decoder layer. Only would
                 be passed when `as_two_stage`
                 is True. Default to None.


        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.

                - inter_states: Outputs from decoder. If
                    return_intermediate_dec is True output has shape \
                      (num_dec_layers, bs, num_query, embed_dims), else has \
                      shape (1, bs, num_query, embed_dims).
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 4).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs,num_query, embed_dims)
                - enc_outputs_class: The classification score of \
                    proposals generated from \
                    encoder's feature maps, has shape \
                    (batch, h*w, num_classes). \
                    Only would be returned when `as_two_stage` is True, \
                    otherwise None.
                - enc_outputs_coord_unact: The regression results \
                    generated from encoder's feature maps., has shape \
                    (batch, h*w, 4). Only would \
                    be returned when `as_two_stage` is True, \
                    otherwise None.
        """
        assert self.as_two_stage or query_embed is not None

        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(
                zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            feat = feat.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack(
            [self.get_valid_ratio(m) for m in mlvl_masks], 1)

        reference_points = \
            self.get_reference_points(spatial_shapes,
                                      valid_ratios,
                                      device=feat.device)

        feat_flatten = feat_flatten.permute(1, 0, 2)  # (H*W, bs, embed_dims)
        lvl_pos_embed_flatten = lvl_pos_embed_flatten.permute(
            1, 0, 2)  # (H*W, bs, embed_dims)
        memory = self.encoder(
            query=feat_flatten,
            key=None,
            value=None,
            query_pos=lvl_pos_embed_flatten,
            query_key_padding_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            reference_points=reference_points,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            **kwargs)

        memory = memory.permute(1, 0, 2)
        bs, _, c = memory.shape
        if self.as_two_stage:
            output_memory, output_proposals = \
                self.gen_encoder_output_proposals(
                    memory, mask_flatten, spatial_shapes)
            enc_outputs_class = cls_branches[self.decoder.num_layers](
                output_memory)
            enc_outputs_coord_unact = \
                reg_branches[
                    self.decoder.num_layers](output_memory) + output_proposals

            topk = self.two_stage_num_proposals
            topk_proposals = torch.topk(
                enc_outputs_class[..., 0], topk, dim=1)[1]
            topk_coords_unact = torch.gather(
                enc_outputs_coord_unact, 1,
                topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            init_reference_out = reference_points
            pos_trans_out = self.pos_trans_norm(
                self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
            query_pos, query = torch.split(pos_trans_out, c, dim=2)
        else:
            query_pos, query = torch.split(query_embed, c, dim=1)
            query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
            query = query.unsqueeze(0).expand(bs, -1, -1)
            reference_points = self.reference_points(query_pos).sigmoid()
            init_reference_out = reference_points

        # decoder
        query = query.permute(1, 0, 2)
        memory = memory.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=memory,
            query_pos=query_pos,
            key_padding_mask=mask_flatten,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=reg_branches,
            **kwargs)

        inter_references_out = inter_references

        # additional info for mask decoder
        args_tuple = (
                memory, # [sum(H_i*W_i), bs, embed_dims]
                lvl_pos_embed_flatten, # [sum(H_i*W_i), bs, embed_dims]
                mask_flatten, # [bs, sum(H_i*W_i)]
                inter_states[-1], # [num_query, bs, embed_dims]
                query_pos, # [num_query, bs, embed_dims]
                )

        if self.as_two_stage:
            return inter_states, init_reference_out,\
                inter_references_out, enc_outputs_class,\
                enc_outputs_coord_unact, args_tuple
        return inter_states, init_reference_out, \
            inter_references_out, None, None, args_tuple

