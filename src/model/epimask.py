import torch
import torch.nn as nn
from einops.einops import rearrange

from epimask.external.LoFTR.src.loftr.backbone import build_backbone
from epimask.external.LoFTR.src.loftr.loftr_module import LocalFeatureTransformer, FinePreprocess
from epimask.external.LoFTR.src.loftr.utils.coarse_matching import CoarseMatching
from epimask.external.LoFTR.src.loftr.utils.fine_matching import FineMatching
from epimask.external.LoFTR.src.loftr.utils.position_encoding import PositionEncodingSine

from .backbone import build_satlasfpn
from .epimask_module.transformer import MaskedXATransformer
from .epimask_module.epipolar_geometry import make_2d_coordinates
from .epimask_module.epipolar_geometry import Calculate_Epipolar_Distance
from .utils.masked_coarse_matching import MaskedCoarseMatching

class EpiMask(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Misc
        self.config = config

        # Modules
        if config["backbone_type"] == "Satlas":
            self.backbone = build_satlasfpn(config)
        elif config["backbone_type"] == "ResNetFPN":
            self.backbone = build_backbone(config)
        else:
            raise ValueError(f'Backbone type {config["backbone_type"]} not supported.')

        if 'use_pos_encoding' in config and config['use_pos_encoding']:
            self.pos_encoding = PositionEncodingSine(
                config['coarse']['d_model'],
                temp_bug_fix=True)
        else:
            self.pos_encoding = None
        
        self.coarse_transformer = MaskedXATransformer(config['coarse'])

        if 'coarse_matching_type' not in config or config['coarse_matching_type'] == 'unmasked':
            self.coarse_matching = CoarseMatching(config['match_coarse'])
        elif config['coarse_matching_type'] == 'masked':
            self.coarse_matching = MaskedCoarseMatching(config['match_coarse'])
        else:
            raise ValueError(f'Coarse Matching type {config["coarse_matching_type"]} not supported.')

        self.fine_preprocess = FinePreprocess(config)
        self.fine_transformer = LocalFeatureTransformer(config["fine"])
        self.fine_matching = FineMatching()

    def apply_epipolar_dist_thr_decay(self):
        """
        Apply decay to the epipolar distance threshold in the coarse transformer.
        This is useful for training the model with a focus band that reduces over time.
        """
        self.coarse_transformer.apply_decay()
        return

    def forward(self, data):
        """ 
        Update:
            data (dict): {
                'image0': (torch.Tensor): (N, 1, H, W)
                'image1': (torch.Tensor): (N, 1, H, W)
                'mask0'(optional) : (torch.Tensor): (N, H, W) '0' indicates a padded position
                'mask1'(optional) : (torch.Tensor): (N, H, W)
            }
        """
        # 1. Local Feature CNN
        data.update({
            'bs': data['image0'].size(0),
            'hw0_i': data['image0'].shape[2:], 'hw1_i': data['image1'].shape[2:]
        })

        if data['hw0_i'] == data['hw1_i']:  # faster & better BN convergence
            feats_c, feats_f = self.backbone(torch.cat([data['image0'], data['image1']], dim=0))
            (feat_c0, feat_c1), (feat_f0, feat_f1) = feats_c.split(data['bs']), feats_f.split(data['bs'])
        else:  # handle different input shapes
            (feat_c0, feat_f0), (feat_c1, feat_f1) = self.backbone(data['image0']), self.backbone(data['image1'])

        if torch.isnan(feat_c0).any():
            raise ValueError("feat_c0 has NAN values")
        if torch.isnan(feat_f0).any():
            raise ValueError("feat_f0 has NAN values")
        if torch.isnan(feat_c1).any():
            raise ValueError("feat_c1 has NAN values")
        if torch.isnan(feat_f1).any():
            raise ValueError("feat_f1 has NAN values")
        
        data.update({
            'hw0_c': feat_c0.shape[2:], 'hw1_c': feat_c1.shape[2:],
            'hw0_f': feat_f0.shape[2:], 'hw1_f': feat_f1.shape[2:]
        })

        # 2. coarse-level module
        # add featmap with positional encoding, then flatten it to sequence [N, HW, C]
        if self.pos_encoding is not None:
            feat_c0 = rearrange(self.pos_encoding(feat_c0), 'n c h w -> n (h w) c')
            feat_c1 = rearrange(self.pos_encoding(feat_c1), 'n c h w -> n (h w) c')
        else:
            feat_c0 = rearrange(feat_c0, 'n c h w -> n (h w) c')
            feat_c1 = rearrange(feat_c1, 'n c h w -> n (h w) c')

        device = feat_c0.device
        dtype  = feat_c0.dtype

        mask_c0 = mask_c1 = None  # mask is useful in training
        if 'mask0' in data:
            mask_c0, mask_c1 = data['mask0'].flatten(-2), data['mask1'].flatten(-2)
                    
        h0_i, w0_i = data['hw0_i']
        h0_c, w0_c = data['hw0_c']
        h1_c, w1_c = data['hw1_c']
        sx = torch.tensor([h0_i/h0_c], device = device, dtype=dtype)
        sy = torch.tensor([w0_i/w0_c], device = device, dtype=dtype)
        
        F_gt = data['F_gt'].to(device=device, dtype=dtype)
        assert F_gt.size(1) == 3 and F_gt.size(2) == 3

        # make 2d coordinates for the coarse level
        x0, y0 = make_2d_coordinates(h0_c, w0_c, device=feat_c0.device) # [h0, w0]
        x1, y1 = make_2d_coordinates(h1_c, w1_c, device=feat_c0.device) # [h1, w1]

        # scale the coords
        x0 = x0 * sx
        x1 = x1 * sx
        y0 = y0 * sy
        y1 = y1 * sy

        # add the coordinates to data 
        data.update({
            'x0': x0, 'y0': y0,
            'x1': x1, 'y1': y1
        })
        
        # coarse transformer
        feat_c0, feat_c1 = self.coarse_transformer(feat_c0, feat_c1,
                                                x0, y0, x1, y1,
                                                mask0 = mask_c0,
                                                mask1 = mask_c1,
                                                F_gt = F_gt
                                                )
        # calc epi mask
        epi_dist = Calculate_Epipolar_Distance()(
            F_gt,
            x0, y0, x1, y1)
        cur_epi_dist_thr_for_last_attn_layer = self.coarse_transformer.epi_dist_thr_schedule[-1]
        epi_mask = epi_dist > cur_epi_dist_thr_for_last_attn_layer # [N, hw0 , hw1]

        if torch.isnan(feat_c0).any():
            raise ValueError("feat_c0 after coarse transformer has NAN values")
        if torch.isnan(feat_c1).any():
            raise ValueError("feat_c1 after coarse transformer has NAN values")

        # 3. match coarse-level
        if 'coarse_matching_type' not in self.config or self.config['coarse_matching_type'] == 'unmasked':
            self.coarse_matching(feat_c0, feat_c1, data, 
                                mask_c0=mask_c0, mask_c1=mask_c1)
        else:
            self.coarse_matching(feat_c0, feat_c1, data, 
                                mask_c0=mask_c0, mask_c1=mask_c1, 
                                epi_mask=epi_mask)


        # 4. fine-level refinement
        feat_f0_unfold, feat_f1_unfold = self.fine_preprocess(feat_f0, feat_f1, feat_c0, feat_c1, data)
        if feat_f0_unfold.size(0) != 0:  # at least one coarse level predicted
            feat_f0_unfold, feat_f1_unfold = self.fine_transformer(feat_f0_unfold, feat_f1_unfold)

        # 5. match fine-level
        self.fine_matching(feat_f0_unfold, feat_f1_unfold, data)

    def load_state_dict(self, state_dict, *args, **kwargs):
        for k in list(state_dict.keys()):
            if k.startswith('matcher.'):
                state_dict[k.replace('matcher.', '', 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)
    


