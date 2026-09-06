import torch
import torch.nn as nn

from epimask.external.LoFTR.src.loftr.loftr_module.transformer import LoFTREncoderLayer

from epimask.src.model.epimask_module.epipolar_geometry import Calculate_Epipolar_Distance

class MaskedXAttentionLayer(nn.Module):
    def __init__(self,
                 epi_dist_thr:float=1.0,
                 cam_type:str='affine'):
        super(MaskedXAttentionLayer, self).__init__()
        self.epi_dist_thr = epi_dist_thr
        self.cam_type = cam_type

        self.calculate_epi_distance = Calculate_Epipolar_Distance()
        # for hook
        self.softmax = nn.Softmax(dim=2)
        return

    def update_epi_dist_thr(self, epi_dist_thr:float):
        self.epi_dist_thr = epi_dist_thr
        return

    def forward(self, q, k, v,
                x0, y0, x1, y1,
                q_mask= None,
                kv_mask = None,
                F_gt=None):
        """
        Args:
            q (torch.Tensor): [N, L, H, D]
            k (torch.Tensor): [N, S, H, D]
            v (torch.Tensor): [N, S, H, D]
            q_mask (torch.Tensor): [N, L] -- binary mask on q to mask out padded tokens (for nodata, clouds etc)
            kv_mask (torch.Tensor): [N, S] -- binary mask on k and v to mask out padded tokens (for nodata, clouds etc)
            F_gt (torch.Tensor): [N, 3, 3] -- ground truth fundamental matrix, used to teacher-force the epipolar mask
            #NOTE: Make sure F_gt is scaled to the same resolution as the feature maps
        Returns:
            out (torch.Tensor): [N, L, H, D]
        """
        assert F_gt is not None, "F_gt is required to compute the epipolar mask"

        qk = torch.einsum("nlhd, nshd -> nlsh", q, k) # [N, L, S, H]

        # apply data mask (q_mask and kv_mask)
        if kv_mask is not None:
            qk.masked_fill_(~(q_mask[:, :, None, None] * kv_mask[:, None, :, None]), float('-inf'))

        # apply epi distance mask (teacher forcing)
        epi_dist = self.calculate_epi_distance(F_gt, x0, y0, x1, y1) # [N, L, S]

        # calc epi distance mask
        epi_mask = epi_dist > self.epi_dist_thr # [N, L, S]

        # Ensure at least one valid entry per row
        valid_mask = epi_mask.sum(dim=2, keepdim=True) < epi_mask.shape[2]
        epi_mask = epi_mask & valid_mask  # At least one valid element remains

        # apply epi distance mask
        qk.masked_fill_(epi_mask[:, :, :, None], float('-inf'))

        # calculate the softmax temperature
        temp = 1. / q.shape[-1] ** 0.5

        attn = self.softmax(qk * temp) # [N, L, S, H]
        out = torch.einsum("nlsh, nshd -> nlhd", attn, v) # [N, L, H, D]
        if torch.isnan(attn).any():
            raise ValueError("attn has NAN values")
        return out.contiguous()

class MaskedXATransformerLayer(nn.Module):
    def __init__(self,
                d_model,
                nhead,
                epi_dist_thr:float = 1.0,
                cam_type:str='affine'):
        super(MaskedXATransformerLayer, self).__init__()
        self.cam_type = cam_type
        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        self.attention = MaskedXAttentionLayer(
            epi_dist_thr= epi_dist_thr,
            cam_type=self.cam_type)
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def update_epi_dist_thr(self, epi_dist_thr:float):
        self.attention.update_epi_dist_thr(epi_dist_thr)
        return

    def forward(self, x, source,
                x0, y0, x1, y1,
                x_mask=None, source_mask=None, F_gt=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x0 (torch.Tensor): [h0, w0]
            y0 (torch.Tensor): [h0, w0]
            x1 (torch.Tensor): [h1, w1]
            y1 (torch.Tensor): [h1, w1]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
            F_gt (torch.Tensor): [N, 3, 3]
        """
        bs = x.size(0)
        query, key, value = x, source, source

        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention(query, key, value,
                                        x0, y0, x1, y1,
                                        q_mask=x_mask, kv_mask=source_mask, F_gt = F_gt)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead*self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return x + message
    
def decay(initial_value: float,
        reduction_factor: float,
        num_iters: int):
    """
    Linearly decays an initial value such that the final value is initial_value * reduction_factor.

    Args:
        initial_value (float): The starting value.
        reduction_factor (float): The factor by which the final value is determined.
        num_iters (int): Number of values to return (one per step); values[0] is
            initial_value and values[-1] is initial_value * reduction_factor.

    Returns:
        list: The decayed values, one per step (length num_iters).
    """
    final_value = initial_value * reduction_factor

    if num_iters == 1:
        return [final_value]

    denom = num_iters - 1
    step = (initial_value - final_value) / denom
    return [initial_value - i * step for i in range(num_iters)]

class MaskedXATransformer(nn.Module):
    def __init__(self, config):
        super(MaskedXATransformer, self).__init__()
        self.config = config
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.layer_names = config['layer_names'] #['self', 'maskedx']

        # setup schedule for epi_dist
        self.epi_dist_thr_start = config['epi_dist_thr_start']
        self.reduction_factor = config['epi_dist_thr_reduction_factor']

        # count number of 'maskedx' layers
        num_maskedx_layers = sum([1 for layer_name in self.layer_names if layer_name == 'maskedx'])
        if num_maskedx_layers > 0:
            self.epi_dist_thr_schedule = [1e4] * num_maskedx_layers
        else:
            # no 'maskedx' layers
            self.epi_dist_thr_schedule = None

        self.layers = nn.ModuleList()
        _count = 0
        for layer_name in self.layer_names:
            if layer_name == 'self':
                _layer = LoFTREncoderLayer(config['d_model'], config['nhead'], config['self_attention'])
            elif layer_name == 'maskedx':
                _layer = MaskedXATransformerLayer(config['d_model'],
                                                 config['nhead'],
                                                 epi_dist_thr= self.epi_dist_thr_schedule[_count],
                                                 cam_type= config['cam_type'])
                _count += 1
            else:
                raise ValueError(f"Invalid MaskedXATransformer layer name {layer_name}")
            self.layers.append(_layer)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def apply_decay(self):
        """
        Apply the decay to the epipolar distance threshold based on the current epoch.
        """
        num_maskedx_layers = len(self.epi_dist_thr_schedule)
        epi_dist_thr_schedule = decay(self.epi_dist_thr_start,
                                    self.reduction_factor,
                                    num_maskedx_layers)
        print(f"Applying decay to epipolar distance threshold: {epi_dist_thr_schedule}")
        self.epi_dist_thr_schedule = epi_dist_thr_schedule

        # Apply updated epi_dist_thr values to each layer
        count = 0
        for i_layer, layer in enumerate(self.layers):
            if isinstance(layer, MaskedXATransformerLayer):
                layer.update_epi_dist_thr(self.epi_dist_thr_schedule[count])
                count += 1
        return

    def forward(self, feat0, feat1,
                x0, y0, x1, y1,
                mask0=None, mask1=None, F_gt=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            x0 (torch.Tensor): [h0, w0]
            y0 (torch.Tensor): [h0, w0]
            x1 (torch.Tensor): [h1, w1]
            y1 (torch.Tensor): [h1, w1]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
            F_gt  (torch.Tensor): [N, 3,3]
        Returns:
            feat0 (torch.Tensor) [N, L, C] transformed features
            feat1 (torch.Tensor): [N, S, C] transformed features
        """

        assert self.d_model == feat0.size(2), "the feature number of src and transformer must be equal. Got d_model %d and feat dim %d"%(self.d_model, feat0.size(2))

        for i_layer, (layer, name) in enumerate(zip(self.layers, self.layer_names)):
            if name == 'self':
                feat0 = layer(feat0, feat0,
                            x_mask = mask0, source_mask = mask0)
                feat1 = layer(feat1, feat1,
                            x_mask = mask1, source_mask = mask1)
                if torch.isnan(feat0).any():
                    raise ValueError("feat0 has NAN values for self attention layer %d"%(i_layer))
                if torch.isnan(feat1).any():
                    raise ValueError("feat1 has NAN values for self attention layer %d"%(i_layer))
            elif name == 'maskedx':
                feat0 = layer(feat0, feat1,
                                    x0, y0, x1, y1,
                                    x_mask = mask0, source_mask = mask1,
                                    F_gt = F_gt)
                feat1 = layer(feat1, feat0,
                                    x1, y1, x0, y0,
                                    x_mask = mask1, source_mask = mask0,
                                    F_gt = F_gt.transpose(1,2) if F_gt is not None else None)
                if torch.isnan(feat0).any():
                    raise ValueError("feat0 has NAN values for maskedx attention layer %d"%(i_layer))
                if torch.isnan(feat1).any():
                    raise ValueError("feat1 has NAN values for maskedx attention layer %d"%(i_layer))
            else:
                raise KeyError

        return feat0, feat1
