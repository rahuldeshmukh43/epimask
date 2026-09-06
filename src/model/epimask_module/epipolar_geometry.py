import torch
import torch.nn as nn

class Calculate_Epipolar_Distance(nn.Module):
    def __init__(self):
        super(Calculate_Epipolar_Distance, self).__init__()
        return

    def forward(self, F_mat, x0, y0, x1, y1):
        """
        Args:
            F_mat (torch.Tensor): [N, 3, 3]
            x0 (torch.Tensor): [h0, w0] coordinate (requires_grad=False)
            y0 (torch.Tensor): [h0, w0] coordinate (requires_grad=False)
            x1 (torch.Tensor): [h1, w1] coordinate (requires_grad=False)
            y1 (torch.Tensor): [h1, w1] coordinate (requires_grad=False)
        Returns:
            epi_dist (torch.Tensor): [N, L, S], L = hw0, S = hw1
        """
        eps = 0.0
        N, _, _ = F_mat.shape

        # reshape x0, y0, x1, y1 to [hw0], [hw1]
        x0 = x0.reshape(-1) # [hw0]
        y0 = y0.reshape(-1)
        x1 = x1.reshape(-1) # [hw1]
        y1 = y1.reshape(-1)

        # create homogeneous coordinates
        pt0 = torch.stack([x0, y0, torch.ones_like(x0)], dim=0).unsqueeze(0).expand(N, -1, -1) # [B, 3, hw0]
        pt1 = torch.stack([x1, y1, torch.ones_like(x1)], dim=0).unsqueeze(0).expand(N, -1, -1) # [B, 3, hw1]

        # calculate the symmetric epipolar distance -- 
        # Compute l1 = F_mat @ pt0 and l2 = F_mat^T @ pt1
        l0 = torch.einsum('bij,bjk->bik', F_mat, pt0)  # [N, 3, hw0]
        l1 = torch.einsum('bij,bjk->bik', F_mat.transpose(1, 2), pt1)  # [N, 3, hw1]

        # Compute the squared direction of lines
        l0_dir = l0[:, 0, :]**2 + l0[:, 1, :]**2 + eps # [N, hw0]
        l1_dir = l1[:, 0, :]**2 + l1[:, 1, :]**2 + eps # [N, hw1]
        
        n0 = torch.einsum('bds,bdl->bls', pt1, l0) # [N, hw0, hw1]
        n1 = torch.einsum('bdl,bds->bls', pt0, l1) # [N, hw0, hw1]

        d0 = (n0 ** 2) / l0_dir.unsqueeze(-1)  # [N, hw0, hw1]
        d1 = (n1 ** 2) / l1_dir.unsqueeze(1) # [N, hw0, hw1]

        epi_dist = 0.5 * (torch.sqrt(d0) + torch.sqrt(d1))
        # epi_dist = torch.sqrt(d0)

        if torch.isnan(epi_dist).any():
            raise ValueError("epi_dist has NAN values")
        return epi_dist# [hw0, hw1]

def make_2d_coordinates(h: int, 
                        w: int,                           
                        device: str = 'cpu'):
    """
    Args:
        h (int): height
        w (int): width
        device (str): 'cpu' or 'cuda'
    Returns:
        x (torch.Tensor): [h, w], requires_grad=False
        y (torch.Tensor): [h, w], requires_grad=False
    """
    x = torch.arange(w, dtype=torch.float32, device=device, requires_grad=False)
    y = torch.arange(h, dtype=torch.float32, device=device, requires_grad=False)
    x, y = torch.meshgrid(x, y, indexing='xy')
    return x, y
