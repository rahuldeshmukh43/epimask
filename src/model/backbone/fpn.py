import torch
import torch.nn as nn
import torch.nn.functional as F
from .satlas import SatlasBackbone

class SatlasFPN_16_4(nn.Module):
    """
    FPN Decoder that produces decoded feature maps at 1/16 and 1/4 scales.
    Expects a configuration dictionary with keys:
      - in_channels: dict with keys "1/32", "1/16", "1/8", and "1/4"
      - out_channels: int, the number of output channels.
    """
    def __init__(self, config):
        super(SatlasFPN_16_4, self).__init__()
        self.satlas_encoder = SatlasBackbone(config)
        # num_channels = config['num_channels']
        num_channels = {config['num_channels']['keys'][ik]: config['num_channels']['values'][ik] for ik in range(len(config['num_channels']['keys']))}
        out_channels = {config['out_channels']['keys'][ik]: config['out_channels']['values'][ik] for ik in range(len(config['out_channels']['keys']))}

        self.refine_16 = nn.Sequential(
            nn.Conv2d(num_channels['1/16'], out_channels['1/16'], kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels['1/16']),
            nn.LeakyReLU(inplace=True)
        )                                                                                               #512 -> 256
        # Transformation layers (1x1 convolution to adjust channels)
        self.conv_16 = nn.Conv2d(out_channels['1/16'], out_channels['1/8'], kernel_size=1, bias=False)  #256 -> 256
        
        self.refine_8 = nn.Sequential(
            nn.Conv2d(num_channels['1/8'], out_channels['1/4'], kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels['1/4']),
            nn.LeakyReLU(inplace=True)
        )                                                                                               #256 -> 128          
        
        self.conv_8  = nn.Conv2d(out_channels['1/4'],  out_channels['1/4'], kernel_size=1, bias=False)  #128 -> 128
        # Refinement layers (3x3 conv + BatchNorm + LeakyReLU)
        
        
        self.refine_4 = nn.Sequential(
            nn.Conv2d(out_channels['1/4'], out_channels['1/4'], kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels['1/4']),
            nn.LeakyReLU(inplace=True)
        )                                                                                              #128 -> 128   

    def forward(self, x):
        """
        Args:
            features (dict): Feature maps with keys "1/32", "1/16", "1/8", and "1/4".
        Returns:
            dict: Decoded feature maps at 1/16 and 1/4 scales.
        """

        features = self.satlas_encoder(x)

        f_16 = self.refine_16(features['1/16']) # 512 -> 256
        f_8  = self.refine_8(features['1/8']  + F.interpolate(self.conv_16(f_16), scale_factor=2, mode='bilinear', align_corners=True)) # 256 -> 128
        f_4  = self.refine_4(features['1/4']  + F.interpolate(self.conv_8(f_8),  scale_factor=2, mode='bilinear', align_corners=True)) # 128 -> 128
        return [f_16, f_4]


class SatlasFPN_8_2_v0(nn.Module):
    """
    FPN Decoder that produces decoded feature maps at 1/8 and 1/2 scales.
    Expects a configuration dictionary with keys:
      - in_channels: dict with keys "1/32", "1/16", "1/8", and "1/4"
      - out_channels: int, the number of output channels.
    """
    def __init__(self, config):
        super(SatlasFPN_8_2_v0, self).__init__()
        self.satlas_encoder = SatlasBackbone(config)
        # num_channels = config['num_channels']
        # num_channels = config['num_channels']
        num_channels = {config['num_channels']['keys'][ik]: config['num_channels']['values'][ik] for ik in range(len(config['num_channels']['keys']))}
        out_channels = {config['out_channels']['keys'][ik]: config['out_channels']['values'][ik] for ik in range(len(config['out_channels']['keys']))}

        self.refine_16 = nn.Sequential(
            nn.Conv2d(num_channels['1/16'], out_channels['1/16'], kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels['1/16']),
            nn.LeakyReLU(inplace=True)
        )                                                                                               #512 -> 256
        # Transformation layers (1x1 convolution to adjust channels)
        self.conv_16 = nn.Conv2d(out_channels['1/16'], out_channels['1/8'], kernel_size=1, bias=False)  #256 -> 256
        
        self.refine_8 = nn.Sequential(
            nn.Conv2d(num_channels['1/8'], out_channels['1/8'], kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels['1/8']),
            nn.LeakyReLU(inplace=True)
        )                                                                                               #256 -> 256          
        
        self.conv_8  = nn.Conv2d(out_channels['1/8'],  out_channels['1/4'], kernel_size=1, bias=False)  #256 -> 128
        # Refinement layers (3x3 conv + BatchNorm + LeakyReLU)
        
        
        self.refine_4 = nn.Sequential(
            nn.Conv2d(out_channels['1/4'], out_channels['1/4'], kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels['1/4']),
            nn.LeakyReLU(inplace=True)
        )                                                                                               #128 -> 128   

        self.refine_2 = nn.Sequential(
            nn.Conv2d(out_channels['1/4'], out_channels['1/2'], kernel_size=3, padding=1, bias=False),  # 1/2 -> 128 channels
            nn.GroupNorm(1, out_channels['1/2']),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        """
        Args:
            features (dict): Feature maps with keys "1/32", "1/16", "1/8", and "1/4".
        Returns:
            dict: Decoded feature maps at 1/16 and 1/4 scales.
        """

        features = self.satlas_encoder(x)

        
        f_16 = self.refine_16(features['1/16']) # 512 -> 256
        f_8  = self.refine_8(features['1/8']  + F.interpolate(self.conv_16(f_16), scale_factor=2, mode='bilinear', align_corners=True)) # 256 -> 256
        f_4  = self.refine_4(features['1/4']  + F.interpolate(self.conv_8(f_8),  scale_factor=2, mode='bilinear', align_corners=True)) # 256 -> 128
        f_2  = self.refine_2(F.interpolate(f_4, scale_factor=2, mode='bilinear', align_corners=True))
        return [f_8, f_2]

# v3 - fuse 1/8, 1/4 and return 8_2
class SatlasFPN_8_2(SatlasFPN_8_2_v0):
    """
    FPN Decoder v3: produces decoded feature maps at 1/8 and 1/2 scales.
    - Fuses 1/16 -> 1/8 using concatenation
    - Fuses 1/8 -> 1/4 using concatenation
    """

    def __init__(self, config):
        super().__init__(config)

        out_channels = {
            config['out_channels']['keys'][ik]: config['out_channels']['values'][ik] 
            for ik in range(len(config['out_channels']['keys']))
        }

        # fusion at 1/8: cat [f8 skip, upsampled conv16(f16)]
        self.fuse_8 = nn.Sequential(
            nn.Conv2d(
                2* out_channels['1/8'],   # skip + upsampled
                out_channels['1/8'], kernel_size=3, padding=1, bias=False
            ),
            nn.GroupNorm(1, out_channels['1/8']),
            nn.LeakyReLU(inplace=True)
        )

        # fusion at 1/4: cat [f4 skip, upsampled conv8(f8)]
        self.fuse_4 = nn.Sequential(
            nn.Conv2d(
                2* out_channels['1/4'],   # skip + upsampled
                out_channels['1/4'], kernel_size=3, padding=1, bias=False
            ),
            nn.GroupNorm(1, out_channels['1/4']),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        features = self.satlas_encoder(x)

        # refine 1/16
        f_16 = self.refine_16(features['1/16'])

        # concat fusion at 1/8
        f_8_input = torch.cat([
            self.refine_8(features['1/8']),
            F.interpolate(self.conv_16(f_16), scale_factor=2, mode='bilinear', align_corners=True)
        ], dim=1)
        f_8 = self.fuse_8(f_8_input)

        # concat fusion at 1/4
        f_4_input = torch.cat([
            self.refine_4(features['1/4']),
            F.interpolate(self.conv_8(f_8), scale_factor=2, mode='bilinear', align_corners=True)
        ], dim=1)
        f_4 = self.fuse_4(f_4_input)

        # final 1/2
        f_2 = self.refine_2(
            F.interpolate(f_4, scale_factor=2, mode='bilinear', align_corners=True)
        )

        return [f_8, f_2]
    
class SatlasFPN_4_2_v0(SatlasFPN_8_2_v0):
    """
    FPN Decoder that produces decoded feature maps at 1/4 and 1/2 scales.
    Expects a configuration dictionary with keys:
      - in_channels: dict with keys "1/32", "1/16", "1/8", and "1/4"
      - out_channels: int, the number of output channels.
    """
    def __init__(self, config):
        super().__init__(config)
    
    def forward(self, x):
        """
        Args:
            features (dict): Feature maps with keys "1/32", "1/16", "1/8", and "1/4".
        Returns:
            dict: Decoded feature maps at 1/16 and 1/4 scales.
        """

        features = self.satlas_encoder(x)

        f_16 = self.refine_16(features['1/16']) # 512 -> 256
        f_8  = self.refine_8(features['1/8']  + F.interpolate(self.conv_16(f_16), scale_factor=2, mode='bilinear', align_corners=True)) # 256 -> 256
        f_4  = self.refine_4(features['1/4']  + F.interpolate(self.conv_8(f_8),  scale_factor=2, mode='bilinear', align_corners=True)) # 256 -> 128
        f_2  = self.refine_2(F.interpolate(f_4, scale_factor=2, mode='bilinear', align_corners=True))
        return [f_4, f_2]
#v1
class SatlasFPN_4_2_v1(SatlasFPN_8_2_v0):
    """
    FPN Decoder that produces decoded feature maps at 1/4 and 1/2 scales.
    Expects a configuration dictionary with keys:
      - in_channels: dict with keys "1/32", "1/16", "1/8", and "1/4"
      - out_channels: int, the number of output channels.
    """
    def __init__(self, config):
        super().__init__(config)
    
    def forward(self, x):
        """
        Args:
            features (dict): Feature maps with keys "1/32", "1/16", "1/8", and "1/4".
        Returns:
            dict: Decoded feature maps at 1/16 and 1/4 scales.
        """

        features = self.satlas_encoder(x)

        # f_16 = self.refine_16(features['1/16']) # 512 -> 256
        # f_8  = self.refine_8(features['1/8']  + F.interpolate(self.conv_16(f_16), scale_factor=2, mode='bilinear', align_corners=True)) # 256 -> 256
        f_8 = features['1/8']
        f_4  = self.refine_4(features['1/4']  + F.interpolate(self.conv_8(f_8),  scale_factor=2, mode='bilinear', align_corners=True)) # 256 -> 128
        f_2  = self.refine_2(F.interpolate(f_4, scale_factor=2, mode='bilinear', align_corners=True))
        return [f_4, f_2]
    
# v2 - use cat + conv to combine skip instead of add 
class SatlasFPN_4_2_v2(SatlasFPN_8_2_v0):
    """
    FPN Decoder v2: produces decoded feature maps at 1/4 and 1/2 scales.
    Uses concatenation + conv for skip fusion (instead of addition).
    """

    def __init__(self, config):
        super().__init__(config)

        out_channels = {config['out_channels']['keys'][ik]: config['out_channels']['values'][ik] 
                        for ik in range(len(config['out_channels']['keys']))}

        # Fusion convs after concatenation
        self.fuse_4 = nn.Sequential(
            nn.Conv2d(2 * out_channels['1/4'],  # cat [skip, upsampled]
                      out_channels['1/4'], kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels['1/4']),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        features = self.satlas_encoder(x)

        # use raw 1/8 features (no fusion with 1/16 here, like v1)
        f_8 = features['1/8']  

        # instead of addition: concatenate skip + upsampled
        f_4_input = torch.cat([features['1/4'], 
                               F.interpolate(self.conv_8(f_8), scale_factor=2, 
                                             mode='bilinear', align_corners=True)], dim=1)
        f_4 = self.fuse_4(f_4_input)

        f_2  = self.refine_2(F.interpolate(f_4, scale_factor=2, mode='bilinear', align_corners=True))
        return [f_4, f_2]
    

    
# v3 - fuse 1/8, 1/4 and return 4_2
class SatlasFPN_4_2_v3(SatlasFPN_8_2_v0):
    """
    FPN Decoder v3: produces decoded feature maps at 1/4 and 1/2 scales.
    - Fuses 1/16 -> 1/8 using concatenation
    - Fuses 1/8 -> 1/4 using concatenation
    """

    def __init__(self, config):
        super().__init__(config)

        out_channels = {
            config['out_channels']['keys'][ik]: config['out_channels']['values'][ik] 
            for ik in range(len(config['out_channels']['keys']))
        }

        # fusion at 1/8: cat [f8 skip, upsampled conv16(f16)]
        self.fuse_8 = nn.Sequential(
            nn.Conv2d(
                2* out_channels['1/8'],   # skip + upsampled
                out_channels['1/8'], kernel_size=3, padding=1, bias=False
            ),
            nn.GroupNorm(1, out_channels['1/8']),
            nn.LeakyReLU(inplace=True)
        )

        # fusion at 1/4: cat [f4 skip, upsampled conv8(f8)]
        self.fuse_4 = nn.Sequential(
            nn.Conv2d(
                2* out_channels['1/4'],   # skip + upsampled
                out_channels['1/4'], kernel_size=3, padding=1, bias=False
            ),
            nn.GroupNorm(1, out_channels['1/4']),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        features = self.satlas_encoder(x)

        # refine 1/16
        f_16 = self.refine_16(features['1/16'])

        # concat fusion at 1/8
        f_8_input = torch.cat([
            self.refine_8(features['1/8']),
            F.interpolate(self.conv_16(f_16), scale_factor=2, mode='bilinear', align_corners=True)
        ], dim=1)
        f_8 = self.fuse_8(f_8_input)

        # concat fusion at 1/4
        f_4_input = torch.cat([
            self.refine_4(features['1/4']),
            F.interpolate(self.conv_8(f_8), scale_factor=2, mode='bilinear', align_corners=True)
        ], dim=1)
        f_4 = self.fuse_4(f_4_input)

        # final 1/2
        f_2 = self.refine_2(
            F.interpolate(f_4, scale_factor=2, mode='bilinear', align_corners=True)
        )

        return [f_4, f_2]
    


def SatlasFPN_4_2(config):
    if "fpn_fusion_type" in config:
        if config["fpn_fusion_type"] == "concat_conv":
            print("USING FPN_FUSION_TYPE CONCAT_CONV")
            return SatlasFPN_4_2_v3(config)
        elif config["fpn_fusion_type"] == "add":
            print("USING FPN_FUSION_TYPE ADD")
            return SatlasFPN_4_2_v0(config)
    else:
        # default is concat + conv
        print("USING FPN_FUSION_TYPE CONCAT_CONV")
        return SatlasFPN_4_2_v3(config)

