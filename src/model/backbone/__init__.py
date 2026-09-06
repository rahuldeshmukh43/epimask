from .fpn import SatlasFPN_8_2, SatlasFPN_16_4, SatlasFPN_4_2

def build_satlasfpn(config):
    if config["resolution"] == [8, 2]:
        return SatlasFPN_8_2(config["satlas_fpn"])
    elif config["resolution"] == [16, 4]:
        return SatlasFPN_16_4(config["satlas_fpn"])
    elif config["resolution"] == [4, 2]:
        return SatlasFPN_4_2(config["satlas_fpn"])
    else:
        raise ValueError(f"EPIMASK.RESOLUTION {config['resolution']} not supported.")
    return
