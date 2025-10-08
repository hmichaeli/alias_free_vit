import torchvision
from timm.models.registry import register_model


@register_model
def swin_tiny_pretrained(**kwargs):
    model = torchvision.models.swin_t(weights=torchvision.models.Swin_T_Weights.DEFAULT)
    return model

@register_model
def vit_base_pretrained(**kwargs):
    model = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.DEFAULT)
    return model

@register_model
def cvt_13(**kwargs):
    try:
        import CvT.tools._init_paths
        import config as config_lib
        from CvT.lib.models.build import build_model
    except ImportError:
        print("CvT not available. Use git submodule update --init --recursive")
        raise ImportError
    cfg_file = "./CvT/experiments/imagenet/cvt/cvt-13-224x224.yaml"
    config_lib._update_config_from_file(config_lib.config, cfg_file)
    model = build_model(config_lib.config)
    return model

