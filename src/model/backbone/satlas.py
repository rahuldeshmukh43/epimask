from enum import Enum, auto  
import torch                 
import torch.nn as nn        
from peft import get_peft_model, LoraConfig, TaskType  
import sys
import os
import yaml

# Add the project root (parent of external) to sys.path
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
# sys.path.append(project_root)
# print(f"Updated sys.path: {sys.path}")
# from external.satlaspretrain import satlaspretrain_models
from epimask.external.satlaspretrain import satlaspretrain_models

class SatlasBackbone(nn.Module):
    def __init__(self, config):
        super(SatlasBackbone, self).__init__()
        self.config = config

        # Parse configuration
        self.num_channels = config["in_channels"]
        self.use_default_fpn = config["use_default_fpn"]
        self.use_lora_finetuning = config["use_lora_finetuning"]

        weights_manager = satlaspretrain_models.Weights()
        self.backbone = weights_manager.get_pretrained_model(
            config["pretrained_modelname"],
            fpn=self.use_default_fpn
        )

        # Apply LoRA fine-tuning if enabled
        if self.use_lora_finetuning:    
            lora_conf = {config["lora"]["conf_keys"][ik]: config["lora"]["conf_values"][ik] for ik in range(len(config["lora"]["conf_keys"]))}
            self.lora_config = lora_conf
            # self.lora_config = config["lora"] ---> Uncomment if test from config.yaml else use above 2 lines for using in actual training and testing.
            self.apply_lora()
        self.freeze_base_parameters()  

    def apply_lora(self):
        """Applies LoRA to the backbone's attention layers."""
        lora = self.lora_config
        # Defaults (base for everyone)
        base_r     = lora["rank"]
        base_alpha = lora["alpha"]
        dropout    = lora["dropout"]
        targets    = lora["modules"]

        # Build/merge rank_pattern / alpha_pattern
        rank_pattern  = dict(lora["rank_pattern"])
        alpha_pattern = dict(lora["alpha_pattern"])
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            target_modules=targets,
            r=base_r,
            lora_alpha=base_alpha,
            lora_dropout=dropout,
            rank_pattern=rank_pattern if rank_pattern else None,
            alpha_pattern=alpha_pattern if alpha_pattern else None,
        )
        self.backbone = get_peft_model(self.backbone, lora_config)
        original_forward = self.backbone.base_model.forward

        def new_forward(x):
            return original_forward(x)

        self.backbone.forward = new_forward

    def freeze_base_parameters(self):
        """Freezes all base model parameters except LoRA layers."""
        for name, param in self.backbone.named_parameters():
            if 'lora_' in name:  
                param.requires_grad = True
            else:
                param.requires_grad = False

    def forward(self, x):
        if self.num_channels == 1:
            x = x.repeat(1, 3, 1, 1)
        out = self.backbone(x)
        # try:
        #     out = self.backbone(x)
        # except TypeError:
        #     out = self.backbone(input_ids=x)
        return {'1/4': out[0], '1/8': out[1], '1/16': out[2], '1/32': out[3]}




'''

TESTING CODE FOR SATLAS

'''
# Print the backbone and highlight LoRA-added layers
def print_backbone_with_lora(model):
    for name, module in model.named_modules():
        # Check if the module has LoRA parameters
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            print(f"[LoRA] {name}: {module}")
        else:
            print(f"[No LoRA] {name}: {module}")


# Read YAML Configuration
def load_config(yaml_file):
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
    return config["SatlasConfig"]

def print_lora_params(model):
    for mod_name, mod in model.named_modules():
        if hasattr(mod, "lora_A") and hasattr(mod, "lora_B") and hasattr(mod, "r"):
            adapters = getattr(mod, "r", {})
            for adapter_name, rank in adapters.items():
                alpha = getattr(mod, "lora_alpha", {}).get(adapter_name, None)

                out_features = None
                try:
                    B = mod.lora_B[adapter_name]
                    out_features = getattr(B, "out_features", None)
                except Exception:
                    pass

                print(f"[LoRA] {mod_name:60s} | adapter={adapter_name:8s} "
                      f"| rank={rank:<4} | alpha={alpha} | B.out_features={out_features}")

# Example Usage
if __name__ == "__main__":
    config = load_config("config.yaml")

    model = SatlasBackbone(config)
    print_lora_params(model)
    print(model.backbone.peft_config)
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(n, p.shape)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
