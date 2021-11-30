from models.preact_resnet import PreActResNet18, PreActResNet34, PreActResNet50
from utils.utils import load_from_checkpoint

model_type = dict(resnet18=PreActResNet18,
                  resnet34=PreActResNet34,
                  resnet50=PreActResNet50)


def get_model(model_name, in_channels, pretrained=False,ckpt_path=None):
    model_fn = model_type[model_name]
    model = model_fn(in_channels, pretrained)
    if ckpt_path:
        model=load_from_checkpoint(ckpt_path,model)
    return model
