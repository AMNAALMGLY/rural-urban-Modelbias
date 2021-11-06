from models.resnet import resnet18, resnet34, resnet50

model_type = dict(resnet18=resnet18,
                  resnet34=resnet34,
                  resnet50=resnet50)


def get_model(model_name, in_channels, pretrained):
    model_fn = model_type[model_name]
    model = model_fn(in_channels, pretrained)
    return model
