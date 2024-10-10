import warnings
from dynamic_network_architectures.architectures.unet import PlainConvUNet, ResidualEncoderUNet


class CustomModel(PlainConvUNet):
    def __init__(self, *args, **kwargs):
        super(CustomModel, self).__init__(*args, **kwargs)
        print(f"==> Using custom model from {__file__}")

        self.first_iter = True

    def forward(self, x):
        if self.first_iter:
            warnings.warn(f"==> Model input shape is {x.shape}")
            self.first_iter = False
        return super(CustomModel, self).forward(x)
