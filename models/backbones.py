import timm
import torch
import torch.nn as nn

class Backbone(nn.Module):
    def __init__(self, model_name='efficientnet_b3', pretrained=True, output_stride=32, out_indices=(1, 4)):
        super(Backbone, self).__init__()
        kwargs = {
            'features_only': True,
            'out_indices': out_indices,
            'pretrained': pretrained
        }
        # Only pass output_stride if supported by model (most timm models support it via kwargs or model args)
        # However, some models might not.
        # Safe way: try passing it, if fails, catch? No, too risky.
        # EfficientNet and ConvNeXt generally support it.
        try:
            self.model = timm.create_model(model_name, output_stride=output_stride, **kwargs)
        except TypeError:
            # Fallback if output_stride is not supported
            print(f"Warning: {model_name} does not support output_stride, using default.")
            self.model = timm.create_model(model_name, **kwargs)

        # Get channel information dynamically
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 256, 256)
            features = self.model(dummy_input)
            self.channels = [f.shape[1] for f in features]

    def forward(self, x):
        return self.model(x)

    def get_channels(self):
        return self.channels
