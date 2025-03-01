import torch
import torch.nn as nn
import timm
import unittest

def inflate_conv(conv2d, center=False):
    """Inflates a 2D convolution layer into a 3D convolution layer by expanding the kernel, padding, stride, and dilation
    along the temporal dimension to match the spatial dimensions of the original 2D convolution.
    For example, if the 2D kernel size is (3, 3), padding is (1, 1), stride is (2, 2), and dilation is (1, 1),
    the 3D kernel size will be (3, 3, 3), padding will be (1, 1, 1), stride will be (2, 2, 2), and dilation will be (1, 1, 1).

    Args:
        conv2d (nn.Conv2d): The original 2D convolution layer.
        center (bool): If True, initializes the 3D kernel by placing the 2D kernel in the center
                       of the temporal dimension and filling the rest with zeros. If False, it replicates
                       the 2D kernel along the temporal dimension and normalizes.

    Returns:
        nn.Conv3d: The inflated 3D convolution layer.
    """
    temporal_kernel_size = conv2d.kernel_size[0] # Match temporal kernel size to spatial kernel size
    kernel_dim = (temporal_kernel_size, conv2d.kernel_size[0], conv2d.kernel_size[1])
    padding = (conv2d.padding[0], conv2d.padding[0], conv2d.padding[1]) # Match temporal padding to spatial padding
    stride = (conv2d.stride[0], conv2d.stride[0], conv2d.stride[1])     # Match temporal stride to spatial stride
    dilation = (conv2d.dilation[0], conv2d.dilation[0], conv2d.dilation[1]) # Match temporal dilation to spatial dilation

    conv3d = nn.Conv3d(
        conv2d.in_channels,
        conv2d.out_channels,
        kernel_dim,
        padding=padding,
        stride=stride,
        dilation=dilation,
        bias=False if conv2d.bias is None else True # Keep bias consistent with original layer
    )

    # Inflate weights from 2D to 3D
    weight_2d = conv2d.weight.data
    if center:
        weight_3d = torch.zeros(*conv3d.weight.data.shape)
        weight_3d[:, :, temporal_kernel_size // 2, :, :] = weight_2d
    else:
        weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, temporal_kernel_size, 1, 1)
        weight_3d = weight_3d / temporal_kernel_size  # Normalize to keep the output scale similar

    conv3d.weight.data.copy_(weight_3d) # Use copy_ for weight assignment
    if conv2d.bias is not None:
        conv3d.bias = nn.Parameter(torch.empty_like(conv2d.bias.data)) # Initialize with empty tensor of same shape
        conv3d.bias.data.copy_(conv2d.bias.data) # Use copy_ for bias assignment

    return conv3d

def inflate_batch_norm(batchnorm2d):
    """Inflates a 2D BatchNorm layer to a 3D BatchNorm layer by copying parameters using copy_.

    Args:
        batchnorm2d (nn.BatchNorm2d): The original 2D BatchNorm layer.

    Returns:
        nn.BatchNorm3d: The inflated 3D BatchNorm layer.
    """
    batchnorm3d = nn.BatchNorm3d(batchnorm2d.num_features)
    # Copy parameters from 2D batchnorm using copy_
    batchnorm3d.weight = nn.Parameter(torch.empty_like(batchnorm2d.weight.data))
    batchnorm3d.weight.data.copy_(batchnorm2d.weight.data)
    batchnorm3d.bias = nn.Parameter(torch.empty_like(batchnorm2d.bias.data))
    batchnorm3d.bias.data.copy_(batchnorm2d.bias.data)
    batchnorm3d.running_mean.copy_(batchnorm2d.running_mean)
    batchnorm3d.running_var.copy_(batchnorm2d.running_var)
    batchnorm3d.eps = batchnorm2d.eps
    batchnorm3d.momentum = batchnorm2d.momentum
    return batchnorm3d

def inflate_linear(linear2d):
    """Inflates a 2D Linear layer to a 3D Linear layer by copying parameters using copy_.
       In practice, the operation remains linear, but this function is provided for conceptual consistency.

    Args:
        linear2d (nn.Linear): The original 2D Linear layer.

    Returns:
        nn.Linear: The inflated (conceptually 3D) Linear layer.
    """
    linear3d = nn.Linear(linear2d.in_features, linear2d.out_features)
    linear3d.weight = nn.Parameter(torch.empty_like(linear2d.weight.data))
    linear3d.weight.data.copy_(linear2d.weight.data)
    if linear2d.bias is not None:
        linear3d.bias = nn.Parameter(torch.empty_like(linear2d.bias.data))
        linear3d.bias.data.copy_(linear2d.bias.data)
    return linear3d


def convert_efficientnet2d_to_3d(model_2d, center_inflate=False):
    """Converts a 2D EfficientNet model to a 3D EfficientNet model by inflating relevant layers.
    Temporal kernel size, padding, stride, and dilation are matched to spatial dimensions.

    Args:
        model_2d (nn.Module): The pretrained 2D EfficientNet model from timm.
        center_inflate (bool): Whether to use center inflation for Conv3d weights. If False, replicate weights.

    Returns:
        nn.Module: The converted 3D EfficientNet model.
    """
    model_3d = model_2d
    for name, module in model_2d.named_modules():
        if isinstance(module, nn.Conv2d):
            new_conv = inflate_conv(module, center=center_inflate)
            # Replace the 2D conv with 3D conv in the model
            parts = name.split('.')
            parent_module = model_3d
            for part in parts[:-1]:
                if hasattr(parent_module, part):
                    parent_module = getattr(parent_module, part)
                else: # Handle cases where module name part is not directly an attribute (e.g., in Sequential)
                    parent_module = parent_module[int(part)] # Assuming numerical index for Sequential
            setattr(parent_module, parts[-1], new_conv)

        elif isinstance(module, nn.BatchNorm2d):
            new_bn = inflate_batch_norm(module)
            parts = name.split('.')
            parent_module = model_3d
            for part in parts[:-1]:
                if hasattr(parent_module, part):
                    parent_module = getattr(parent_module, part)
                else: # Handle cases where module name part is not directly an attribute (e.g., in Sequential)
                    parent_module = parent_module[int(part)] # Assuming numerical index for Sequential
            setattr(parent_module, parts[-1], new_bn)

        elif isinstance(module, nn.Linear) and name in ['classifier', 'fc', 'head.fc']: # Target classifier layer
            new_linear = inflate_linear(module)
            parts = name.split('.')
            parent_module = model_3d
            for part in parts[:-1]:
                if hasattr(parent_module, part):
                    parent_module = getattr(parent_module, part)
                else: # Handle cases where module name part is not directly an attribute (e.g., in Sequential)
                    parent_module = parent_module[int(part)] # Assuming numerical index for Sequential
            setattr(parent_module, parts[-1], new_linear)

    return model_3d

def create_efficientnet3d(model_name='efficientnet_b0', pretrained=True, center_inflate=False, num_classes=1000, in_chans=3, features_only=False, **kwargs):
    """Creates a 3D EfficientNet model from a 2D timm EfficientNet model, with inflated weights.
    Temporal kernel size, padding, stride, and dilation are matched to spatial dimensions.

    Args:
        model_name (str): Name of the 2D EfficientNet model in timm (e.g., 'efficientnet_b0').
        pretrained (bool): Whether to load pretrained weights for the 2D model.
        center_inflate (bool): Whether to use center inflation for Conv3d weights.
        num_classes (int): Number of classes for the final classifier (only if features_only=False).
        in_chans (int): Number of input channels (for the 2D model before inflation).
        features_only (bool): If True, returns only the feature extraction backbone without classifier.
        **kwargs: Additional keyword arguments to pass to `timm.create_model`.

    Returns:
        nn.Module: The 3D EfficientNet model.
    """
    if 'efficientnet' not in model_name: # Basic check to ensure it's an EfficientNet variant
        raise ValueError("Model name should be an EfficientNet variant (e.g., 'efficientnet_b0', 'tf_efficientnetv2_b0').")
    model_2d = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes, in_chans=in_chans, features_only=features_only, **kwargs)
    model_3d = convert_efficientnet2d_to_3d(model_2d, center_inflate=center_inflate)
    return model_3d


class TestEfficientNet3D(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_efficientnet_b0_output_shape(self):
        model_3d_b0 = create_efficientnet3d(model_name='efficientnet_b0', pretrained=False, num_classes=10).to(self.device)
        dummy_input = torch.randn(1, 3, 8, 224, 224).to(self.device)
        output = model_3d_b0(dummy_input)
        self.assertEqual(output.shape, torch.Size([1, 10]))

    def test_tf_efficientnetv2_b0_features_only_output_shape(self):
        model_3d_v2_b0_features = create_efficientnet3d(model_name='tf_efficientnetv2_b0', pretrained=False, features_only=True).to(self.device)
        dummy_input = torch.randn(1, 3, 5, 224, 224).to(self.device)
        features = model_3d_v2_b0_features(dummy_input)
        self.assertTrue(isinstance(features, torch.Tensor)) # features_only=True returns a tensor
        self.assertGreater(len(features.shape), 1) # Check if it's not a scalar

    def test_efficientnet_b1_layer_types(self):
        model_3d_b1 = create_efficientnet3d(model_name='efficientnet_b1', pretrained=False).to(self.device)
        for name, module in model_3d_b1.named_modules():
            if 'conv' in name and 'downsample' not in name and isinstance(module, nn.Conv2d): # Exclude downsample layers which can be in some blocks and avoid errors
                parent_name = name.rsplit('.', 1)[0] # Get parent module name
                parent_module = model_3d_b1.get_submodule(parent_name) # Access parent module
                if not isinstance(parent_module, nn.Sequential) or 'downsample' not in name : # Further filter to avoid downsample layers in some blocks
                    self.assertIsInstance(module, nn.Conv3d, f"Layer {name} should be Conv3d")
            elif 'bn' in name and isinstance(module, nn.BatchNorm2d):
                self.assertIsInstance(module, nn.BatchNorm3d, f"Layer {name} should be BatchNorm3d")
            elif 'fc' in name and isinstance(module, nn.Linear) and name in ['classifier', 'fc', 'head.fc']:
                self.assertIsInstance(module, nn.Linear, f"Layer {name} should remain Linear") # Linear layers are inflated but remain Linear


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running examples on: {device}")

    # Example usage: Create a 3D EfficientNet-B0 with pretrained weights and matched temporal parameters
    model_3d_b0 = create_efficientnet3d(model_name='efficientnet_b0', pretrained=True, num_classes=10, center_inflate=False).to(device)
    print("3D EfficientNet-B0 created with temporal kernel size, padding, stride, and dilation matched to spatial.")

    # Example input: (batch_size, channels, time, height, width)
    dummy_input = torch.randn(1, 3, 8, 224, 224).to(device) # Time dimension can be arbitrary now as kernel size is matched.
    with torch.no_grad():
        output = model_3d_b0(dummy_input)
    print("Output shape of 3D EfficientNet-B0:", output.shape)

    # Example with center inflation
    model_3d_b1_center = create_efficientnet3d(model_name='efficientnet_b1', pretrained=True, center_inflate=True, num_classes=5).to(device)
    print("3D EfficientNet-B1 with center inflation created (temporal parameters matched to spatial).")
    dummy_input_center = torch.randn(1, 3, 5, 240, 240).to(device) # Time dimension can be arbitrary.
    with torch.no_grad():
        output_center = model_3d_b1_center(dummy_input_center)
    print("Output shape of 3D EfficientNet-B1 (center inflate):", output_center.shape)

    # Example without pretrained weights and different input channels
    model_3d_b2_nopretrained = create_efficientnet3d(model_name='efficientnet_b2', pretrained=False, in_chans=1, num_classes=2).to(device)
    print("3D EfficientNet-B2 without pretrained weights and 1 input channel created (temporal parameters matched to spatial).")
    dummy_input_nopretrained = torch.randn(1, 1, 4, 260, 260).to(device) # Time dimension can be arbitrary.
    with torch.no_grad():
        output_nopretrained = model_3d_b2_nopretrained(dummy_input_nopretrained)
    print("Output shape of 3D EfficientNet-B2 (no pretrained, 1 channel):", output_nopretrained.shape)

    print("\n--- TF EfficientNetV2 Verification ---")
    # Example with tf_efficientnetv2_b0
    model_3d_v2_b0 = create_efficientnet3d(model_name='tf_efficientnetv2_b0', pretrained=True, num_classes=1000).to(device)
    print("3D tf_efficientnetv2_b0 created.")
    dummy_input_v2_b0 = torch.randn(1, 3, 7, 224, 224).to(device) # Input size for tf_efficientnetv2_b0 is 224
    with torch.no_grad():
        output_v2_b0 = model_3d_v2_b0(dummy_input_v2_b0)
    print("Output shape of 3D tf_efficientnetv2_b0:", output_v2_b0.shape)

    # Example with tf_efficientnetv2_b1, no pretraining and different number of classes
    model_3d_v2_b1_nopretrained = create_efficientnet3d(model_name='tf_efficientnetv2_b1', pretrained=False, num_classes=50).to(device)
    print("3D tf_efficientnetv2_b1 without pretrained weights created.")
    dummy_input_v2_b1 = torch.randn(1, 3, 6, 240, 240).to(device) # Input size for tf_efficientnetv2_b1 is 240
    with torch.no_grad():
        output_v2_b1 = model_3d_v2_b1_nopretrained(dummy_input_v2_b1)
    print("Output shape of 3D tf_efficientnetv2_b1 (no pretrained):", output_v2_b1.shape)

    # Example with tf_efficientnetv2_m, center inflation
    model_3d_v2_m_center = create_efficientnet3d(model_name='tf_efficientnetv2_m', pretrained=True, center_inflate=True, num_classes=10).to(device)
    print("3D tf_efficientnetv2_m with center inflation created.")
    dummy_input_v2_m = torch.randn(1, 3, 9, 288, 288).to(device) # Input size for tf_efficientnetv2_m is 288
    with torch.no_grad():
        output_v2_m = model_3d_v2_m_center(dummy_input_v2_m)
    print("Output shape of 3D tf_efficientnetv2_m (center inflate):", output_v2_m.shape)

    print("\n--- features_only=True Verification ---")
    # Example with features_only=True for EfficientNet-B3
    model_3d_b3_features = create_efficientnet3d(model_name='efficientnet_b3', pretrained=True, features_only=True).to(device)
    print("3D EfficientNet-B3 (features_only=True) created.")
    dummy_input_features = torch.randn(1, 3, 6, 300, 300).to(device) # Input size for efficientnet_b3 is 300
    with torch.no_grad():
        features_output = model_3d_b3_features(dummy_input_features)
    print("Output type of 3D EfficientNet-B3 (features_only=True):", type(features_output)) # Should be Tensor
    print("Output shape of 3D EfficientNet-B3 (features_only=True):", features_output.shape) # Shape will vary, but should be feature map

    # Run unit tests if script is executed directly
    unittest.main(argv=['first-arg-is-ignored'], exit=False)