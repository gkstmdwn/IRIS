import torch

def gray_to_rgb_tensor(x_gray:torch.Tensor, normalize:bool = True) -> torch.Tensor:
    """
    Turns gray(1 channel) Tensor Batch to rgb(3 channel) Tensor Batch.
    The input shape must look like [B, 1, H, W] where B is batch size, H is image Height, W is image Width.
    The pixel values should be in range [0, 1].
    if Pixel values are in range [0, 255] with uint8, they must be normalized to fit the range before applying this function.
    The return shape looks like [B, 3, H, W].

    Args:
        x_gray (torch.Tensor): [B, 1, H, W] shaped. Pixel should be in range [0, 1].
        normalize (bool, optional): if false, does not regularize to ImageNet Dataset. Defaults to True.

    Returns:
        torch.Tensor: [B, 3, H, W] shaped Tensor, with ImageNet Dataset Regularization.
    """
    x_rgb = x_gray.repeat(1, 3, 1, 1)
    
    if normalize == False:
        return x_rgb
    
    mean = torch.tensor([0.485, 0.456, 0.406], device=x_rgb.device).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=x_rgb.device).view(1,3,1,1)
    
    return (x_rgb - mean) / std

