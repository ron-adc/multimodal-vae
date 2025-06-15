import torch
from src.models.image_encoder import ImageEncoder

def test_image_encoder_dynamic_input():
    batch_size = 4
    channels = 3
    img_sizes = [(64, 64), (32, 128), (393, 640)]  # Test with different image sizes
    latent_dim = 16

    for img_size in img_sizes:
        encoder = ImageEncoder(in_channels=channels, image_size=img_size, latent_dim=latent_dim)
        x = torch.randn(batch_size, channels, img_size[0], img_size[1])
        mu, logvar = encoder(x)

        assert mu.shape == (batch_size, latent_dim), f"Unexpected mu shape for input size {img_size}: {mu.shape}"
        assert logvar.shape == (batch_size, latent_dim), f"Unexpected logvar shape for input size {img_size}: {logvar.shape}"
