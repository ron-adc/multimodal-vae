import pytest
import torch
from src.models.image_encoder import ImageEncoder

@pytest.fixture
def image_encoder():
    return ImageEncoder(in_channels=3, image_size=(64, 64), latent_dim=128)


def test_image_encoder_output_shapes(image_encoder):
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 64, 64)  # (B, C, H, W)
    mu, logvar = image_encoder(input_tensor)

    assert mu.shape == (batch_size, 128), f"Expected mu shape (B, 128), got {mu.shape}"
    assert logvar.shape == (batch_size, 128), f"Expected logvar shape (B, 128), got {logvar.shape}"


def test_image_encoder_forward_pass(image_encoder):
    input_tensor = torch.randn(2, 3, 64, 64)  # (B, C, H, W)
    mu, logvar = image_encoder(input_tensor)

    assert torch.is_tensor(mu), "Output mu is not a tensor"
    assert torch.is_tensor(logvar), "Output logvar is not a tensor"


def test_image_encoder_invalid_input(image_encoder):
    invalid_tensor = torch.randn(2, 1, 64, 64)  # Wrong number of channels
    with pytest.raises(RuntimeError):
        image_encoder(invalid_tensor)


def test_image_encoder_different_image_size():
    encoder = ImageEncoder(in_channels=3, image_size=(128, 128), latent_dim=64)
    input_tensor = torch.randn(2, 3, 128, 128)
    mu, logvar = encoder(input_tensor)

    assert mu.shape == (2, 64), f"Expected mu shape (2, 64), got {mu.shape}"
    assert logvar.shape == (2, 64), f"Expected logvar shape (2, 64), got {logvar.shape}"