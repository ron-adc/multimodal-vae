import torch
import torch.nn as nn


class ImageEncoder(nn.Module):
    """
    Simple CNN encoder that maps an image tensor to a latent mean and log-variance vector.
    """
    def __init__(self, in_channels: int = 3, image_size: tuple[int, int] = (64, 64), latent_dim: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1)  # Reduce HxW
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.flatten = nn.Flatten()

        self.encoder_cnn = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            self.conv2,
            nn.ReLU(),
            self.conv3,
            nn.ReLU(),
            self.flatten
        )
        feat_size = 256 * (image_size[0] // 8) * (image_size[1] // 8)  # Assuming 3 downsampling layers with stride=2 (2^3)

        # Linear layers to produce mean and logvar
        self.fc_mu = nn.Linear(feat_size, latent_dim)
        self.fc_logvar = nn.Linear(feat_size, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Image tensor of shape (B, C, H, W)
        Returns:
            mu: Latent mean tensor (B, latent_dim)
            logvar: Latent log-variance tensor (B, latent_dim)
        """
        h = self.encoder_cnn(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
