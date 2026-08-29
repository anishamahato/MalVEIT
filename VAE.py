import torch
import torch.nn as nn


class VAEBottleneck(nn.Module):
    """
    VAE Bottleneck for MalVEIT

    Input:
        x : (B, H, W, C)

    Output:
        z      : latent representation
        mu     : mean of latent distribution
        logvar : log variance of latent distribution
    """

    def __init__(self, input_dim=768, latent_dim=256):

        super().__init__()

        # Mean projection
        self.fc_mu = nn.Linear(
            input_dim,
            latent_dim
        )

        # Log-variance projection
        self.fc_logvar = nn.Linear(
            input_dim,
            latent_dim
        )

    def forward(self, x):

        B, H, W, C = x.shape

        # Global average pooling over spatial dimensions
        x = x.mean(dim=(1, 2))

        # μ
        mu = self.fc_mu(x)

        # log(σ²)
        logvar = self.fc_logvar(x)

        # Standard deviation
        std = torch.exp(
            0.5 * logvar
        )

        # Random noise
        eps = torch.randn_like(std)

        # Reparameterization
        z = mu + std * eps

        return z, mu, logvar



class VAEDecoder(nn.Module):
    """
    VAE Decoder for MalVEIT

    Input:
        z : (B, latent_dim)

    Output:
        reconstruction : (B, 3, 224, 224)
    """

    def __init__(
        self,
        latent_dim=256,
        output_channels=3
    ):

        super().__init__()

        # Project latent vector to spatial feature map
        self.fc = nn.Linear(
            latent_dim,
            7 * 7 * 256
        )

        # Decoder blocks
        self.decoder = nn.Sequential(

            # 7 × 7 → 14 × 14
            nn.ConvTranspose2d(
                256,
                256,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # 14 × 14 → 28 × 28
            nn.ConvTranspose2d(
                256,
                128,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 28 × 28 → 56 × 56
            nn.ConvTranspose2d(
                128,
                64,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 56 × 56 → 112 × 112
            nn.ConvTranspose2d(
                64,
                32,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # 112 × 112 → 224 × 224
            nn.ConvTranspose2d(
                32,
                output_channels,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            # Input images assumed to be normalized to [0,1]
            nn.Sigmoid()
        )

    def forward(self, z):

        # Latent vector → spatial feature map
        x = self.fc(z)

        # Reshape
        x = x.view(
            z.size(0),
            256,
            7,
            7
        )

        # Decode
        reconstruction = self.decoder(x)

        return reconstruction
