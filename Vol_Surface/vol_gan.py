"""
VolGAN: Generative Adversarial Network for Arbitrage-Free Volatility Surfaces

Based on research:
- VolGAN (2024-2025) - Applied Mathematical Finance
- Computing Volatility Surfaces using GANs with Minimal Arbitrage Violations (2023)

References:
- arXiv:2304.13128
- DOI: 10.1080/1350486X.2025.2471317
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.stats import norm
from scipy.optimize import brentq
import warnings
warnings.filterwarnings('ignore')


class VolatilitySurfaceDataset(Dataset):
    """Dataset for training GAN on volatility surfaces"""

    def __init__(self, surfaces, spot_prices, strike_grid, maturity_grid):
        """
        Args:
            surfaces: (N, K, T) array of implied volatilities
            spot_prices: (N,) array of spot prices
            strike_grid: (K,) array of strike prices
            maturity_grid: (T,) array of maturities
        """
        self.surfaces = torch.FloatTensor(surfaces)
        self.spot_prices = torch.FloatTensor(spot_prices)
        self.strike_grid = torch.FloatTensor(strike_grid)
        self.maturity_grid = torch.FloatTensor(maturity_grid)

    def __len__(self):
        return len(self.surfaces)

    def __getitem__(self, idx):
        return {
            'surface': self.surfaces[idx],
            'spot': self.spot_prices[idx],
            'strikes': self.strike_grid,
            'maturities': self.maturity_grid
        }


class Generator(nn.Module):
    """
    Generator network for volatility surfaces

    Architecture: Conditional GAN with smoothness penalty
    Input: noise vector + spot price condition
    Output: volatility surface (strikes x maturities)
    """

    def __init__(self, latent_dim=100, n_strikes=20, n_maturities=10, hidden_dims=[256, 512, 512, 256]):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.n_strikes = n_strikes
        self.n_maturities = n_maturities
        self.output_dim = n_strikes * n_maturities

        # Condition on spot price (1 additional input)
        input_dim = latent_dim + 1

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, self.output_dim))
        layers.append(nn.Sigmoid())  # Volatility is positive, bounded

        self.model = nn.Sequential(*layers)

    def forward(self, noise, spot_price):
        """
        Generate volatility surface conditioned on spot price

        Args:
            noise: (batch_size, latent_dim)
            spot_price: (batch_size, 1)

        Returns:
            surface: (batch_size, n_strikes, n_maturities)
        """
        x = torch.cat([noise, spot_price], dim=1)
        output = self.model(x)

        # Scale to reasonable volatility range [0.05, 2.0]
        output = 0.05 + output * 1.95

        # Reshape to surface
        surface = output.view(-1, self.n_strikes, self.n_maturities)

        return surface


class Discriminator(nn.Module):
    """
    Discriminator network for volatility surfaces

    Distinguishes real market surfaces from generated ones
    """

    def __init__(self, n_strikes=20, n_maturities=10, hidden_dims=[512, 256, 128]):
        super(Discriminator, self).__init__()

        self.n_strikes = n_strikes
        self.n_maturities = n_maturities
        input_dim = n_strikes * n_maturities + 1  # +1 for spot price condition

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim

        # Output layer - probability of being real
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, surface, spot_price):
        """
        Classify surface as real or fake

        Args:
            surface: (batch_size, n_strikes, n_maturities)
            spot_price: (batch_size, 1)

        Returns:
            probability: (batch_size, 1) - probability of being real
        """
        # Flatten surface
        flat_surface = surface.view(surface.size(0), -1)
        x = torch.cat([flat_surface, spot_price], dim=1)

        return self.model(x)


class ArbitrageLoss:
    """
    Compute arbitrage violation penalties

    Based on:
    - Calendar spread arbitrage
    - Butterfly spread arbitrage
    """

    def __init__(self, lambda_calendar=1.0, lambda_butterfly=1.0):
        self.lambda_calendar = lambda_calendar
        self.lambda_butterfly = lambda_butterfly

    def calendar_spread_penalty(self, surface, maturities):
        """
        Calendar spread arbitrage: total variance should be increasing in maturity

        Total variance w = sigma^2 * T should satisfy:
        w(T2) >= w(T1) for T2 > T1

        Args:
            surface: (batch_size, n_strikes, n_maturities) - implied vols
            maturities: (n_maturities,) - time to maturity in years

        Returns:
            penalty: scalar - sum of violations
        """
        batch_size, n_strikes, n_maturities = surface.shape

        # Compute total variance: sigma^2 * T
        maturities_expanded = maturities.view(1, 1, -1).expand(batch_size, n_strikes, -1)
        total_variance = surface ** 2 * maturities_expanded

        # Check monotonicity: w[t+1] - w[t] should be >= 0
        diff = total_variance[:, :, 1:] - total_variance[:, :, :-1]

        # Penalty for violations (negative differences)
        penalty = torch.sum(torch.relu(-diff))

        return penalty

    def butterfly_spread_penalty(self, surface, strikes):
        """
        Butterfly spread arbitrage: implied variance should be convex in strike

        For strikes K1 < K2 < K3 with equal spacing:
        w(K2) <= (w(K1) + w(K3)) / 2

        Args:
            surface: (batch_size, n_strikes, n_maturities) - implied vols
            strikes: (n_strikes,) - strike prices

        Returns:
            penalty: scalar - sum of violations
        """
        batch_size, n_strikes, n_maturities = surface.shape

        # For each maturity, check convexity across strikes
        penalty = 0.0

        for i in range(1, n_strikes - 1):
            # Central difference approximation of second derivative
            # Should be >= 0 for convexity
            second_deriv = surface[:, i+1, :] - 2 * surface[:, i, :] + surface[:, i-1, :]

            # Penalty for concavity (negative second derivative)
            penalty += torch.sum(torch.relu(-second_deriv))

        return penalty

    def smoothness_penalty(self, surface):
        """
        Smoothness penalty: discourage rapid changes in volatility

        Args:
            surface: (batch_size, n_strikes, n_maturities)

        Returns:
            penalty: scalar - total variation
        """
        # Variation across strikes
        strike_variation = torch.sum((surface[:, 1:, :] - surface[:, :-1, :]) ** 2)

        # Variation across maturities
        maturity_variation = torch.sum((surface[:, :, 1:] - surface[:, :, :-1]) ** 2)

        return strike_variation + maturity_variation

    def total_loss(self, surface, strikes, maturities):
        """
        Combined arbitrage penalty

        Args:
            surface: (batch_size, n_strikes, n_maturities)
            strikes: (n_strikes,)
            maturities: (n_maturities,)

        Returns:
            loss: scalar - weighted sum of penalties
        """
        calendar_penalty = self.calendar_spread_penalty(surface, maturities)
        butterfly_penalty = self.butterfly_spread_penalty(surface, strikes)
        smoothness = self.smoothness_penalty(surface)

        total = (self.lambda_calendar * calendar_penalty +
                self.lambda_butterfly * butterfly_penalty +
                0.01 * smoothness)

        return total, {
            'calendar': calendar_penalty.item() if torch.is_tensor(calendar_penalty) else calendar_penalty,
            'butterfly': butterfly_penalty.item() if torch.is_tensor(butterfly_penalty) else butterfly_penalty,
            'smoothness': smoothness.item() if torch.is_tensor(smoothness) else smoothness
        }


class VolGAN:
    """
    Complete VolGAN system for generating arbitrage-free volatility surfaces
    """

    def __init__(self, latent_dim=100, n_strikes=20, n_maturities=10,
                 lambda_calendar=10.0, lambda_butterfly=10.0, device='cuda'):

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.latent_dim = latent_dim
        self.n_strikes = n_strikes
        self.n_maturities = n_maturities

        # Initialize networks
        self.generator = Generator(latent_dim, n_strikes, n_maturities).to(self.device)
        self.discriminator = Discriminator(n_strikes, n_maturities).to(self.device)

        # Initialize arbitrage loss
        self.arbitrage_loss = ArbitrageLoss(lambda_calendar, lambda_butterfly)

        # Optimizers
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

        # Loss function
        self.criterion = nn.BCELoss()

        # Training history
        self.history = {
            'g_loss': [], 'd_loss': [], 'arbitrage_loss': [],
            'calendar_penalty': [], 'butterfly_penalty': [], 'smoothness': []
        }

    def train_step(self, real_surfaces, spot_prices, strikes, maturities):
        """
        Single training step

        Args:
            real_surfaces: (batch_size, n_strikes, n_maturities)
            spot_prices: (batch_size, 1)
            strikes: (n_strikes,)
            maturities: (n_maturities,)

        Returns:
            losses: dict of loss values
        """
        batch_size = real_surfaces.size(0)

        # Labels for real and fake
        real_labels = torch.ones(batch_size, 1).to(self.device)
        fake_labels = torch.zeros(batch_size, 1).to(self.device)

        # ---------------------
        # Train Discriminator
        # ---------------------
        self.d_optimizer.zero_grad()

        # Real surfaces
        real_output = self.discriminator(real_surfaces, spot_prices)
        d_loss_real = self.criterion(real_output, real_labels)

        # Fake surfaces
        noise = torch.randn(batch_size, self.latent_dim).to(self.device)
        fake_surfaces = self.generator(noise, spot_prices)
        fake_output = self.discriminator(fake_surfaces.detach(), spot_prices)
        d_loss_fake = self.criterion(fake_output, fake_labels)

        # Total discriminator loss
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        self.d_optimizer.step()

        # -----------------
        # Train Generator
        # -----------------
        self.g_optimizer.zero_grad()

        # Generate fake surfaces
        noise = torch.randn(batch_size, self.latent_dim).to(self.device)
        fake_surfaces = self.generator(noise, spot_prices)

        # Generator loss - fool discriminator
        fake_output = self.discriminator(fake_surfaces, spot_prices)
        g_loss_adv = self.criterion(fake_output, real_labels)

        # Arbitrage penalty
        arb_loss, arb_components = self.arbitrage_loss.total_loss(
            fake_surfaces, strikes, maturities
        )

        # Total generator loss
        g_loss = g_loss_adv + arb_loss
        g_loss.backward()
        self.g_optimizer.step()

        return {
            'g_loss': g_loss.item(),
            'd_loss': d_loss.item(),
            'g_loss_adv': g_loss_adv.item(),
            'arbitrage_loss': arb_loss.item() if torch.is_tensor(arb_loss) else arb_loss,
            **arb_components
        }

    def train(self, dataloader, n_epochs=100, verbose=True):
        """
        Train the GAN

        Args:
            dataloader: DataLoader with volatility surfaces
            n_epochs: number of training epochs
            verbose: print progress
        """
        for epoch in range(n_epochs):
            epoch_losses = {
                'g_loss': [], 'd_loss': [], 'arbitrage_loss': [],
                'calendar': [], 'butterfly': [], 'smoothness': []
            }

            for batch in dataloader:
                surfaces = batch['surface'].to(self.device)
                spots = batch['spot'].unsqueeze(1).to(self.device)
                strikes = batch['strikes'][0].to(self.device)
                maturities = batch['maturities'][0].to(self.device)

                losses = self.train_step(surfaces, spots, strikes, maturities)

                # Accumulate losses
                for key in epoch_losses:
                    if key in losses:
                        epoch_losses[key].append(losses[key])

            # Average losses for epoch
            avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}

            # Store in history
            for key in self.history:
                if key in avg_losses:
                    self.history[key].append(avg_losses[key])

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{n_epochs}] "
                      f"G_Loss: {avg_losses['g_loss']:.4f} "
                      f"D_Loss: {avg_losses['d_loss']:.4f} "
                      f"Arb_Loss: {avg_losses['arbitrage_loss']:.4f} "
                      f"Calendar: {avg_losses['calendar']:.4f} "
                      f"Butterfly: {avg_losses['butterfly']:.4f}")

    def generate(self, spot_price, n_samples=1):
        """
        Generate volatility surfaces

        Args:
            spot_price: float or (n_samples,) array
            n_samples: number of surfaces to generate

        Returns:
            surfaces: (n_samples, n_strikes, n_maturities) array
        """
        self.generator.eval()

        with torch.no_grad():
            if isinstance(spot_price, (int, float)):
                spot_price = np.array([spot_price] * n_samples)

            spot_tensor = torch.FloatTensor(spot_price).unsqueeze(1).to(self.device)
            noise = torch.randn(n_samples, self.latent_dim).to(self.device)

            surfaces = self.generator(noise, spot_tensor)

        self.generator.train()

        return surfaces.cpu().numpy()

    def save(self, path):
        """Save model weights"""
        torch.save({
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'g_optimizer': self.g_optimizer.state_dict(),
            'd_optimizer': self.d_optimizer.state_dict(),
            'history': self.history
        }, path)

    def load(self, path):
        """Load model weights"""
        checkpoint = torch.load(path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer'])
        self.history = checkpoint['history']
