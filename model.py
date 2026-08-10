import torch
import torch.nn as nn

LATENT_DIM = 32
N_ACTIONS = 4


class Encoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 4, stride=2, padding=1), nn.ReLU(),   # 48->24
            nn.Conv2d(16, 32, 4, stride=2, padding=1), nn.ReLU(),  # 24->12
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),  # 12->6
        )
        self.fc = nn.Linear(64 * 6 * 6, latent_dim)

    def forward(self, x):
        h = self.net(x)
        h = h.flatten(1)
        return self.fc(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 64 * 6 * 6)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),  # 6->12
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.ReLU(),  # 12->24
            nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1), nn.Sigmoid(),  # 24->48
        )

    def forward(self, z):
        h = self.fc(z).view(-1, 64, 6, 6)
        return self.net(h)


class Transition(nn.Module):
    """Predicts the next latent given current latent + action."""
    def __init__(self, latent_dim=LATENT_DIM, n_actions=N_ACTIONS):
        super().__init__()
        self.action_emb = nn.Embedding(n_actions, 16)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 16, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, latent_dim),
        )

    def forward(self, z, action):
        a = self.action_emb(action)
        h = torch.cat([z, a], dim=-1)
        return z + self.net(h)  # residual: predict the delta


class WorldModel(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, n_actions=N_ACTIONS):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.transition = Transition(latent_dim, n_actions)

    def forward(self, frame, action):
        z = self.encoder(frame)
        z_next = self.transition(z, action)
        pred_next_frame = self.decoder(z_next)
        return pred_next_frame, z, z_next

    def imagine_step(self, z, action):
        z_next = self.transition(z, action)
        frame_next = self.decoder(z_next)
        return z_next, frame_next
