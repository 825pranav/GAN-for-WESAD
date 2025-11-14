import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os

# --- 1. GPU Diagnostic & Device Setup ---
print("--- PyTorch GPU Diagnostic ---")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"Found GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Warning: No GPU found by PyTorch. Using CPU.")
print(f"Using device: {device}")
print("----------------------------\n")

# --- 2. Load Data and Define Hyperparameters ---

# Check if data files exist
if not os.path.exists('baseline_seq.npy') or not os.path.exists('stress_seq.npy'):
    print("Error: 'baseline_seq.npy' or 'stress_seq.npy' not found.")
    print("Please run main.py first to generate them.")
    exit()

# Load preprocessed WESAD sequences
baseline_data = np.load('baseline_seq.npy')
stress_data = np.load('stress_seq.npy')
full_data = np.concatenate([baseline_data, stress_data], axis=0).astype(np.float32)

# --- Data parameters ---
SEQ_LEN = full_data.shape[1]     # e.g., 70
N_FEATURES = full_data.shape[2]  # e.g., 3

# --- Model parameters ---
HIDDEN_DIM = 24
NOISE_DIM = 100
BATCH_SIZE = 128
K = 1  # D steps per G step

# --- Training Epochs ---
EPOCHS_AE = 20 # Autoencoder training (already done, will load)
GAN_EPOCHS = 1000 # Full 1000 epoch run

print(f"Data Shapes (Combined): {full_data.shape}")
print(f"Seq Len: {SEQ_LEN}, Features: {N_FEATURES}, Hidden: {HIDDEN_DIM}\n")

# Convert data to PyTorch tensors
full_data_tensor = torch.tensor(full_data)
train_dataset = TensorDataset(full_data_tensor)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# ============================================================
# 3. Autoencoder Components (NOW UNIDIRECTIONAL)
# ============================================================

class Encoder(nn.Module):
    def __init__(self, seq_len, n_features, hidden_dim):
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False # --- SIMPLIFIED ---
        )

    def forward(self, x):
        output, _ = self.gru(x)
        return output

class Decoder(nn.Module):
    def __init__(self, seq_len, n_features, hidden_dim):
        super().__init__()
        self.gru = nn.GRU(
            input_size=hidden_dim, # --- SIMPLIFIED (was * 2) ---
            hidden_size=hidden_dim, # --- SIMPLIFIED (was * 2) ---
            num_layers=1,
            batch_first=True,
            bidirectional=False # --- SIMPLIFIED ---
        )
        self.fc = nn.Linear(hidden_dim, n_features) # --- SIMPLIFIED (was * 4) ---

    def forward(self, x):
        output, _ = self.gru(x)
        output = self.fc(output)
        return torch.sigmoid(output)  # keep within [0,1]

class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        latent = self.encoder(x)
        return self.decoder(latent)

# ============================================================
# 4. Generator and Discriminator (NOW UNIDIRECTIONAL)
# ============================================================

class Generator(nn.Module):
    def __init__(self, seq_len, hidden_dim, noise_dim):
        super().__init__()
        self.gru = nn.GRU(
            input_size=noise_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False # --- SIMPLIFIED ---
        )
        self.fc = nn.Linear(hidden_dim, hidden_dim) # --- SIMPLIFIED (was * 2) ---

    def forward(self, x):
        output, _ = self.gru(x)
        output = self.fc(output)
        return torch.tanh(output)  # symmetric latent space

class Discriminator(nn.Module):
    def __init__(self, seq_len, hidden_dim):
        super().__init__()
        self.gru = nn.GRU(
            input_size=hidden_dim, # --- SIMPLIFIED (was * 2) ---
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False # --- SIMPLIFIED ---
        )
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(hidden_dim, 1) # --- SIMPLIFIED (was * 2) ---

    def forward(self, x):
        _, hn = self.gru(x)
        # --- SIMPLIFIED (no need to cat) ---
        # hn[0] is shape (batch_size, hidden_dim)
        hn_last = self.dropout(hn[0]) 
        return torch.sigmoid(self.fc(hn_last))

# ============================================================
# 5. Initialize Models, Loss, and Optimizers
# ============================================================
encoder = Encoder(SEQ_LEN, N_FEATURES, HIDDEN_DIM).to(device)
decoder = Decoder(SEQ_LEN, N_FEATURES, HIDDEN_DIM).to(device)
autoencoder = Autoencoder(encoder, decoder).to(device)

generator = Generator(SEQ_LEN, HIDDEN_DIM, NOISE_DIM).to(device)
discriminator = Discriminator(SEQ_LEN, HIDDEN_DIM).to(device)

criterion_ae = nn.MSELoss()
criterion_gan = nn.MSELoss() # Use stable LSGAN loss

optimizer_ae = optim.Adam(autoencoder.parameters(), lr=0.001)
# Use 2:1 LR ratio for Generator:Discriminator
optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

# ============================================================
# 6. Helper Functions
# ============================================================

def make_random_noise(batch_size, seq_len, noise_dim):
    # Gaussian noise instead of uniform
    return torch.randn(batch_size, seq_len, noise_dim, device=device)

def clip_grads(model):
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# ============================================================
# 7. Train Autoencoder
# ============================================================
print("\n--- Training Autoencoder ---")
# --- Check if AE weights already exist ---
# We MUST retrain the autoencoder because the architecture has changed
if os.path.exists('autoencoder.pth'):
    print("Old autoencoder weights found, but architecture changed. Retraining...")
    
for epoch in range(EPOCHS_AE):
    for batch in train_loader:
        real_batch = batch[0].to(device)
        reconstruction = autoencoder(real_batch)
        loss = criterion_ae(reconstruction, real_batch)
        optimizer_ae.zero_grad()
        loss.backward()
        clip_grads(autoencoder)
        optimizer_ae.step()
    print(f"AE Epoch [{epoch+1}/{EPOCHS_AE}] | Loss: {loss.item():.6f}")

print("\n--- Autoencoder Training Complete ---")
# Save weights
torch.save(autoencoder.state_dict(), 'autoencoder.pth')

# Ensure standalone models have the weights
encoder.load_state_dict(autoencoder.encoder.state_dict())
decoder.load_state_dict(autoencoder.decoder.state_dict())


# ============================================================
# 8. Adversarial Training
# ============================================================
print(f"\n--- Starting Adversarial Training ({GAN_EPOCHS} epochs) ---")

for epoch in range(GAN_EPOCHS):
    for batch in train_loader:
        real_batch = batch[0].to(device)
        bs = real_batch.shape[0]

        # Encode real batch
        real_latent = encoder(real_batch).detach()
        real_latent += 0.05 * torch.randn_like(real_latent)  # noise regularization

        # Train Discriminator
        for _ in range(K):
            real_labels = torch.ones(bs, 1).to(device) # LSGAN uses 1 and 0
            fake_labels = torch.zeros(bs, 1).to(device)

            noise = make_random_noise(bs, SEQ_LEN, NOISE_DIM)
            fake_latent = generator(noise).detach()
            fake_latent += 0.05 * torch.randn_like(fake_latent)

            d_real = discriminator(real_latent)
            d_fake = discriminator(fake_latent)
            loss_d = (criterion_gan(d_real, real_labels) + criterion_gan(d_fake, fake_labels)) / 2

            optimizer_d.zero_grad()
            loss_d.backward()
            clip_grads(discriminator)
            optimizer_d.step()

        # Train Generator
        noise = make_random_noise(bs, SEQ_LEN, NOISE_DIM)
        fake_latent = generator(noise)
        g_labels = torch.ones(bs, 1).to(device)
        d_out = discriminator(fake_latent)
        loss_g = criterion_gan(d_out, g_labels)

        # Feature-matching regularization
        fake_decoded = decoder(fake_latent)
        reencoded = encoder(fake_decoded)
        fm_loss = nn.MSELoss()(fake_latent, reencoded)
        loss_g += 0.1 * fm_loss

        optimizer_g.zero_grad()
        loss_g.backward()
        clip_grads(generator)
        optimizer_g.step()

    # Logging (print every 10 epochs for a cleaner log)
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1}/{GAN_EPOCHS}] | D Loss: {loss_d.item():.4f} | G Loss: {loss_g.item():.4f}")

print("\n--- Adversarial Training Complete ---")

# ============================================================
# 9. Save Final Models
# ============================================================
torch.save(generator.state_dict(), 'generator.pth')
torch.save(discriminator.state_dict(), 'discriminator.pth')
print("Saved final Generator and Discriminator weights.\n")

# ============================================================
# 10. Generate Synthetic Data
# ============================================================
print("--- Generating Synthetic Data ---")
generator.eval()
decoder.eval()

num_samples = 1000
with torch.no_grad():
    noise = make_random_noise(num_samples, SEQ_LEN, NOISE_DIM)
    synthetic_latent = generator(noise)
    synthetic_data_tensor = decoder(synthetic_latent)

synthetic_data = synthetic_data_tensor.cpu().numpy()
np.save('synthetic_wesad_data.npy', synthetic_data)
print(f"Generated and saved {num_samples} synthetic samples to 'synthetic_wesad_data.npy'.")