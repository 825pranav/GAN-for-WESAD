
import tensorflow as tf
import os
import numpy as np

# --- Add this diagnostic code ---
print("--- GPU Diagnostic ---")
print(f"TensorFlow Version: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    print(f"Found {len(gpus)} GPUs:")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu}")
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            print("    - Memory growth enabled")
        except RuntimeError as e:
            print(f"    - ERROR: {e}")
else:
    print("Error: No GPUs found by TensorFlow.")

print("----------------------\n")
# --- End of diagnostic code ---

# Your existing script continues below...import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dense, Bidirectional
from tensorflow.keras.optimizers import Adam
import os

# --- 1. Load Data and Define Hyperparameters ---

# Check if data files exist
if not os.path.exists('baseline_seq.npy') or not os.path.exists('stress_seq.npy'):
    print("Error: 'baseline_seq.npy' or 'stress_seq.npy' not found.")
    print("Please run main.py one last time to save these files.")
    exit()

# Load our processed data
baseline_data = np.load('baseline_seq.npy')
stress_data = np.load('stress_seq.npy')

# For training the autoencoder, we can use all data
full_data = np.concatenate([baseline_data, stress_data], axis=0)

# Shuffle the data
np.random.shuffle(full_data)

# --- Data parameters ---
# (num_windows, seq_len, num_features)
SEQ_LEN = full_data.shape[1]     # 70
N_FEATURES = full_data.shape[2]  # 3

# --- Model parameters ---
HIDDEN_DIM = 24  # Size of the latent space (GRU hidden state)
NOISE_DIM = 100  # Size of the random noise vector for the generator
BATCH_SIZE = 128
EPOCHS_AE = 100  # We'll train the Autoencoder first for 100 epochs

print(f"Data Shapes (Combined): {full_data.shape}")
print(f"Seq Len: {SEQ_LEN}, Features: {N_FEATURES}")
print(f"Latent Dim: {HIDDEN_DIM}\n")


# --- 2. Build the Autoencoder Components ---

def build_encoder(seq_len, n_features, hidden_dim):
    """
    Builds the Encoder network.
    Input: (seq_len, n_features)
    Output: (seq_len, hidden_dim)
    """
    input_layer = Input(shape=(seq_len, n_features), name="Encoder_Input")
    
    # Using GRU (Gated Recurrent Unit) - simpler than LSTM
    # return_sequences=True makes it output at each time step
    # This creates the "latent space" H
    gru_layer = GRU(hidden_dim, return_sequences=True, name="Encoder_GRU")
    
    # We use Bidirectional to capture patterns in both forward and reverse time
    # This is optional but often helps.
    bidirectional_gru = Bidirectional(gru_layer, name="Encoder_Bidirectional_GRU")
    
    encoder_output = bidirectional_gru(input_layer)
    
    # The Encoder model
    encoder = Model(input_layer, encoder_output, name="Encoder")
    return encoder

def build_decoder(seq_len, n_features, hidden_dim):
    """
    Builds the Decoder network.
    Input: (seq_len, hidden_dim * 2) <- *2 because Bidirectional doubles it
    Output: (seq_len, n_features)
    """
    # Input shape must match the Encoder's output
    input_layer = Input(shape=(seq_len, hidden_dim * 2), name="Decoder_Input")
    
    gru_layer = GRU(hidden_dim * 2, return_sequences=True, name="Decoder_GRU")
    bidirectional_gru = Bidirectional(gru_layer, name="Decoder_Bidirectional_GRU")
    
    decoder_hidden = bidirectional_gru(input_layer)
    
    # A Dense layer applied to every time step to map back to n_features
    # 'sigmoid' activation is used because our data is scaled 0-1
    output_layer = Dense(n_features, activation='sigmoid', name="Decoder_Output_Dense")
    
    # TimeDistributed applies the Dense layer to each of the 70 time steps
    decoder_output = tf.keras.layers.TimeDistributed(output_layer, name="Decoder_Output")(decoder_hidden)
    
    # The Decoder model
    decoder = Model(input_layer, decoder_output, name="Decoder")
    return decoder

# --- 3. Build and Compile the Combined Autoencoder ---

# Build the components
encoder = build_encoder(SEQ_LEN, N_FEATURES, HIDDEN_DIM)
decoder = build_decoder(SEQ_LEN, N_FEATURES, HIDDEN_DIM)

print("--- Encoder Summary ---")
encoder.summary()
print("\n--- Decoder Summary ---")
decoder.summary()

# Combine them into one "Autoencoder" model for easy training
autoencoder_input = Input(shape=(SEQ_LEN, N_FEATURES), name="AE_Input")
latent_representation = encoder(autoencoder_input)
reconstructed_signal = decoder(latent_representation)

autoencoder = Model(autoencoder_input, reconstructed_signal, name="Autoencoder")

# Compile the Autoencoder
# We use 'binary_crossentropy' because our output is sigmoid (0-1)
# This measures how well the output reconstructs the input

#changed Binary to MSE 
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

print("\n--- Autoencoder (Combined) Summary ---")
autoencoder.summary()

# --- 4. Train the Autoencoder ---
# We train the autoencoder to learn the latent representation first.
# This stabilizes the full TimeGAN training later.
print("\n--- Training Autoencoder ---")

# We train the autoencoder to reconstruct its own input
# X (features) = full_data
# y (target) = full_data
history = autoencoder.fit(
    full_data,
    full_data,
    epochs=EPOCHS_AE,
    batch_size=BATCH_SIZE,
    validation_split=0.1,  # Use 10% of data for validation
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='min')
    ]
)

print("\n--- Autoencoder Training Complete ---")

# We can optionally save the weights
encoder.save_weights('encoder_weights.h5')
decoder.save_weights('decoder_weights.h5')
print("Autoencoder weights saved.")