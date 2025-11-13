import pickle
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess_subject(subject_id, data_path='WESAD', downsample_factor=10):
    """
    Loads, selects, downsamples, and filters data for a single subject.
    """
    file_path = os.path.join(data_path, subject_id, f'{subject_id}.pkl')
    
    if not os.path.exists(file_path):
        print(f"File not found for subject {subject_id}")
        return None, None

    with open(file_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    # 1. Get chest signals and labels
    chest_data = data['signal']['chest']
    ecg = chest_data['ECG']
    eda = chest_data['EDA']
    resp = chest_data['Resp']
    labels = data['label']

    # 2. Stack them
    stacked_signals = np.concatenate([
        ecg.reshape(-1, 1), 
        eda.reshape(-1, 1), 
        resp.reshape(-1, 1)
    ], axis=1)

    # 3. Downsample
    stacked_signals_down = stacked_signals[::downsample_factor]
    labels_down = labels[::downsample_factor]

    # 4. Filter for labels 1 (baseline) and 2 (stress)
    mask = np.isin(labels_down, [1, 2])
    
    filtered_signals = stacked_signals_down[mask]
    filtered_labels = labels_down[mask]

    # 5. Relabel: 1 -> 0, 2 -> 1
    filtered_labels[filtered_labels == 1] = 0  # Baseline
    filtered_labels[filtered_labels == 2] = 1  # Stress

    # --- 6. Normalize the data ---
    # We must fit the scaler to ALL data (baseline + stress) to use a consistent scale
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_signals = scaler.fit_transform(filtered_signals)
    
    print(f"--- Preprocessing complete for {subject_id} ---")
    print(f"Scaled signal shape: {scaled_signals.shape}")
    print(f"Label shape: {filtered_labels.shape}\n")
    
    return scaled_signals, filtered_labels.reshape(-1, 1), scaler

def create_sequences(data, labels, seq_len=70):
    """
    Creates overlapping sequences from the data.
    We will create separate lists for each label.
    """
    sequences = {0: [], 1: []} # Dict to hold baseline (0) and stress (1) sequences

    # We iterate up to the point where a full sequence can be extracted
    for i in range(len(data) - seq_len):
        # Check the label *at the end* of the window
        # This is a common way to label a sequence
        label = labels[i + seq_len - 1][0]
        
        # Check if the *entire* window has the same label
        # This prevents mixing baseline and stress in one window
        if np.all(labels[i:i + seq_len] == label):
            if label in sequences:
                sequences[label].append(data[i:i + seq_len])
                
    # Convert lists to NumPy arrays
    seq_baseline = np.array(sequences[0])
    seq_stress = np.array(sequences[1])
    
    return seq_baseline, seq_stress

# --- --- ---
# Now let's run the full pipeline
# --- --- ---

SEQ_LENGTH = 70 # 1 second of data (70 Hz)

# 1. Load, filter, and scale the data
scaled_data, filtered_labels, scaler = load_and_preprocess_subject('S2')

# 2. Create the windowed sequences
baseline_seq, stress_seq = create_sequences(scaled_data, filtered_labels, SEQ_LENGTH)

print(f"--- Sequence creation complete ---")
print(f"Sequence length (window size): {SEQ_LENGTH}\n")
print(f"Number of Baseline (0) sequences: {baseline_seq.shape[0]}")
print(f"Shape of Baseline data: {baseline_seq.shape}")
print(f"Shape (example): (Num_Windows, Seq_Len, Num_Features)\n")

print(f"Number of Stress (1) sequences: {stress_seq.shape[0]}")
print(f"Shape of Stress data: {stress_seq.shape}")

    # We'll also save the scaler. We need it later to 'un-scale' the fake data
    #
import joblib
joblib.dump(scaler, 'scaler.gz')
print("\nScaler saved to 'scaler.gz'")


# At the very end of main.py
print("Saving processed arrays...")
np.save('baseline_seq.npy', baseline_seq)
np.save('stress_seq.npy', stress_seq)
print("Arrays saved as baseline_seq.npy and stress_seq.npy")