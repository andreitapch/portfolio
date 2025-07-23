import os

def get_available_model_timestamps(model_dir):
    files = os.listdir(model_dir)
    timestamps = sorted(set(
        f.replace("autoencoder_", "").replace(".pt", "")
        for f in files
        if f.startswith("autoencoder_") and f.endswith(".pt")
    ))
    return timestamps
