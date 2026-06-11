import h5py

with h5py.File("models/morph_model.h5", "r") as f:
    print("Keys:", list(f.keys()))

    if "model_config" in f.attrs:
        print("\nMODEL CONFIG FOUND")
        print(f.attrs["model_config"][:500])