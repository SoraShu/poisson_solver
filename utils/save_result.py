import h5py

def save_results(f, filename="result.h5"):
    with h5py.File(filename, "w") as hf:
        hf.create_dataset("grid", data=f)