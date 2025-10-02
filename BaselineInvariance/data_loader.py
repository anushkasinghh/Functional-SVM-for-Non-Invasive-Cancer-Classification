import pandas as pd
import numpy as np
from pathlib import Path

def load_healthy(folder=Path("/home/anushkasingh/Desktop/Thesis/Code/ALLDataGross/healthyCohort")):

    folder = Path(folder)
    spectra, filenames = [], []
    wn_ref = None
    # print("step 1 done!")
    # Provided metadata (match order to files!)
    normVP = np.array([
        420, 420, 428, 448, 417, 430, 420, 449, 483, 499,
        438, 465, 438, 428, 503, 505, 504, 454, 515, 441,
        404, 363
    ])
    infoP = np.array([
        "F", "M", "M", "F", "F", "F", "F", "M", "M", "M",
        "M", "F", "M", "M", "F", "M", "M", "M", "M", "M",
        "M", "M"
    ])

    # print("norm vp, gender info loaded!")
    i = 0
    # print("Looking for files in:", folder.resolve())
    # print("Found files:", list(folder.glob("*")))
    
    for idx, file in enumerate(sorted(folder.glob("*.dpt"))):
        i+=1
        # print(f"processing file {i}")
        try:
            # whitespace flexible (tabs or spaces)
            df = pd.read_csv(file, sep=r"\s+", engine="python")
            if df.shape[1] != 2:  # fallback: comma separated
                df = pd.read_csv(file, sep=",", engine="python")

            df.columns = ["Wavenumber", "Intensity"]
            wn, intensity = df["Wavenumber"].values, df["Intensity"].values

            # Reference wavenumber axis
            if wn_ref is None:
                wn_ref = wn
            elif not np.allclose(wn_ref, wn, rtol=1e-3, atol=1e-3):
                print(f"⚠️ Wavenumber mismatch in {file.name}")

            spectra.append(intensity)
            filenames.append(file.stem)

            print(f"✅ Loaded {file.name} (VP={normVP[idx]}, Sex={infoP[idx]})")

        except Exception as e:
            print(f"❌ Error loading {file.name}: {e}")
    print("exited loop!")
    return np.array(spectra), wn_ref, filenames, normVP, infoP
# print(load_healthy())