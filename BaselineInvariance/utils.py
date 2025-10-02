# utils.py
import matplotlib.pyplot as plt

def plot_spectra(x, ys, labels=None, title="Spectra", n=6):
    plt.figure(figsize=(10,4))
    for i, y in enumerate(ys[:n]):
        lab = labels[i] if labels is not None else f"{i}"
        plt.plot(x, y, label=lab, alpha=0.8)
    plt.gca().invert_xaxis()
    plt.xlabel("Wavenumber (cm^-1)")
    plt.ylabel("Intensity (a.u.)")
    plt.title(title)
    if labels is not None: plt.legend()
    plt.show()
