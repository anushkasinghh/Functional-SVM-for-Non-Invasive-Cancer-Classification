import numpy as np

def baseline_roy(x, y, norm_factor_i, ref_region=(2550, 2600), 
                                                #  L=[515,  990, 1035, 1115, 1155, 1175, 1188, 2155], #3rd order correction for each SR used in SVM model.                                                                                              
                                                #  H=[545, 1020, 1065, 1145, 1185, 1205, 1218, 2185]): 
                   L=[990, 2140, 1150], H=[1020, 2205, 1400]): # For other samples                    
    """
    Mimics the exact logic of the MATLAB 'Second order baseline correction' section.
    
    x: Wavenumbers (must be sorted)
    y: spectrum (Absorbance)
    norm_factor_i: The specific NormF value for this spectrum.
    
    The correct pipeline is:
    For each sample:
        1st order BC  →  full spectrum
        2nd order BC  →  full spectrum
        3rd order BC  →  applied segment by segment, written into full array

    After all samples are processed:(FOR PLOTTING PURPOSE ONLY) 
        For each segment window:
            average across samples  →  mean spectrum per window
            smooth                  →  smoothed mean per window
    Matlab deals with a small spectral window of 30 cm^-1. hence step 2 has been skipped. and averaging and smoothening is applied on y3 results - useful for plotting. 
    """
    
    # Ensure numpy arrays & sorted wavenumbers
    x = np.array(x, float)
    y = np.array(y, float)
    if x[0] > x[-1]:
        x, y = x[::-1], y[::-1]
    # 0. Normalize
    scale_factor = 500 / norm_factor_i
    y = y * scale_factor

    # 1. First-order shift using ref_region: uniform shift over FULL spectrum ───────────────
    lo, hi = ref_region
    mask_ref = (x >= lo) & (x <= hi)
    LS = np.mean(y[mask_ref]) if mask_ref.any() else 0.0
    y1 = y - LS

    # 2. Second-order: global tilt over FULL spectrum ─────────────────
    N = len(y1)
    slope = (y1[-1] - y1[0]) / (N)
    trend = slope * np.arange(N)[::-1] # [0,1,2,3,4,5..][:-1]
    
    y2 = y1 - trend

   # ── 3rd order: local tilt per segment ─────────────────────────
    y3 = y2.copy()              # initialized ONCE before the loop


    for i in range(len(L)):
        lo = L[i]
        hi = H[i]
        mask = (x >= lo) & (x <= hi)
        if not mask.any():
            print("Warning: Segment not found.")
            continue
# BASELINE CORRECT PAPER IMPLEMENTATION: apply 2nd order correction to all SRs used in the downstream SVM pipelines. 

        # x_seg = x[mask]
        # y_seg_in = y2[mask]
        # N_seg = len(x_seg)
        # slope = (y_seg_in[-1] - y_seg_in[0]) / N_seg 
        # trend = slope * np.arange(N_seg)[::-1]
        # y3[mask] = y_seg_in - trend
        

# SUSMITA'S R CODE IMPLEMENTATION:
    
        x_seg = x[mask]
        y_seg_in = y2[mask]
        
        N_seg = len(x_seg)

        y_start = y_seg_in[0]    # Value at low end (limL)
        y_end = y_seg_in[-1]   # Value at high end (limH)
        
        slope = (y_start - y_end) / N_seg
        
        bcf_vector = y_end + np.arange(N_seg) * slope
        y_seg_add = y_seg_in + bcf_vector

        shift_amount = y_seg_add[0]
        y_seg_final = y_seg_add - shift_amount
        
        # scale_factor = 500 / norm_factor_i
        # y_seg_final = y_seg_shift * scale_factor
    
        y3[mask] = y_seg_final        
        


    return y3, y2, y1

def moving_average(x, window=5):
    x = np.asarray(x)
    n = len(x)
    y = np.zeros(n)

    half = window // 2

    for i in range(n):
        # Dynamic window like MATLAB
        start = max(0, i - half)
        end   = min(n, i + half + 1)
        y[i] = np.mean(x[start:end])

    return y




def process_all_samples(x, all_spectra, norm_factors,
                        segments=[(990, 1020), (2140, 2205), (1150, 1400)],
                        ):
    """
    Full pipeline over ALL samples.

    Parameters
    ----------
    x            : wavenumber array,  shape (Mf,)
    all_spectra  : raw absorbance matrix, shape (Mf, Nf)  — one column per sample 
    norm_factors : array of length Nf, one normalization constant per sample 

    Returns
    -------

    - dataS — shape (Mf, 74): the fully baseline-corrected spectra matrix. Each column is one sample's corrected intensity array (after all 3 orders of baseline correction).                            
    - av_data — shape (Mf,): the mean spectrum across all 74 samples (column-wise mean of dataS).                                                                                                                  
    - av_smoothed — USE FOR PLOTTING , etc. shape (Mf,): same as av_data but with a 5-point moving average applied within each segment window [(990,1020), (2140,2205), (1150,1400)]. Outside those windows it's identical to av_data. 
    """
    Mf, Nf = all_spectra.shape
    dataS = np.zeros((Mf, Nf))

    # ── Step 1: correct each sample individually ──────────────────
    for i in range(Nf): 
        y_corrected, _, _ = baseline_roy(x,all_spectra[:, i],
            norm_factors[i])

        dataS[:, i] = y_corrected

    # ── Step 2: average across samples (per segment only) ─────────
    # Mean of corrected spectra across all Nf samples
    av_data = np.mean(dataS, axis=1)      # shape (Mf,)

    # ── Step 3: smooth the average (per segment only) ─────────────
    # Smoothing is applied only within each segment window,
    # exactly as MATLAB smooth(avData) acts on the windowed data
    av_smoothed = av_data.copy()
    for (limL, limH) in segments:
        mask = (x >= limL) & (x <= limH)
        if mask.sum() >= 5:
            av_smoothed[mask] = moving_average(av_data[mask], window=5)

    return dataS, av_data, av_smoothed


# def baseline_intermediates(x, y, norm_factor_i, ref_region=(2550, 2600), limL=990, limH=1020):
#     """
#     Mimics the exact logic of the MATLAB 'Second order baseline correction' section.
    
#     x: Wavenumbers (must be sorted)
#     y: Pre-corrected spectrum (Absorbance)
#     norm_factor_i: The specific NormF value for this spectrum.
#     """
    
#     # Ensure numpy arrays & sorted wavenumbers
#     x = np.array(x, float)
#     y = np.array(y, float)
#     if x[0] > x[-1]:
#         x, y = x[::-1], y[::-1]

#     scale_factor = 500 / norm_factor_i
#     y = y * scale_factor

#     window = 5
#     kernel = np.ones(window) / window
#     # y = np.convolve(y, kernel, mode='same')


#     # 1. First-order shift using ref_region
#     lo, hi = ref_region
#     mask_ref = (x >= lo) & (x <= hi)
#     if not mask_ref.any():
#         shift = 0.0
#     else:
#         shift = np.mean(y[mask_ref])
#     y1 = y - shift

#     # 2. Second-order: remove global linear tilt
#     N = len(y1)
#     slope = (y1[-1] - y1[0]) / (N)
#     trend = slope * np.arange(N)[::-1] # [0,1,2,3,4,5..][:-1]
    
#     y2 = y1 - trend

#     return y2, y1
