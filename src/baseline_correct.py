import numpy as np


# def baseline_roy(x, y, norm_factor_i,
#                  ref_region=(2550, 2600), 
#                  segment_edges = [990, 1020]):
#     """
#     Hierarchical baseline correction per Roy & Maiti (2024).
#     1. Linear shift based on region.
#     2. Remove global linear tilt.
#     3. Remove local tilt per segment.
#     """

#     # Ensure numpy arrays & sorted wavenumbers
#     x = np.array(x, float)
#     y = np.array(y, float)
#     if x[0] > x[-1]:
#         x, y = x[::-1], y[::-1]

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

#     # # 3. Third-order: segment-wise correction
    
#     # y3 = y2.copy()
#     # for i in range(len(segment_edges)-1):
#     #     loe, hie = segment_edges[i], segment_edges[i+1]
#     #     print(loe, hie)
#     #     mask = (x >= loe) & (x < hie)
#     #     if mask.sum() < 2:
#     #         continue
#     #     xi = x[mask]; yi = y2[mask]
#     #     # linear detrend within segment
#     #     m, b = np.polyfit(xi, yi, 1)
#     #     y3[mask] = yi - (m * xi + b)

#     # return y3, y2, y1  # return all stages for QC
    
#     # 3. Third-order: apply local rotation ONLY to user-defined segment
#     y3 = y2.copy()

#     # User-defined segment boundaries
#     x1 = segment_edges[0]  
#     x2 = segment_edges[1]

#     # Apply local tilt correction only within given segment
#     mask = (x >= x1) & (x < x2)

#     if mask.sum() >= 2:
#         # x_segment = x[mask]
#         y_segment = y2[mask]
        
#         # Use the same slope finding logic as step 2, but locally
#         N_segment = len(y_segment)
#         slope_local = (y_segment[-1] - y_segment[0]) / N_segment
#         trend_local = slope_local * np.arange(N_segment)[::-1]
        
#         y3[mask] = y_segment - trend_local
    
#     scale_factor = 500 / norm_factor_i
#     y3 = y3 * scale_factor

#     # All points outside the user segment remain unchanged from y2

#     # return y3, y2, y1
#     return y3


def baseline_roy(x, y, norm_factor_i, ref_region=(2550, 2600), L=[990, 2140, 1150], H=[1020, 2205, 1400]):
    """
    Mimics the exact logic of the MATLAB 'Second order baseline correction' section.
    
    x: Wavenumbers (must be sorted)
    y: spectrum (Absorbance)
    norm_factor_i: The specific NormF value for this spectrum.
    """
    
    # Ensure numpy arrays & sorted wavenumbers
    x = np.array(x, float)
    y = np.array(y, float)
    if x[0] > x[-1]:
        x, y = x[::-1], y[::-1]

    # 1. First-order shift using ref_region
    lo, hi = ref_region
    mask_ref = (x >= lo) & (x <= hi)
    if not mask_ref.any():
        shift = 0.0
    else:
        shift = np.mean(y[mask_ref])   # SHIFT TO ZERO. 
    y1 = y - shift

    # 2. Second-order: r emove global linear tilt
    N = len(y1)
    slope = (y1[-1] - y1[0]) / (N)
    trend = slope * np.arange(N)[::-1] # [0,1,2,3,4,5..][:-1]
    
    y2 = y1 - trend

    for i in range(len(L)):
        limL = L[i]
        limH = H[i]
        # 1. Segment Selection (Equivalent to MATLAB creating dataL)
        # The MATLAB code operates ONLY on the segment defined by limL and limH.
        mask = (x >= limL) & (x <= limH)
        if not mask.any():
            print("Warning: Segment not found.")
            return np.zeros_like(y1)

        x_seg = x[mask]
        y_seg_in = y1[mask]
        
        N_seg = len(x_seg)

        # 2. Linear Baseline Vector Generation (bcf calculation)
        # The MATLAB code uses the segment endpoints (y_seg_in[0] at limL, y_seg_in[-1] at limH)
        # to define a linear ramp (bcf).
        
        y_start = y_seg_in[0]    # Value at low end (limL)
        y_end = y_seg_in[-1]   # Value at high end (limH)
        
        # MATLAB: bcf = dataL(end,i+1):(dataL(1,i+1)-dataL(end,i+1))/length(dataL):dataL(1,i+1);
        # This generates a ramp from y_end to y_start in N_seg+1 points, then discards the last point.
        # The slope calculation is (y_start - y_end) / N_seg
        slope = (y_start - y_end) / N_seg
        
        # We generate the linear ramp B, starting at y_end and increasing by slope
        # B[k] = y_end + k * slope for k=0 to N_seg-1
        # Note: MATLAB's index manipulation is tricky; this is the most direct translation:
        bcf_vector = y_end + np.arange(N_seg) * slope

        # 3. Apply Correction 1: Linear Addition
        # MATLAB: data = dataL(:,i+1) + bcf'; (ADDITION)
        y_seg_add = y_seg_in + bcf_vector

        # 4. Apply Correction 2: Shift First Point to Zero
        # MATLAB: data = data - data(1);
        shift_amount = y_seg_add[0]
        y_seg_shift = y_seg_add - shift_amount
        
        # 5. Apply Correction 3: Normalization/Scaling
        # MATLAB: data = data*(500/normF(i));
        scale_factor = 500 / norm_factor_i
        y_seg_final = y_seg_shift * scale_factor
        
        # 6. Reconstruct the full spectrum (y3)
        # The full spectrum output (y3) should contain the processed segment 
        # and the original values outside the segment.
        y3 = y1.copy()
        y3[mask] = y_seg_final

        
        # Smoothening 
        # MATLAB default is a span of 5
        window_size = 5 
        kernel = np.ones(window_size) / window_size

        # mode='same' ensures the output is the same size as the input
        y3 = np.convolve(y3, kernel, mode='same')
        
        # We only return the final corrected spectrum, as the MATLAB code saves it.
        return y3


def baseline_intermediates(x, y, norm_factor_i, ref_region=(2550, 2600), limL=990, limH=1020):
    """
    Mimics the exact logic of the MATLAB 'Second order baseline correction' section.
    
    x: Wavenumbers (must be sorted)
    y: Pre-corrected spectrum (Absorbance)
    norm_factor_i: The specific NormF value for this spectrum.
    """
    
    # Ensure numpy arrays & sorted wavenumbers
    x = np.array(x, float)
    y = np.array(y, float)
    if x[0] > x[-1]:
        x, y = x[::-1], y[::-1]

    scale_factor = 500 / norm_factor_i
    y = y * scale_factor

    window = 5
    kernel = np.ones(window) / window
    # y = np.convolve(y, kernel, mode='same')


    # 1. First-order shift using ref_region
    lo, hi = ref_region
    mask_ref = (x >= lo) & (x <= hi)
    if not mask_ref.any():
        shift = 0.0
    else:
        shift = np.mean(y[mask_ref])
    y1 = y - shift

    # 2. Second-order: remove global linear tilt
    N = len(y1)
    slope = (y1[-1] - y1[0]) / (N)
    trend = slope * np.arange(N)[::-1] # [0,1,2,3,4,5..][:-1]
    
    y2 = y1 - trend

    return y2, y1
