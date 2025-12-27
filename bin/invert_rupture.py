#!/usr/bin/env python
"""
Script: invert_rupture.py (formerly plotpairs.py)
Description:
    Performs the main EGF deconvolution and directivity analysis.
    
    Methodology:
    1. Reads processed SH-waves for the Reference (EGF) and Target (Mainshock) events.
    2. Computes the Source Time Function (STF) via Non-Negative Least Squares (NNLS).
       Target_Envelope = STF * EGF_Envelope
    3. Analyzes the azimuthal variation of the STF duration to estimate rupture properties
       (Length and Direction) using a Doppler effect model.
    4. Plots the waveform fits and the directivity fit.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.signal import hilbert, windows
from scipy.optimize import nnls, least_squares
from obspy.imaging.beachball import beach

# --- Utility Functions ---

def hann_smooth(a, nhann):
    """
    Applies Hanning window smoothing to a signal.
    Used to smooth the high-frequency envelopes.
    """
    hann = np.hanning(nhann)
    a2 = np.convolve(a, hann)
    n = len(a)
    nhalf = int(nhann/2)
    # Return the central part of the convolution to match original length
    asmo = a2[nhalf : nhalf+n]
    return asmo

def direc_func(x, d, l, a):
    """
    Theoretical directivity function (Unilateral Rupture).
    T(theta) = T0 - (L/v) * cos(theta - rupture_azimuth)
    
    Here approximated as: d - l/3 * cos((a - x))
    where 'l/3' implies a rupture velocity assumption (v_r ~ 3 km/s).
    """
    return d - l/3 * np.cos((a - x) / 180 * np.pi)

def res_func(params, x, y):
    """Residual function for Least Squares optimization (Free Azimuth)."""
    return direc_func(x, params[0], params[1], params[2]) - y

def res_func2(params, x, y, stk):
    """Residual function for Least Squares (Fixed Azimuth)."""
    return direc_func(x, params[0], params[1], stk) - y

# --- Main Analysis ---

def main():
    # --- 1. Parameter Initialization ---
    n0 = 1500       # Max points per trace
    nsta0 = 150     # Max number of stations
    hannlen = 1     # Smoothing window length (seconds)
    
    # Storage arrays
    a0 = np.zeros((n0, nsta0, 2))  # Raw data
    a  = np.zeros((n0, nsta0, 2))  # Filtered data
    aa = np.zeros((n0, nsta0))     # Processed envelope (Event 1)
    bb = np.zeros((n0, nsta0))     # Processed envelope (Event 2)
    
    # Metadata storage
    stid = np.empty((nsta0, 2), dtype=object) # Station IDs
    azi = np.zeros((nsta0, 2))                # Azimuths
    snr = np.zeros((nsta0, 2))                # SNRs
    nsta = np.zeros(2, dtype=int)             # Station counts
    qlon = np.zeros(2, dtype=float)           # Event Longitudes
    qlat = np.zeros(2, dtype=float)           # Event Latitudes

    # --- 2. Load Control Files ---
    
    # Load analysis config
    with open('ctrl.plotpairs.dat') as f:
        s = f.read().split()
        evnm1 = s[1]
        evnm2 = s[3]
        istrap = int(s[5])

    # Load Source Parameters (Event 1)
    with open('e1params.dat') as f:
        s = f.read().split()
        e1mw, e1dep, e1stk, e1dip, e1rak = s

    # Load Source Parameters (Event 2)
    with open('e2params.dat') as f:
        s = f.read().split()
        e2mw, e2dep, e2stk, e2dip, e2rak = s

    # Load Conjugate Plane Parameters (Event 2)
    with open('e2mt_2.dat') as f:
        s = f.read().split()
        e2stk2, e2dip2, e2rak2 = s

    # Calculate alternative strike angles (+/- 180 degrees) for ambiguity resolution
    e2stk_int = int(e2stk)
    e2stk2_int = int(e2stk2)
    
    e2stk3 = e2stk_int + 180 if e2stk_int < 180 else e2stk_int - 180
    e2stk4 = e2stk2_int + 180 if e2stk2_int < 180 else e2stk2_int - 180

    # Define input filenames
    efile1 = evnm1 + '.procbest'
    efile2 = evnm2 + '.procbest'
    
    # Clean up strings for plotting titles
    ef1b = evnm1.replace('EFS_', '').replace('.procbest', '')
    ef2b = evnm2.replace('EFS_', '').replace('.procbest', '')

    # --- 3. Read Waveform Data ---
    
    for iev, filename in enumerate([efile1, efile2]):
        with open(filename, 'r') as filein:
            # Read header line
            line = filein.readline()
            print(f"Reading {filename} header: {line.strip()}")
            
            # Parse header parameters
            # Assumes format: name delmin delmax freqcut stnmin tw1 tw2 lat lon
            parts = line.split()
            k_efs = 0 
            # Logic to find where parameters start after the filename
            for idx, part in enumerate(parts):
                if ".efs" in part:
                    k_efs = idx
                    break
            
            vals = [float(x) for x in parts[k_efs+1:]]
            delmin, delmax, freqcut, stnmin, tw1, tw2, qlat[iev], qlon[iev] = vals

            # Normalize longitude
            if qlon[iev] < 0:
                qlon[iev] += 360

            # Read Station Data
            ista = -1
            while True:
                line = filein.readline()
                if not line: break
                ista += 1
                
                # Parse station line
                # Format ends with: dt lat lon dist snr az mag
                # We split by looking for the '0.02' (dt) pattern or just standard split
                parts = line.split()
                # Finding the index of the float data
                # Assuming the last 7 columns are numbers
                data_part = parts[-7:]
                dt, slat, slon, delta, snr_val, azi_val, mag_val = [float(x) for x in data_part]
                
                # Station ID is everything before the numbers
                stid_str = " ".join(parts[:-7])
                stid[ista, iev] = stid_str
                
                snr[ista, iev] = snr_val
                azi[ista, iev] = azi_val
                
                if iev == 0: qmag1 = mag_val
                else: qmag2 = mag_val

                # Read Waveform Data points
                line = filein.readline()
                npts = int(line)
                
                for i in range(npts):
                    line = filein.readline()
                    val_raw, val_filt = [float(x) for x in line.split()]
                    a0[i, ista, iev] = val_raw
                    a[i, ista, iev] = val_filt
            
            nsta[iev] = ista + 1

    # Distance check between events
    dist_deg = np.sqrt(((qlon[0]-qlon[1])*np.cos(np.radians((qlat[0]+qlat[1])/2)))**2 + (qlat[0]-qlat[1])**2)
    distance_km = dist_deg * 111.2
    if distance_km > 200:
        print("Events too far apart (> 200km). Exiting.")
        quit()

    # --- 4. Preprocessing & Time Windowing ---
    
    t = np.linspace(tw1, tw2-dt, npts)
    nhann = int(hannlen / dt)
    
    # Calculate indices for slicing the SH-wave pulse
    i0 = int(-tw1 / dt)
    ibeg0 = i0 - int(0.5 / dt) # Start 0.5s before P
    
    # Alignment correction
    imax = 0 + int(0.5 / dt)
    imax = imax + ibeg0
    ialign = imax - i0
    talign = ialign * dt
    
    i0cor = imax
    ibeg = i0cor - int(0.5 / dt)
    imid = i0cor + int(2.5 / dt)
    imid2 = i0cor + int(3.0 / dt)
    iend = i0cor + int(5.0 / dt) # 5.0s window
    
    # Adjust end based on magnitude duration
    magdur = max(np.ceil(3.2**(qmag2-4.7)*(1+1/1.)*10)/10, 2.)
    iend2 = i0cor + int(magdur / dt)
    if iend2 > iend: iend2 = iend
    
    n1 = iend - ibeg + 1 # Length of data trace segment
    nt = 10              # Decimation/Smoothing step for plotting
    n2 = int((iend2 - ibeg) / nt) # Length of Source Time Function (STF)

    # --- 5. Initial Screening & Matching ---
    
    # Filter valid stations (SNR check & Envelope shape check)
    valid_indices = []
    tempid = list(stid[:, 0])
    # Remove None types
    tempid = [x for x in tempid if x is not None]
    nsta_common = len(tempid)
    
    for jj in range(nsta_common):
        # Calculate Envelopes (Hilbert Transform)
        # Event 1 (EGF)
        env1 = np.abs(hilbert(a[:, jj, 0]))**2
        env1 = hann_smooth(env1, nhann)
        env1 = env1 / max(env1) # Normalize
        
        # Event 2 (Mainshock)
        env2 = np.abs(hilbert(a[:, jj, 1]))**2
        env2 = hann_smooth(env2, nhann)
        env2 = env2 / max(env2) # Normalize

        # Apply Tukey Window to taper edges
        env1[ibeg:iend+1] *= windows.tukey(len(env1[ibeg:iend+1]), 0.3)
        env2[ibeg-nt:iend-nt+1] *= windows.tukey(len(env2[ibeg-nt:iend-nt+1]), 0.3)

        # Amplitude ratio check (Post-cursor / Main Pulse)
        # Ensures clean impulse-like SH-waves
        r1 = max(np.abs(env1[ibeg:imid])) / max(np.abs(env1[imid:iend]) + 1e-8)
        r2 = max(np.abs(env2[ibeg-nt:imid2-nt])) / max(np.abs(env2[imid2-nt:iend-nt+1]) + 1e-8)
        
        if snr[jj, 0] >= 4 and snr[jj, 1] >= 4 and r1 > 1 and r2 > 1:
            valid_indices.append(jj)

    # Sort valid stations by azimuth
    azi_subset = azi[valid_indices, 0]
    indices_sorted = [x for _, x in sorted(zip(azi_subset, valid_indices))]
    
    nnsta = len(indices_sorted)
    print(f'Number of useful stations (nnsta) = {nnsta}')

    # --- 6. Construct Matrix for Deconvolution (First Pass) ---
    # We solve Ax = d using NNLS, where A is constructed from EGF and d is the Mainshock.
    
    AA = np.zeros(((n1+n2)*nnsta + n2*nnsta, n2*nnsta))
    dd = np.zeros((n1+n2)*nnsta + n2*nnsta)
    
    amp1 = np.zeros(nnsta)
    amp2 = np.zeros(nnsta)
    
    for jj in range(nnsta):
        k1 = indices_sorted[jj]
        
        # Re-compute envelopes for the sorted list
        # EGF
        env1 = np.abs(hilbert(a[:, k1, 0]))**2
        env1 = hann_smooth(env1, nhann)
        amp1[jj] = np.sqrt(max(env1[ibeg:iend+1]))
        env1 = env1 / max(env1)
        
        # Mainshock
        env2 = np.abs(hilbert(a[:, k1, 1]))**2
        env2 = hann_smooth(env2, nhann)
        amp2[jj] = np.sqrt(max(env2[ibeg-nt:iend-nt+1]))
        env2 = env2 / max(env2)
        
        # Store for later
        aa[:, k1] = env1
        bb[:, k1] = env2 
        
        alpha = 2 * max(env1[ibeg:iend]) # Regularization weight scaling
        
        # Apply window
        aa[ibeg:iend+1, k1] *= windows.tukey(len(aa[ibeg:iend+1, k1]), 0.3)
        bb[ibeg-nt:iend-nt+1, k1] *= windows.tukey(len(bb[ibeg-nt:iend-nt+1, k1]), 0.3)
        
        # Build Convolution Matrix A for this station
        A_sub = np.zeros((n1+n2, n2))
        for i in range(n1):
            for kk in range(n2):
                j_idx = i - nt * kk
                if j_idx >= n1: continue
                if j_idx < 0: break
                A_sub[i, kk] = aa[j_idx + ibeg, k1]
        
        # Add Smoothness Constraints to A_sub
        # (Second derivative constraint)
        for i in range(n1, n1+n2-2):
            A_sub[i, i-n1] = alpha
            A_sub[i, i-n1+1] = -2*alpha
            A_sub[i, i-n1+2] = alpha
            
        A_sub[n1+n2-2, 0] = 5*alpha      # Anchor start
        A_sub[n1+n2-1, n2-1] = alpha     # Anchor end
        
        # Prepare data vector d
        cc = np.zeros(n2) # Constraint zeros
        d_sub = np.append(bb[ibeg-nt:iend-nt+1, k1], cc)
        
        # Insert into global matrices
        row_start = (n1+n2)*jj
        row_end = (n1+n2)*(jj+1)
        col_start = n2*jj
        col_end = n2*(jj+1)
        
        AA[row_start:row_end, col_start:col_end] = A_sub
        dd[row_start:row_end] = d_sub

    # Add Azimuthal Smoothing (Regularization across stations)
    # Penalizes differences between adjacent stations based on azimuth difference
    for ii in range(n2):
        for jj in range(nnsta - 1):
            k1 = indices_sorted[jj]
            k2 = indices_sorted[jj+1]
            
            diffaz = np.abs(azi[k1, 0] - azi[k2, 0])
            if diffaz > 180: diffaz = np.abs(360 - diffaz)
            
            beta = np.exp(-diffaz / 20)
            
            row_idx = (n1+n2)*nnsta + (ii-1)*nnsta + jj
            AA[row_idx, ii + n2*jj] = alpha * beta
            AA[row_idx, ii + n2*(jj+1)] = -alpha * beta
        
        # Wrap around (last station to first station)
        jj = nnsta - 1
        k1 = indices_sorted[jj]
        k0 = indices_sorted[0]
        diffaz = np.abs(azi[k1, 0] - azi[k0, 0])
        if diffaz > 180: diffaz = np.abs(360 - diffaz)
        beta = np.exp(-diffaz / 20)
        
        row_idx = (n1+n2)*nnsta + (ii-1)*nnsta + jj
        AA[row_idx, ii + n2*jj] = alpha * beta
        AA[row_idx, ii] = -alpha * beta

    # --- 7. Perform Inversion (First Pass) ---
    
    # Non-Negative Least Squares
    bb2, rnorm = nnls(AA, dd)
    bbs = np.dot(AA, bb2) # Predicted data
    bb2 = bb2 / dt        # Normalize by sampling rate

    # --- 8. Selection Refinement ---
    # Remove stations with poor fits (low correlation or high residual)
    
    indices_final = []
    coeff = np.zeros(nnsta)
    
    for jj in range(nnsta):
        k1 = indices_sorted[jj]
        
        # Predicted vs Observed
        pred = bbs[(n1+n2)*jj : (n1+n2)*jj + len(t[ibeg:iend])]
        obs = bb[ibeg-nt : iend-nt, k1]
        
        # Correlation
        rr = pearsonr(pred, obs)
        coeff[jj] = rr[0]
        
        # Residual error
        r2 = np.linalg.norm(pred - obs)**2 / (np.linalg.norm(pred)+1e-8) / (np.linalg.norm(obs)+1e-8)
        
        # STF duration check
        # Find effective duration of the retrieved STF
        tbb2 = bb2[n2*jj : n2*(jj+1)]
        maxtbb2 = max(tbb2)
        tempcutdur = len(tbb2) # default full length
        for kk in range(len(tbb2)):
            if tbb2[kk] >= 0.3 * maxtbb2:
                tempcutdur = kk + 1 # Update last significant point
        
        # Selection Criteria: Correlation >= 0.2, Error < 0.8
        if coeff[jj] >= 0.2 and r2 < 0.8 and tempcutdur < len(tbb2):
            indices_final.append(k1)

    nnsta2 = len(indices_final)
    print(f"Stations retained after refinement: {nnsta2}")

    # --- 9. Re-run Inversion with Selected Stations ---
    # (Repeating Matrix Construction step for the refined subset)
    
    AA = np.zeros(((n1+n2)*nnsta2 + n2*nnsta2, n2*nnsta2))
    dd = np.zeros((n1+n2)*nnsta2 + n2*nnsta2)
    
    # ... [Repeat AA construction logic for indices_final] ...
    # To save space, using simplified loop mirroring previous step
    for jj in range(nnsta2):
        k1 = indices_final[jj]
        
        # Re-apply windowing strongly
        aa[ibeg:iend+1, k1] *= windows.tukey(len(aa[ibeg:iend+1, k1]), 0.2)
        bb[ibeg-nt:iend-nt+1, k1] *= windows.tukey(len(bb[ibeg-nt:iend-nt+1, k1]), 0.2)
        
        alpha = 2 * max(aa[ibeg:iend, k1])
        
        A_sub = np.zeros((n1+n2, n2))
        for i in range(n1):
            for kk in range(n2):
                j_idx = i - nt * kk
                if j_idx >= n1: continue
                if j_idx < 0: break
                A_sub[i, kk] = aa[j_idx + ibeg, k1]
        
        # Regularization
        for i in range(n1, n1+n2-2):
            A_sub[i, i-n1] = alpha
            A_sub[i, i-n1+1] = -2*alpha
            A_sub[i, i-n1+2] = alpha
        A_sub[n1+n2-2, 0] = 5*alpha
        A_sub[n1+n2-1, n2-1] = alpha
        
        cc = np.zeros(n2)
        d_sub = np.append(bb[ibeg-nt:iend-nt+1, k1], cc)
        
        AA[(n1+n2)*jj:(n1+n2)*(jj+1), n2*jj:n2*(jj+1)] = A_sub
        dd[(n1+n2)*jj:(n1+n2)*(jj+1)] = d_sub

    # Re-apply Azimuthal Regularization
    for ii in range(n2):
        for jj in range(nnsta2 - 1):
            k1 = indices_final[jj]
            k2 = indices_final[jj+1]
            diffaz = np.abs(azi[k1, 0] - azi[k2, 0])
            if diffaz > 180: diffaz = np.abs(360 - diffaz)
            beta = np.exp(-diffaz / 20) * 1
            
            row = (n1+n2)*nnsta2 + (ii-1)*nnsta2 + jj
            AA[row, ii + n2*jj] = alpha * beta
            AA[row, ii + n2*(jj+1)] = -alpha * beta
            
        jj = nnsta2 - 1
        k1 = indices_final[jj]
        k0 = indices_final[0]
        diffaz = np.abs(azi[k1, 0] - azi[k0, 0])
        if diffaz > 180: diffaz = np.abs(360 - diffaz)
        beta = np.exp(-diffaz / 20) * 1
        
        row = (n1+n2)*nnsta2 + (ii-1)*nnsta2 + jj
        AA[row, ii + n2*jj] = alpha * beta
        AA[row, ii] = -alpha * beta

    # Final NNLS Inversion
    bb2, rnorm = nnls(AA, dd)
    bbs = np.dot(AA, bb2)
    bb2 = bb2 / dt

    # --- 10. Post-processing & Plotting ---
    
    plt.figure() # Start plotting
    
    cutdur = np.zeros(nnsta2)
    cutaz = np.zeros(nnsta2)
    ratios = np.zeros(nnsta2)
    mmsft = np.zeros(nnsta2)
    
    ratio_scale = np.sqrt(32**(qmag2 - qmag1)) if (qmag2 > qmag1) else 1.0

    for jj in range(nnsta2):
        k1 = indices_final[jj]
        yoff = azi[k1, 0]
        
        # Extract STF for this station
        tbb2 = bb2[n2*jj : n2*(jj+1)]
        maxtbb2 = max(tbb2)
        
        # Determine apparent duration (cutoff at 30% max amp)
        for kk in range(len(tbb2)):
            if tbb2[kk] >= 0.3 * maxtbb2:
                cutdur[jj] = kk * nt * dt
        
        cutaz[jj] = azi[k1, 0]
        
        # Calculate misfit
        pred = bbs[(n1+n2)*jj : (n1+n2)*jj + len(t[ibeg:iend])]
        obs = bb[ibeg-nt : iend-nt, k1]
        mmsft[jj] = np.linalg.norm(obs - pred)**2 / (np.linalg.norm(obs)+1e-8) / (np.linalg.norm(pred)+1e-8)
        
        # Amplitude Ratio
        ratios[jj] = (1 / ratio_scale) * (amp2[indices_sorted.index(k1)] / amp1[indices_sorted.index(k1)])
        
        # --- Plot Waveforms ---
        # Plot EGF (black)
        plt.plot(t[ibeg:iend] - talign, 10.*aa[ibeg:iend, k1] + yoff, '-k', linewidth=1)
        # Plot Mainshock (black)
        plt.plot(8. + t[ibeg:iend] - talign, 10.*bb[ibeg-nt:iend-nt, k1] + yoff, '-k', linewidth=1)
        # Plot Fit (red)
        plt.plot(8. + t[ibeg:iend] - talign, 10.*pred + yoff, '-r', linewidth=0.8)
        
        # Plot STF (red, shifted right)
        # 1. Define X-axis data first to know required length
        x_stf = 16. + t[ibeg:iend:nt] - talign
        required_len = len(x_stf)
        
        # 2. Pad tbb2 with zeros to match x-axis length
        # tbb2 only covers up to iend2 (magnitude duration), but plot covers up to iend
        if len(tbb2) < required_len:
            pad_width = required_len - len(tbb2)
            tbb2_plot = np.pad(tbb2, (0, pad_width), mode='constant')
        else:
            tbb2_plot = tbb2[:required_len]

        # 3. Plot with safe normalization
        max_val = max(tbb2_plot) if max(tbb2_plot) > 0 else 1.0
        plt.plot(x_stf, 10 * tbb2_plot / max_val + yoff, '-r', linewidth=1)

        # Annotations
        stid_str = stid[k1, 0].replace('BHT', '').replace('HHT', '')
        plt.text(22, yoff, f"{stid_str.split()[1]} {int(yoff)}", fontsize=6)
        plt.text(21, yoff, f"{ratios[jj]:.1f}", fontsize=5)
        plt.text(19.3, yoff, f"{mmsft[jj]:.2f}", fontsize=5)

    plt.xlim(-1., 24.)

    # --- 11. Directivity Fitting (Doppler) ---
    
    # 1. Fit with free azimuth
    ax0 = [1, 3, 180] # Initial guess: Duration, Length, Azimuth
    res_sl1 = least_squares(res_func, ax0, loss='soft_l1', f_scale=0.5, 
                            args=(cutaz, cutdur), 
                            bounds=([0.01, 0.01, -180], [5, 10, 540]))
    params = res_sl1.x
    
    # Outlier removal (3-sigma)
    pred_dur = direc_func(cutaz, params[0], params[1], params[2])
    residuals = np.abs(cutdur - pred_dur)
    mask = residuals <= 3 * np.std(residuals)
    
    cutaz = cutaz[mask]
    cutdur = cutdur[mask]
    
    # 2. Fit with FIXED azimuths (from Focal Mechanisms)
    # Test 4 possible rupture directions (from 2 nodal planes, +/- 180)
    possible_azimuths = [e2stk_int, e2stk2_int, e2stk3, e2stk4]
    results = []
    
    for az_test in possible_azimuths:
        # Only optimize Duration and Length
        res = least_squares(res_func2, [1, 3], loss='soft_l1', f_scale=0.5, 
                            args=(cutaz, cutdur, az_test), 
                            bounds=([0.01, 0.01], [5, 10]))
        results.append((res.cost, res.x, az_test))
        print(f"Cost: {res.cost:.4f}, Params: {res.x}, Azimuth: {az_test}")

    # Sort by lowest cost (best fit)
    results.sort(key=lambda x: x[0])
    
    best_res = results[0]
    second_best = results[1]
    
    optimal_cost, optimal_params, optimal_az = best_res
    
    # Check resolvability
    if (second_best[0] - optimal_cost) / optimal_cost > 0.25:
        optimal_flag = 1 # Clear winner
    else:
        optimal_flag = -1 # Ambiguous
        print("Result is ambiguous (bad).")

    # Output Results
    print(f"Optimal: Cost={optimal_cost}, Azimuth={optimal_az}")
    
    with open("results.dat", "w") as f:
        f.write(f"{evnm1} {evnm2} {optimal_params[0]:.1f} {optimal_params[1]:.1f} "
                f"{optimal_az:.0f} {optimal_cost:.3f} {second_best[0]:.3f} {optimal_flag:.0f}\n")

    # Save detailed data for plotting elsewhere if needed
    with open("results2.dat", "w") as f:
        for ii in range(len(cutaz)):
            pred_val = direc_func(cutaz[ii], optimal_params[0], optimal_params[1], optimal_az)
            f.write(f"{cutaz[ii]:.0f} {cutdur[ii]:.2f} {pred_val:.2f} {mmsft[ii]:.3f} {ratios[ii]:.2f}\n")

    # --- 12. Final Plot Overlay ---
    
    # Plot measured durations (stars)
    plt.plot(cutdur + 16. + t[ibeg] + 0.1, cutaz, '*k', linewidth=1)
    
    # Plot best fit curve (blue)
    xmin, xmax, ymin, ymax = plt.axis()
    tempaz = np.arange(int(ymin), int(ymax))
    
    fit_curve_best = direc_func(tempaz, optimal_params[0], optimal_params[1], optimal_az)
    plt.plot(fit_curve_best + 16. + t[ibeg] + 0.1, tempaz, '--b', linewidth=0.5)
    
    # Plot second best fit (yellow)
    fit_curve_2nd = direc_func(tempaz, optimal_params[0], optimal_params[1], second_best[2])
    plt.plot(fit_curve_2nd + 16. + t[ibeg] + 0.1, tempaz, '--y', linewidth=0.2)

    # Text Info
    plt.text(15.5, yoff+15, f"dur={optimal_params[0]:.1f}s len={optimal_params[1]:.1f}km dir={optimal_az} flag={optimal_flag}", fontsize=7)
    plt.text(-0.5, yoff+15, f"M{e1mw} {e1dep}km {e1stk}/{e1dip}/{e1rak}", fontsize=7)
    plt.text(7.5, yoff+15, f"M{e2mw} {e2dep}km {e2stk}/{e2dip}/{e2rak}", fontsize=7)

    # Beachballs
    ax = plt.gca()
    fm1 = [float(e1stk), float(e1dip), float(e1rak)]
    fm2 = [float(e2stk), float(e2dip), float(e2rak)]
    
    bb_width = 1.25 / (xmax - xmin) * (ymax - ymin)
    beach1 = beach(fm1, xy=(6.5, yoff+15), width=(1, bb_width), linewidth=0.3)
    beach2 = beach(fm2, xy=(14.5, yoff+15), width=(1, bb_width), linewidth=0.3)
    ax.add_collection(beach1) 
    ax.add_collection(beach2)

    # Formatting
    xtks = np.array([0, 2.5, 5, 8, 10.5, 13, 15.5, 18, 20.5])
    my_xticks = [0, 2.5, 5, 0, 2.5, 5, 0, 2.5, 5]
    plt.xticks(xtks, my_xticks)
    plt.xlabel('Time (s)')
    plt.ylabel('Azimuth ($^\circ$)')
    plt.title(f'SH_{ef1b}_{ef2b}')

    # Save Figure
    prefix = 'fig_' if optimal_flag > 0 else 'xfig_'
    filename_fig = f"{prefix}{ef1b}_{ef2b}.pdf"
    plt.savefig(filename_fig)
    print(f"Figure saved to {filename_fig}")

if __name__ == "__main__":
    main()
