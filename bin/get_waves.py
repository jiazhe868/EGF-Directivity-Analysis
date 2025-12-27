#!/usr/bin/env python
"""
Script: get_waves.py
Description: 
    Reads SAC waveform files, applies bandpass filtering, calculates 
    Signal-to-Noise Ratio (SNR), and exports the processed SH wave data 
    to an ASCII format for the inversion step.
"""

import numpy as np
import obspy
from obspy import read

def get_control_params(filename='ctrl.getpsub.dat'):
    """
    Parses the control file key-value pairs.
    Expected format: key= value key= value ...
    """
    params = {}
    with open(filename, 'r') as f:
        # Read the entire file as a single list of strings
        content = f.read().split()
        
        # Map specific indices to variables based on the known file structure
        # (This adheres to the original script's parsing logic)
        params['stfile'] = content[1]        # Station list file
        params['evnm'] = content[3]          # Event name
        params['delmin'] = float(content[5]) # Min distance
        params['delmax'] = float(content[7]) # Max distance
        params['freqcut1'] = float(content[9])  # Low freq corner
        params['freqcut2'] = float(content[11]) # High freq corner
        params['tw1'] = float(content[13])   # Time window start
        params['tw2'] = float(content[15])   # Time window end
        params['dt'] = float(content[17])    # Sampling rate
    return params

def main():
    # 1. Load control parameters
    p = get_control_params()

    # 2. Read station list
    # Format: Network Station Channel ...
    with open(p['stfile'], 'r') as f:
        station_lines = f.readlines()

    # 3. Output file setup
    outfile_name = p['evnm'] + '.procbest'
    fileout = open(outfile_name, 'w')
    
    # Header info for the output file
    efsfile = p['evnm'] + '.efs'
    
    # 4. Load Waveforms using Obspy
    # Wildcard loads all components matching the event ID
    try:
        stream = read(p['evnm'] + '.*.*.*HT')
    except Exception as e:
        print(f"Error reading SAC files: {e}")
        return

    ntrace = len(stream)
    print(f"ntrace = {ntrace}")

    if ntrace == 0:
        return

    # Write file header based on the first trace's event info
    # We assume all traces share the same event location
    stats0 = stream[0].stats
    qlat = stats0.sac.evla
    qlon = stats0.sac.evlo
    qdep = stats0.sac.evdp
    qmag = stats0.sac.mag
    
    stnmin = 0.0
    header_line = (f"{efsfile} {p['delmin']} {p['delmax']} {p['freqcut1']} "
                   f"{stnmin} {p['tw1']} {p['tw2']} {qlat} {qlon}\n")
    fileout.write(header_line)

    nout = 0

    # 5. Process each trace
    for i in range(ntrace):
        tr = stream[i]
        net = tr.stats.network.strip()
        stname = tr.stats.station.strip()
        
        # Check if station is in our selected list (produced by the bash script)
        skip = True
        for line in station_lines:
            parts = line.split()
            net2, stname2 = parts[0], parts[1]
            # Ensure net/station match and t1 pick is valid (>0)
            if (net == net2 and stname == stname2 and tr.stats.sac.t1 > 0):
                skip = False
                break
        
        if skip:
            continue

        # Prepare data
        # 'a' is raw data, 'af' will be filtered data
        a = tr.data.copy()
        
        # Apply Bandpass Filter
        tr.filter('bandpass', freqmin=p['freqcut1'], freqmax=p['freqcut2'])
        af = tr.data
        
        # Time relative to SH-pick
        # tpred is the SH-arrival time relative to the trace start
        tpred = tr.stats.sac.t1 - tr.stats.sac.b
        
        chan = tr.stats.channel
        slat = tr.stats.sac.stla
        slon = tr.stats.sac.stlo
        delta = tr.stats.sac.dist
        az = tr.stats.sac.az

        # --- SNR Calculation ---
        # Estimate duration based on magnitude
        magdur = max(np.ceil(3.2**(qmag-4.7)*(1+1/1.)*10)/10, 2.)
        
        # Define windows for Noise (before P) and Signal (after P)
        j1 = int(round((tpred - 2.) / p['dt']))
        j2 = int(round((tpred - 0.) / p['dt']))
        j3 = int(round((tpred + magdur + 1) / p['dt']))
        
        # Handle boundary conditions
        if j1 < 0: j1 = 0
        if j2 > len(af): j2 = len(af)
        if j3 > len(af): j3 = len(af)

        # Calculate Mean Square amplitudes
        anoise = 0
        asignal = 0
        if (j2 > j1):
            anoise = sum(np.square(af[j1:j2])) / len(af[j1:j2])
        if (j3 > j2):
            asignal = sum(np.square(af[j2:j3])) / len(af[j2:j3])
            
        stn = 0.
        if anoise != 0.:
            stn = asignal / anoise

        # Write station metadata to output
        # Format: Net Sta Chan dt Lat Lon Dist SNR Az Mag
        meta_buf = (f"{net} {stname} {chan} "
                    f"{p['dt']:6.3f} {slat:10.3f} {slon:10.3f} {delta:8.1f} "
                    f"{round(stn,2):8.2f} {int(az)} {qmag:8.1f}\n")
        fileout.write(meta_buf)

        # Output waveform data
        # Window: [tpred + tw1, tpred + tw2]
        w1 = int(round((tpred + p['tw1']) / p['dt']))
        w2 = int(round((tpred + p['tw2']) / p['dt']))
        n_pts = w2 - w1
        
        fileout.write(f"{n_pts}\n")
        
        for j in range(w1, w2):
            val_raw = a[j] if 0 <= j < len(a) else 0.0
            val_filt = af[j] if 0 <= j < len(af) else 0.0
            fileout.write(f"{val_raw:12.4e} {val_filt:12.4e} \n")

        nout += 1

    fileout.close()
    print(f"#traces written = {nout}")

if __name__ == "__main__":
    main()
