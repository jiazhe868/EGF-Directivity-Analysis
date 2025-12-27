#!/bin/bash

# ==============================================================================
# Script Name: run_analysis.sh
# Description: 
#   Main driver script for Earthquake Directivity Analysis.
#   1. Stages waveform data from the 'example_data' directory.
#   2. Selects stations with valid S-wave picks (t1) for both events.
#   3. Aligns waveforms and prepares metadata.
#   4. Runs Python scripts for preprocessing and inversion.
#
# Usage: 
#   ./run_analysis.sh <EGF_Event_ID> <Target_Event_ID>
#   Example: ./run_analysis.sh 10530000 38450000
# ==============================================================================

# --- 1. Configuration & Input Checks ---

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <EGF_Event_ID> <Target_Event_ID>"
    exit 1
fi

REF=$1   # Small event ID (Empirical Green's Function)
TARG=$2  # Target event ID (Mainshock)

# Directories (Modify these paths as needed)
DATA_DIR="./example_data"
# Database files containing focal mechanism solutions
FM_DB_SMALL="./example_catalog/usedeventfm_egf.dat"
FM_DB_LARGE="./example_catalog/usedeventfm_targ.dat"
BIN_DIR="./bin" 
UTILS_DIR="./utils"
# Define Output Directories
OUT_ROOT="output"
OUT_FIGS="${OUT_ROOT}/figures"
OUT_DATA="${OUT_ROOT}/data"
OUT_LOGS="${OUT_ROOT}/logs"

# Ensure tools exist
command -v saclst >/dev/null 2>&1 || { echo >&2 "Error: 'saclst' is required."; exit 1; }

# Initialize Output Folders
mkdir -p "$OUT_FIGS"
mkdir -p "$OUT_DATA"
mkdir -p "$OUT_LOGS"

# --- 2. Data Staging (The Workspace) ---

echo "--> [Step 1/4] Staging data..."

# Clean workspace of previous temporary files
rm -rf temp
mkdir -p temp
rm -f *HHT *.procbest *.efs

if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory '${DATA_DIR}' not found."
    exit 1
fi

# Copy waveforms to current directory for processing
# (We copy them so we don't modify the originals in example_data during SAC header updates)
cp ${DATA_DIR}/${REF}*HHT . 2>/dev/null
cp ${DATA_DIR}/${TARG}*HHT . 2>/dev/null

count_files=$(ls *HHT 2>/dev/null | wc -l)
if [ "$count_files" -eq 0 ]; then
    echo "Error: No HHT files found. Please check '${DATA_DIR}'."
    exit 1
fi
echo "    Staged ${count_files} waveform files."

# --- 3. SAC Processing & Selection ---

echo "--> [Step 2/4] Processing SAC headers and selecting stations..."

# 3.1: Find stations with valid S-picks (t1 > 0)
saclst t1 f ${REF}.*HHT | gawk '{if ($2>0) print $1}' > list_ref_valid

# 3.2: Intersection with Target event
cat list_ref_valid | gawk -v tt="$TARG" 'BEGIN{FS="."} {print "saclst t1 f "tt"."$2"."$3"."$4}' | sh | gawk ' {if ($2>0) print $1}' > list_both_picked

# 3.3: Check secondary phases
cat list_ref_valid | gawk -v tt="$TARG" 'BEGIN{FS="."} {print "saclst t1 t2 f "tt"."$2"."$3"."$4}' | sh | gawk ' {if ($2<0) print $1,$3}' > list_syn_checks

# 3.4: Handle invalid ref picks
saclst t1 f ${REF}.*HHT | gawk '{if ($2<0) print $1}' > list_ref_invalid
cat list_ref_invalid | gawk -v tt="$TARG" 'BEGIN{FS="."} {print "saclst t2 f "tt"."$2"."$3"."$4}' | sh >> list_syn_checks

# 3.5: Align Time (Update t1 header)
# Note: We use 'temp/' as a scratchpad for the SAC macro operations
sed -i 's/'${TARG}'.//g' list_syn_checks
cat list_syn_checks | gawk '{print "saclst t2 f *."$1}' | sh | gawk '{print "r "$1;print "ch t1 "$2;print "wh"} END{print "q"}' | sac > /dev/null

sed -i 's/'${TARG}'.//g' list_both_picked
cat list_syn_checks | gawk '{print $1}' >> list_both_picked

# Move all HHTs to temp, then restore ONLY the selected/aligned ones
mv *HHT temp/
cat list_both_picked | gawk '{print "cp temp/*."$1" ."}' | sh

# 3.6: Create Station Metadata File
cat list_both_picked | gawk 'BEGIN{FS="."} {print $1,$2,$3}' > list_networks
cat list_both_picked | gawk -v rr="$REF" '{print "saclst stla stlo f "rr"."$1}' | sh > list_coords
paste list_networks list_coords | gawk '{print $1,$2,$3,"         "$5," "$6," 1  0  0"}' > best_stations

# Cleanup intermediate text lists
rm list_ref_valid list_both_picked list_networks list_coords list_syn_checks list_ref_invalid

# --- 4. Python Analysis ---

echo "--> [Step 3/4] Running Python analysis..."

# Prepare Source Parameters
grep ${REF} ${FM_DB_SMALL} | gawk '{print $6,$10,$17,$18,$19}' > e1params.dat
grep ${TARG} ${FM_DB_LARGE} | gawk '{print $6,$10,$17,$18,$19}' > e2params.dat

# Note: Ensure 'to_conjugate.awk' is in the UTILS_DIR
# If utils dir is not set up, user might need to adjust path or copy awk script here
if [ -f "${UTILS_DIR}/to_conjugate.awk" ]; then
    grep ${TARG} ${FM_DB_LARGE} | gawk -v awkscr="${UTILS_DIR}/to_conjugate.awk" '{print $17,$18,$19}' | gawk -v awkscr="${UTILS_DIR}/to_conjugate.awk" '{print "awk -f "awkscr" "$1,$2,$3}' | sh > e2mt_2.dat
else
    # Fallback if utils folder structure isn't strict
    echo "Warning: to_conjugate.awk not found in utils. Trying default path..."
    grep ${TARG} ${FM_DB_LARGE} | gawk '{print $17,$18,$19}' | gawk '{print "awk -f ~/bin/to_conjugate.awk "$1,$2,$3}' | sh > e2mt_2.dat
fi

# Run Preprocessing (get_p_waves.py)
# We use a loop to generate control files for both events
for EVENT in $REF $TARG; do
cat <<EOF > ctrl.getpsub.dat
sta= best_stations
ev= ${EVENT}
gcarcmin= 0
gcarcmax= 2
cf1= 5
cf2= 10
tb= -2
te= 10
dt= 0.02
EOF
    python ${BIN_DIR}/get_p_waves.py
done

# Run Inversion (invert_rupture.py)
cat <<EOF > ctrl.plotpairs.dat
e1= ${REF}
e2= ${TARG}
istrap= 0
EOF
python ${BIN_DIR}/invert_rupture.py

# --- 5. Organization & Cleanup ---

echo "--> [Step 4/4] Organizing results and cleaning up..."

# Move Figures
mv *.pdf "$OUT_FIGS"/ 2>/dev/null

# Move Data Results
mv results.dat results2.dat "$OUT_DATA"/ 2>/dev/null

# Move Intermediate Logs/Files (Optional: delete them if not needed)
mv *.procbest *.efs best_stations e1params.dat e2params.dat e2mt_2.dat ctrl.*.dat "$OUT_LOGS"/ 2>/dev/null

# Remove the Staged HHT files from Root (Keep root clean!)
# Only delete files starting with the Event IDs to be safe
rm ${REF}*HHT ${TARG}*HHT 2>/dev/null

# Remove temp folder
rm -rf temp

echo "=========================================================="
echo "Analysis Complete."
echo "----------------------------------------------------------"
echo "Figures saved to:    $OUT_FIGS"
echo "Data results in:     $OUT_DATA"
echo "Processing logs in:  $OUT_LOGS"
echo "=========================================================="

