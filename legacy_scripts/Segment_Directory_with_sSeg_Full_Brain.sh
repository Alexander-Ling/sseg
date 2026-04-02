#!/bin/bash
#Script to loop sSeg.py through a directory while segmenting ventricles and resection cavities
#Alex Ling
#12/2/2024
#E. Antonio Chiocca Group
#Brigham and Women's Hospital, Boston, MA, USA
#Generated with assistance of ChatGPT4

# Check if an argument was provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <base_directory>"
    exit 1
fi

# Base directory containing all patient data
base_directory="$1"

# Find all timepoint directories
source env/bin/activate
find "$base_directory" -mindepth 0 -maxdepth 2 -type d | while read timepoint_dir; do
    echo "$timepoint_dir"
    # Check if both required nrrd files exist in the directory
    if [[ ! -f $(ls "${timepoint_dir}"/*_seg_UNet_manual.nii.gz 2>/dev/null) ]]; then
        # Check for the presence of both MRI files before processing
        t1c_file=$(ls "${timepoint_dir}"/*_T1c_normalized.nii.gz 2>/dev/null)
		t1n_file=$(ls "${timepoint_dir}"/*_T1n_normalized.nii.gz 2>/dev/null)
        t2f_file=$(ls "${timepoint_dir}"/*_T2f_normalized.nii.gz 2>/dev/null)
		t2w_file=$(ls "${timepoint_dir}"/*_T2w_normalized.nii.gz 2>/dev/null)
		seg_file=$(ls "${timepoint_dir}"/*_seg_UNet.nii.gz 2>/dev/null)
		manual_seg_file=$(ls "${timepoint_dir}"/*_seg_UNet_manual.nii.gz 2>/dev/null)
		brainFeatures_seg_file=$(ls "${timepoint_dir}"/*_brainFeatures_seg.nii.gz 2>/dev/null)
		brainMask_file=$(ls "${timepoint_dir}"/*_brainmask.nii.gz 2>/dev/null)
		
		if [[ -n "$brainFeatures_seg_file" ]]; then
            #echo "BrainFeatures segmentation file exists in $timepoint_dir, skipping..."
            #continue
			segmentation_file="$brainFeatures_seg_file"
		elif [[ -f "$manual_seg_file" ]]; then
			segmentation_file="$manual_seg_file"
		else
			segmentation_file="$seg_file"
		fi

        if [[ -z "$t1c_file" || -z "$t1n_file" || -z "$t2f_file" || -z "$t2w_file" || -z "$segmentation_file" || -z "brainMask_file" ]]; then
            echo "Required MRI files missing in $timepoint_dir, skipping..."
        else
            # All MRI files are present, execute the command
            echo "Processing: $timepoint_dir"
            python3 /mnt/d/Automated_MRI_Segmentation/sSegEnv.py \
                --load_volumes "$t1c_file" "$t1n_file" "$t2f_file" "$t2w_file"\
				--load_segmentations "$segmentation_file"\
				--ROI_mask "$brainMask_file"
        fi
    else
        echo "Both nrrd files exist in $timepoint_dir, skipping..."
    fi
done