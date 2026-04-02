# PowerShell script to loop through directories and process MRI data
# Alexander Ling (alexander.l.ling@gmail.com; alling@bwh.harvard.edu) | ChatGPT4o
# Date: 12/11/2024
# E. Antonio Chiocca Group, Brigham and Women's Hospital, Boston, MA, USA

# Check if the base directory argument is provided
if ($args.Count -eq 0) {
    Write-Output "Usage: .\Segment_Directory_with_sSeg_Full_Brain.ps1 <base_directory>"
    exit 1
}

# Set the base directory from the first argument
$base_directory = $args[0]

# Activate the Python virtual environment
$venv_activate = ".\py_sSegEnv\Scripts\Activate.ps1"
if (Test-Path $venv_activate) {
    . $venv_activate
} else {
    Write-Error "Virtual environment activation script not found. Exiting."
    exit 1
}

# Recursively find subdirectories up to 2 levels deep
Get-ChildItem -Path $base_directory -Recurse -Depth 2 -Directory | ForEach-Object {
    $timepoint_dir = $_.FullName
    Write-Output "Processing directory: $timepoint_dir"

    # Check if *_seg_UNet_manual.nii.gz already exists
    $manual_seg_exists = Get-ChildItem -Path "$timepoint_dir" -Filter "*_seg_UNet_manual.nii.gz" -ErrorAction SilentlyContinue
    if ($manual_seg_exists) {
        Write-Output "Both manual segmentation files exist in $timepoint_dir, skipping..."
        return
    }

    # Find the required MRI files
    $t1c_file = Get-ChildItem -Path "$timepoint_dir" -Filter "*_T1c_normalized.nii.gz" -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName
    $t1n_file = Get-ChildItem -Path "$timepoint_dir" -Filter "*_T1n_normalized.nii.gz" -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName
    $t2f_file = Get-ChildItem -Path "$timepoint_dir" -Filter "*_T2f_normalized.nii.gz" -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName
    $t2w_file = Get-ChildItem -Path "$timepoint_dir" -Filter "*_T2w_normalized.nii.gz" -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName
    $seg_file = Get-ChildItem -Path "$timepoint_dir" -Filter "*_seg_UNet.nii.gz" -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName
    $manual_seg_file = Get-ChildItem -Path "$timepoint_dir" -Filter "*_seg_UNet_manual.nii.gz" -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName
    $brainFeatures_seg_file = Get-ChildItem -Path "$timepoint_dir" -Filter "*_brainFeatures_seg.nii.gz" -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName
    $brainMask_file = Get-ChildItem -Path "$timepoint_dir" -Filter "*_brainmask.nii.gz" -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName

    # Determine which segmentation file to use
    if ($brainFeatures_seg_file) {
        #$segmentation_file = $brainFeatures_seg_file
		Write-Output "BrainFeatures segmentation file exists in $timepoint_dir, skipping..."
		return
    } elseif ($manual_seg_file) {
        $segmentation_file = $manual_seg_file
    } else {
        $segmentation_file = $seg_file
    }

    # Check if all required files are present
    if (-not $t1c_file -or -not $t1n_file -or -not $t2f_file -or -not $t2w_file -or -not $segmentation_file -or -not $brainMask_file) {
        Write-Output "Required MRI files missing in $timepoint_dir, skipping..."
    } else {
        # Run the segmentation Python script
        Write-Output "Processing: $timepoint_dir"
        python sSegEnv.py `
            --load_volumes "$t1c_file" "$t1n_file" "$t2f_file" "$t2w_file" `
            --load_segmentations "$segmentation_file" `
            --ROI_mask "$brainMask_file"
    }
}
