# PowerShell script to set up dependencies for MRI Viewer App in Python 3.11.9
# Author: ChatGPT4o and Alexander Ling (alexander.l.ling@gmail.com; alling@bwh.harvard.edu)

# Function to activate the virtual environment
function Activate-Venv {
    if (Test-Path ".\py_sSegEnv\Scripts\Activate.ps1") {
        Write-Output "Activating virtual environment..."
        . .\py_sSegEnv\Scripts\Activate.ps1
    } else {
        Write-Error "Virtual environment activation script not found. Recreating the virtual environment..."
        Create-Venv
    }
}

# Function to create the virtual environment
function Create-Venv {
    Write-Output "Creating virtual environment 'py_sSegEnv'..."
    python -m venv py_sSegEnv
    if (Test-Path ".\py_sSegEnv\Scripts\Activate.ps1") {
        Write-Output "Virtual environment 'py_sSegEnv' created successfully."
    } else {
        Write-Error "Failed to create virtual environment. Exiting."
        exit 1
    }
}

# Check for Python 3.11.9
Write-Output "Checking Python version..."
$python_version = python --version 2>&1
if ($python_version -notlike "*3.11.9*") {
    Write-Error "Python 3.11.9 not found. Please ensure Python 3.11.9 is installed and added to PATH."
    exit 1
}

# Create the virtual environment if it doesn't exist
if (!(Test-Path "py_sSegEnv")) {
    Create-Venv
} else {
    Write-Output "Virtual environment 'py_sSegEnv' already exists. Skipping creation."
}

# Activate the virtual environment
Activate-Venv

# Upgrade pip using the virtual environment's python
Write-Output "Upgrading pip..."
.\py_sSegEnv\Scripts\python.exe -m pip install --upgrade pip

# Install dependencies inside the virtual environment
Write-Output "Installing required dependencies..."
.\py_sSegEnv\Scripts\python.exe -m pip install `
    nibabel==5.2.0 `
    numpy==1.26.4 `
    scipy==1.11.4 `
    pynrrd==1.0.0 `
    PyQt5==5.15.10 `
    pyvista==0.43.4 `
    pyvistaqt==0.7.0 `
    vtk==9.2.6 `
    matplotlib==3.7.4 `
    nptyping==2.5.0

# Verify installations
Write-Output "Verifying installations..."
try {
    .\py_sSegEnv\Scripts\python.exe -c "import nibabel as nib; print('Nibabel:', nib.__version__)"
    .\py_sSegEnv\Scripts\python.exe -c "import numpy; print('NumPy:', numpy.__version__)"
    .\py_sSegEnv\Scripts\python.exe -c "import pyvista; print('PyVista:', pyvista.__version__)"
    Write-Output "All dependencies installed and verified successfully."
} catch {
    Write-Error "Dependency verification failed. Check for errors above."
    exit 1
}

Write-Output "Setup complete. To run your app:"
Write-Output "`t1. Activate the virtual environment: .\py_sSegEnv\Scripts\Activate.ps1"
Write-Output "`t2. Run your script: python sSegEnv.py --load_volumes <path_to_volumes>"
