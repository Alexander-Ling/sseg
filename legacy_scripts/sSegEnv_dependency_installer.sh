#!/bin/bash

# Automated script to set up MRIViewer dependencies
# Author: Assistant for Alex Ling

# Check if virtual environment folder exists
if [ ! -d "env" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv env
fi

# Activate the virtual environment
source env/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install required dependencies
echo "Installing dependencies..."
pip install \
    numpy==1.26.4 \
    scipy==1.11.4 \
    nibabel==5.2.0 \
    pynrrd==1.0.0 \
    PyQt5==5.15.10 \
    pyvista==0.43.4 \
    pyvistaqt==0.7.0 \
    vtk==9.2.6 \
    matplotlib==3.7.4 \
    nptyping==2.5.0

# Verify installations
echo "Verifying installations..."
python -c "import numpy; print('NumPy:', numpy.__version__)"
python -c "import scipy; print('SciPy:', scipy.__version__)"
python -c "import nibabel; print('Nibabel:', nibabel.__version__)"
python -c "import nrrd; print('PyNRRD: OK')"
python -c "import pyvista; print('PyVista:', pyvista.__version__)"
python -c "import pyvistaqt; print('PyVistaQt:', pyvistaqt.__version__)"
python -c "import vtk; print('VTK:', vtk.VTK_VERSION)"
python -c "import matplotlib; print('Matplotlib:', matplotlib.__version__)"
python -c "import PyQt5; print('PyQt5: OK')"

echo "Setup complete. Run your script using: source env/bin/activate && python3 sSegEnv.py"
