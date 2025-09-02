# MC_project
Simulation of an electron beam passing through deflection plates and an electrostatic lens, computing final positions on a screen with optional phosphor diffusion. Supports multiple models, statistical analysis, and visualization of beam distributions. The goal of the project is to undertand wich are the relevant sources of errors and uncertainties in voltage measurements.

# Electron Beam Simulation

This repository contains a Python simulator for an electron beam passing through deflection plates and an electrostatic lens, computing final impact positions on a screen. The simulator supports different physical models, optional phosphor diffusion, and generates statistical analysis and visualizations.

## Features

- Simulate electron trajectories using:
  - **Convolution model**: accounts for Gaussian fluctuations of the electric field.
  - **First-order model**: integrates motion equations numerically for local fields.
- Include **electrostatic lens effects**.
- Optional **phosphor diffusion** on the screen.
- Statistical outputs: mean, standard deviation, skewness.
- Visualization: X/Y histograms and 2D scatter plots of impact points.
- Fully configurable via **command-line arguments**.

## Requirements

- Python => 3.8
- Required packages:
  '''bash
  pip install numpy matplotlib scipy argparse
  '''

# Usage
Run the simulation from the command line:

'''bash
python simulate_electron_beam.py --model all --lens all --diffusion --outdir results
'''

## Arguments
Argument    Options Description
--model {convolution, first_order, all} Choose the simulation model
--lens	{with, without, all}    Select if using the electrostatic lens
--diffusion	(flag)  Add phosphor diffusion to the results
--outdir	folder_name Directory to save plots and outputs (default: plots)

## Output
Plots: histograms for X/Y distributions and 2D scatter of electron impact points.

Statistics: printed mean, standard deviation, and skewness of the electron beam.


## Example
'''bash
python MC_project.py --model convolution --lens with --outdir my_results
'''
Simulates the convolution model with the lens active and saves plots in my_results/.

License
MIT License