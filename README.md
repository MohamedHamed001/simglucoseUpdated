# Updated Simglucose with RL Controller

This is an updated version of the simglucose simulator with an integrated Reinforcement Learning (RL) controller for automated insulin delivery.

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/simglucoseUpdated.git
cd simglucoseUpdated
```

2. Create a Python virtual environment (recommended):
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. Install the required packages and this project in editable mode:
```bash
pip install -r requirements.txt
pip install -e .
```

> **Note:**
> - You do NOT need to install the original simglucose separately. This package will use the official simglucose as a dependency.
> - The simglucose source code itself is NOT modified. All new features are in the scripts and RL_V3 folder.

## Usage

### Running the Simulation

To run a basic simulation with the trained RL controller:
```bash
python glucose_simulation_example.py
```

This will:
- Load the pre-trained RL controller
- Simulate a 24-hour scenario with 3 meals
- Display glucose levels and insulin delivery plots
- Show simulation statistics

### Training Your Own RL Controller


## File Structure

- `glucose_simulation_example.py`: Main simulation script
- `results/`: Directory where simulation results and trained models are saved

## Requirements

- Python 3.7 or higher
- NumPy
- Matplotlib
- Pandas
- SciPy
- PyTorch
- Simglucose

## Notes

- The simulation uses a 1-minute time step
- The RL controller is trained to maintain blood glucose between 70-180 mg/dL
- The default patient model is 'adolescent#001'
- The simulation includes realistic meal scenarios and sensor noise

## Troubleshooting

If you encounter any issues:
1. Make sure all dependencies are installed correctly
2. Check that you're using Python 3.7 or higher
3. Verify that the virtual environment is activated
4. Ensure you have write permissions in the directory for saving results

## License

This project is based on the original simglucose simulator. Please refer to the original license for details.
