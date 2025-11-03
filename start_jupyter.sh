#!/bin/bash
# Start Jupyter notebook server

echo "Starting Jupyter Notebook server..."
echo "Open your browser to the URL shown below"
echo ""
echo "Notebooks available:"
echo "  00_Setup.ipynb - Setup and data loading"
echo "  01_HMM_Implementation.ipynb - HMM implementation"
echo "  02_RL_Agent.ipynb - RL agent implementation"
echo "  03_Training.ipynb - Training loop"
echo "  04_Evaluation.ipynb - Evaluation on test set"
echo ""

cd "$(dirname "$0")"
jupyter notebook notebooks/

