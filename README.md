# Transformer-based Time Series Forecasting

This repository contains code for training transformer-based models for time series prediction using masked language modeling techniques.

## Inference Visualization

After training a model, you can visualize its performance using the `inference_visualization.py` script. This script:

1. Downloads a trained model from Weights & Biases using a run ID
2. Rebuilds the Echo State Network based on the model's configuration
3. Generates new time series data with this ESN
4. Applies random masking to tokens in the series
5. Performs inference with the transformer model
6. Visualizes the original, masked, and predicted series

### Running the script

```bash
python inference_visualization.py --run_id YOUR_WANDB_RUN_ID [--output path/to/save/plot.png]
```

Arguments:
- `--run_id`: (Required) The Weights & Biases run ID of your trained model
- `--output`: (Optional) Path to save the output plot

### Example

```bash
python inference_visualization.py --run_id abc123def456 --output results/inference_plot.png
```

This will:
1. Download the model from the specified run
2. Generate a new time series using the same ESN parameters
3. Mask ~20% of the tokens (or whatever masking ratio was used during training)
4. Run inference on the masked series
5. Generate a plot showing the original data, masked input, and model predictions
6. Calculate and display the R² score on the masked tokens
7. Save the plot to the specified location (if provided)

## Model Training

To train a new model, use the `run_training.py` script:

```bash
python run_training.py --config configs/transformer_mlm/default.yaml
```

