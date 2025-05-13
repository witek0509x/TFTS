import argparse
import importlib

if __name__ == '__main__':
    # Create parser for command line arguments
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument(
        "--model_type", 
        type=str, 
        default="mlm",
        choices=["mlm"],
        help="Type of model to train"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/transformer_mlm/default.yaml",
        help="Path to the configuration file"
    )
    parser.add_argument(
        "--resume", 
        action="store_true", 
        help="Resume training from a checkpoint"
    )
    parser.add_argument(
        "--run_id", 
        type=str, 
        default=None,
        help="wandb run ID to resume"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Import the appropriate training module based on model_type
    if args.model_type == "mlm":
        from training.train_mlm import train
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    # Call the train function with the parsed arguments
    train(args.config, args.resume, args.run_id)