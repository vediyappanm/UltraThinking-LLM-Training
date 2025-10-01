"""Recover from corrupted or incomplete checkpoints"""
import torch
import os
import argparse
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def recover_checkpoint(checkpoint_dir: str, output_path: str, verbose: bool = True):
    """
    Try to recover a corrupted checkpoint
    
    Args:
        checkpoint_dir: Directory containing checkpoint files
        output_path: Path to save recovered checkpoint
        verbose: Print detailed information
    
    Returns:
        True if recovery successful, False otherwise
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        logger.error(f"Checkpoint directory not found: {checkpoint_dir}")
        return False
    
    # Find all checkpoint files
    checkpoint_files = list(checkpoint_dir.glob("*.pt")) + list(checkpoint_dir.glob("*.pth"))
    
    if not checkpoint_files:
        logger.error(f"No checkpoint files found in {checkpoint_dir}")
        return False
    
    logger.info(f"Found {len(checkpoint_files)} checkpoint files")
    
    # Try to load checkpoints in reverse order (newest first)
    checkpoint_files = sorted(checkpoint_files, key=os.path.getmtime, reverse=True)
    
    for ckpt_file in checkpoint_files:
        try:
            if verbose:
                logger.info(f"Attempting to load: {ckpt_file.name}")
            
            # Try loading
            checkpoint = torch.load(ckpt_file, map_location='cpu')
            
            # Validate checkpoint structure
            if not isinstance(checkpoint, dict):
                logger.warning(f"  ✗ Not a dictionary: {type(checkpoint)}")
                continue
            
            required_keys = ['model_state_dict']
            optional_keys = ['optimizer_state_dict', 'scheduler_state_dict', 'epoch', 'global_step', 'loss']
            
            if not all(k in checkpoint for k in required_keys):
                missing = [k for k in required_keys if k not in checkpoint]
                logger.warning(f"  ✗ Missing required keys: {missing}")
                logger.info(f"  Available keys: {list(checkpoint.keys())}")
                continue
            
            # Checkpoint is valid
            logger.info(f"  ✓ Valid checkpoint found: {ckpt_file.name}")
            
            # Print checkpoint info
            if verbose:
                logger.info(f"  Checkpoint information:")
                for key in optional_keys:
                    if key in checkpoint:
                        value = checkpoint[key]
                        if key in ['epoch', 'global_step']:
                            logger.info(f"    {key}: {value}")
                        elif key == 'loss':
                            logger.info(f"    {key}: {value:.6f}")
                
                # Count model parameters
                model_state = checkpoint['model_state_dict']
                num_params = sum(v.numel() for v in model_state.values())
                logger.info(f"    Parameters: {num_params:,}")
            
            # Save recovered checkpoint
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            torch.save(checkpoint, output_path)
            logger.info(f"  ✓ Saved recovered checkpoint to: {output_path}")
            
            return True
            
        except Exception as e:
            logger.warning(f"  ✗ Failed to load {ckpt_file.name}: {e}")
            continue
    
    logger.error("✗ No valid checkpoint could be recovered")
    return False


def inspect_checkpoint(checkpoint_path: str):
    """Inspect a checkpoint file"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        print("=" * 60)
        print(f"Checkpoint: {checkpoint_path}")
        print("=" * 60)
        
        if not isinstance(checkpoint, dict):
            print(f"Type: {type(checkpoint)}")
            print("Not a dictionary - unexpected format")
            return
        
        print(f"Keys: {list(checkpoint.keys())}")
        print()
        
        # Model state
        if 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
            num_params = sum(v.numel() for v in model_state.values())
            print(f"Model parameters: {num_params:,}")
            print(f"Model state keys: {len(model_state)}")
        
        # Training info
        if 'epoch' in checkpoint:
            print(f"Epoch: {checkpoint['epoch']}")
        
        if 'global_step' in checkpoint:
            print(f"Global step: {checkpoint['global_step']}")
        
        if 'loss' in checkpoint:
            print(f"Loss: {checkpoint['loss']:.6f}")
        
        # Optimizer
        if 'optimizer_state_dict' in checkpoint:
            print("Optimizer state: Present")
        
        # Scheduler
        if 'scheduler_state_dict' in checkpoint:
            print("Scheduler state: Present")
        
        print("=" * 60)
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")


def main():
    parser = argparse.ArgumentParser(description='Recover corrupted checkpoints')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Directory containing checkpoint files')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to save recovered checkpoint')
    parser.add_argument('--inspect', type=str, default=None,
                        help='Inspect a specific checkpoint file')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed information')
    
    args = parser.parse_args()
    
    if args.inspect:
        inspect_checkpoint(args.inspect)
    else:
        success = recover_checkpoint(args.checkpoint_dir, args.output, args.verbose)
        exit(0 if success else 1)


if __name__ == "__main__":
    main()
