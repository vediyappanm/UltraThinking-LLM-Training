"""
Advanced Training Script for ULTRATHINK Model
Supports configuration files, all advanced features, and production deployment
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
import logging
from typing import Dict, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def merge_configs(base_config: Dict, overrides: Dict) -> Dict:
    """Merge override config into base config"""
    result = base_config.copy()
    for key, value in overrides.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


def config_to_args(config: Dict) -> argparse.Namespace:
    """Convert config dict to argparse Namespace for compatibility"""
    args = argparse.Namespace()
    
    # Model config
    model = config.get('model', {})
    args.vocab_size = model.get('vocab_size', 100352)
    args.hidden_size = model.get('hidden_size', 4096)
    args.num_layers = model.get('num_layers', 32)
    args.num_heads = model.get('num_heads', 32)
    args.num_kv_heads = model.get('num_kv_heads', 8)
    args.intermediate_size = model.get('intermediate_size', 14336)
    args.max_seq_length = model.get('max_seq_length', 8192)
    args.activation = model.get('activation', 'swiglu')
    args.dropout = model.get('dropout', 0.0)
    args.attention_dropout = model.get('attention_dropout', 0.0)
    args.use_flash_attention = model.get('use_flash_attention', False)
    args.gradient_checkpointing = model.get('gradient_checkpointing', False)
    
    # Advanced features
    advanced = config.get('advanced', {})
    args.enable_moe = advanced.get('enable_moe', False)
    args.enable_dre = advanced.get('enable_dre', False)
    args.enable_constitutional = advanced.get('enable_constitutional', False)
    args.enable_rlhf = advanced.get('enable_rlhf', False)
    args.enable_multimodal = advanced.get('enable_multimodal', False)
    args.dre_warmup_steps = advanced.get('dre_warmup_steps', 0)
    
    # MoE config
    moe = config.get('moe', {})
    args.num_knowledge_experts = moe.get('num_knowledge_experts', 64)
    args.num_skill_experts = moe.get('num_skill_experts', 32)
    args.num_meta_experts = moe.get('num_meta_experts', 16)
    args.num_safety_experts = moe.get('num_safety_experts', 8)
    args.moe_top_k = moe.get('moe_top_k', 2)
    args.expert_capacity = moe.get('expert_capacity', 1.25)
    
    # Multimodal config
    multimodal = config.get('multimodal', {})
    args.image_size = multimodal.get('image_size', 224)
    args.patch_size = multimodal.get('patch_size', 14)
    args.audio_sample_rate = multimodal.get('audio_sample_rate', 16000)
    
    # Training config
    training = config.get('training', {})
    args.batch_size = training.get('batch_size', 32)
    args.gradient_accumulation_steps = training.get('gradient_accumulation_steps', 4)
    args.learning_rate = training.get('learning_rate', 3e-5)
    args.weight_decay = training.get('weight_decay', 0.01)
    args.adam_beta1 = training.get('adam_beta1', 0.9)
    args.adam_beta2 = training.get('adam_beta2', 0.999)
    args.warmup_steps = training.get('warmup_steps', 10000)
    args.max_steps = training.get('max_steps', 1000000)
    args.num_epochs = training.get('num_epochs', 3)
    args.gradient_clipping = training.get('gradient_clipping', 1.0)
    args.use_amp = training.get('use_amp', False)
    
    # Distributed config
    distributed = config.get('distributed', {})
    args.distributed = distributed.get('enabled', False)
    args.use_4d_parallelism = distributed.get('use_4d_parallelism', False)
    args.data_parallel_size = distributed.get('data_parallel_size', 1)
    args.tensor_parallel_size = distributed.get('tensor_parallel_size', 1)
    args.pipeline_parallel_size = distributed.get('pipeline_parallel_size', 1)
    args.expert_parallel_size = distributed.get('expert_parallel_size', 1)
    args.zero_stage = distributed.get('zero_stage', 0)
    args.deepspeed = distributed.get('deepspeed_config', None)
    args.launcher = distributed.get('launcher', 'none')
    
    # Data config
    data = config.get('data', {})
    args.dataset = data.get('dataset', 'wikitext')
    args.mix_datasets = data.get('mix_datasets', None)
    args.dataset_subset = data.get('dataset_subset', None)
    args.data_path = data.get('data_path', None)
    args.text_column = data.get('text_column', 'text')
    args.tokenizer_name = data.get('tokenizer_name', 'gpt2')
    args.max_samples = data.get('max_samples', None)
    args.train_samples = data.get('train_samples', 10000)
    args.val_samples = data.get('val_samples', 1000)
    args.num_workers = data.get('num_workers', 4)
    args.streaming = data.get('streaming', False)
    args.use_synthetic_data = data.get('use_synthetic_data', False)
    args.synthetic_samples = data.get('synthetic_samples', 5000)
    
    # RLHF config
    rlhf = config.get('rlhf', {})
    args.rlhf_frequency = rlhf.get('rlhf_frequency', 5)
    args.rlhf_iterations = rlhf.get('rlhf_iterations', 100)
    args.rlhf_steps_per_iteration = rlhf.get('rlhf_steps_per_iteration', 1000)
    args.ppo_epochs = rlhf.get('ppo_epochs', 4)
    args.ppo_batch_size = rlhf.get('ppo_batch_size', 32)
    
    # Evaluation config
    evaluation = config.get('evaluation', {})
    args.eval_frequency = evaluation.get('eval_frequency', 5)
    
    # Logging config
    logging_cfg = config.get('logging', {})
    args.use_mlflow = logging_cfg.get('use_mlflow', False)
    args.mlflow_tracking_uri = logging_cfg.get('mlflow_tracking_uri', 'file:./mlruns')
    args.mlflow_experiment = logging_cfg.get('mlflow_experiment', 'UltraThinking-LLM-Training')
    args.run_name = logging_cfg.get('run_name', 'ultrathink_training')
    args.use_wandb = False  # Deprecated
    
    # Output config
    output = config.get('output', {})
    args.output_dir = output.get('output_dir', './outputs/ultrathink')
    
    # Resume/init
    args.init_from_model_dir = None
    args.resume_checkpoint = None
    args.continuous = False
    
    return args


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Advanced Training Script for ULTRATHINK')
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to YAML configuration file (e.g., configs/train_small.yaml)'
    )
    
    parser.add_argument(
        '--override',
        type=str,
        nargs='*',
        help='Override config values (e.g., training.batch_size=16 model.hidden_size=512)'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    
    parser.add_argument(
        '--init-from',
        type=str,
        default=None,
        help='Path to pretrained model directory to initialize from'
    )
    
    parser.add_argument(
        '--continuous',
        action='store_true',
        help='Train continuously until interrupted'
    )
    
    parser.add_argument(
        '--run-name',
        type=str,
        default=None,
        help='Override run name from config'
    )
    
    return parser.parse_args()


def apply_overrides(config: Dict, overrides: list) -> Dict:
    """Apply command-line overrides to config"""
    if not overrides:
        return config
    
    for override in overrides:
        if '=' not in override:
            logger.warning(f"Invalid override format: {override}. Use key=value")
            continue
        
        key_path, value = override.split('=', 1)
        keys = key_path.split('.')
        
        # Try to convert value to appropriate type
        try:
            if value.lower() == 'true':
                value = True
            elif value.lower() == 'false':
                value = False
            elif value.lower() == 'null' or value.lower() == 'none':
                value = None
            elif '.' in value:
                value = float(value)
            else:
                try:
                    value = int(value)
                except ValueError:
                    pass  # Keep as string
        except Exception:
            pass  # Keep as string
        
        # Apply override
        current = config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value
    
    return config


def main():
    """Main training function"""
    # Parse arguments
    args = parse_args()
    
    # Load base configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Apply overrides
    if args.override:
        logger.info(f"Applying overrides: {args.override}")
        config = apply_overrides(config, args.override)
    
    # Convert to argparse Namespace
    train_args = config_to_args(config)
    
    # Apply resume/init flags
    if args.resume:
        train_args.resume_checkpoint = args.resume
    if args.init_from:
        train_args.init_from_model_dir = args.init_from
    if args.continuous:
        train_args.continuous = True
    if args.run_name:
        train_args.run_name = args.run_name
    
    # Create output directory
    Path(train_args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(train_args.output_dir, 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Save effective configuration
    config_save_path = os.path.join(train_args.output_dir, 'effective_config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info(f"Saved effective configuration to {config_save_path}")
    
    # Log configuration
    logger.info("=" * 80)
    logger.info("ULTRATHINK Advanced Training")
    logger.info("=" * 80)
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Output directory: {train_args.output_dir}")
    logger.info(f"Run name: {train_args.run_name}")
    logger.info("")
    logger.info("Model Configuration:")
    logger.info(f"  Hidden size: {train_args.hidden_size}")
    logger.info(f"  Layers: {train_args.num_layers}")
    logger.info(f"  Heads: {train_args.num_heads}")
    logger.info(f"  Sequence length: {train_args.max_seq_length}")
    logger.info("")
    logger.info("Advanced Features:")
    logger.info(f"  MoE: {train_args.enable_moe}")
    logger.info(f"  DRE: {train_args.enable_dre}")
    logger.info(f"  Constitutional AI: {train_args.enable_constitutional}")
    logger.info(f"  RLHF: {train_args.enable_rlhf}")
    logger.info(f"  Multimodal: {train_args.enable_multimodal}")
    logger.info("")
    logger.info("Training Configuration:")
    logger.info(f"  Batch size: {train_args.batch_size}")
    logger.info(f"  Gradient accumulation: {train_args.gradient_accumulation_steps}")
    logger.info(f"  Learning rate: {train_args.learning_rate}")
    logger.info(f"  Epochs: {train_args.num_epochs}")
    logger.info("=" * 80)
    
    # Import and run training
    from train_ultrathink import UltraThinkTrainer
    import mlflow
    
    # Create trainer
    trainer = UltraThinkTrainer(train_args)
    
    # Start MLflow run if enabled
    active_mlflow = False
    if train_args.use_mlflow and trainer.is_main_process():
        try:
            mlflow.set_tracking_uri(train_args.mlflow_tracking_uri)
            mlflow.set_experiment(train_args.mlflow_experiment)
            mlflow.start_run(run_name=train_args.run_name)
            
            # Log configuration as params
            safe_params = {
                k: (str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v)
                for k, v in vars(train_args).items()
            }
            mlflow.log_params(safe_params)
            
            # Log config file as artifact
            mlflow.log_artifact(args.config, artifact_path='config')
            if os.path.exists(config_save_path):
                mlflow.log_artifact(config_save_path, artifact_path='config')
            
            active_mlflow = True
            logger.info(f"MLflow tracking enabled: {train_args.mlflow_tracking_uri}")
        except Exception as e:
            logger.warning(f"Failed to start MLflow run: {e}")
            active_mlflow = False
    
    # Run training
    try:
        results = trainer.train()
        
        # Log final results
        logger.info("=" * 80)
        logger.info("Training completed successfully!")
        logger.info(f"Final results: {results}")
        logger.info("=" * 80)
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        results = {'status': 'interrupted'}
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        results = {'status': 'failed', 'error': str(e)}
        raise
    finally:
        # Cleanup MLflow
        if train_args.use_mlflow and trainer.is_main_process() and active_mlflow:
            try:
                results_path = os.path.join(train_args.output_dir, 'evaluation_results.json')
                if os.path.exists(results_path):
                    mlflow.log_artifact(results_path, artifact_path='evaluation')
            finally:
                try:
                    mlflow.end_run()
                except Exception:
                    pass


if __name__ == "__main__":
    main()
