"""
Complete Training Script for UltraThinking LLM
This script demonstrates all available training flags and features with checkpointing
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def create_complete_training_args():
    """
    Create argument parser with ALL available flags for comprehensive training
    """
    parser = argparse.ArgumentParser(
        description='Complete UltraThinking Model Training with All Features',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # ============================================================================
    # MODEL ARCHITECTURE
    # ============================================================================
    arch_group = parser.add_argument_group('Model Architecture')
    arch_group.add_argument('--vocab_size', type=int, default=50257,
                           help='Vocabulary size (GPT-2: 50257, Llama: 32000)')
    arch_group.add_argument('--hidden_size', type=int, default=768,
                           help='Hidden dimension size (small: 768, base: 1024, large: 2048)')
    arch_group.add_argument('--num_layers', type=int, default=12,
                           help='Number of transformer layers (small: 12, base: 24, large: 32)')
    arch_group.add_argument('--num_heads', type=int, default=12,
                           help='Number of attention heads')
    arch_group.add_argument('--num_kv_heads', type=int, default=12,
                           help='Number of key-value heads for GQA (Grouped Query Attention)')
    arch_group.add_argument('--intermediate_size', type=int, default=3072,
                           help='FFN intermediate size (usually 4x hidden_size)')
    arch_group.add_argument('--max_seq_length', type=int, default=1024,
                           help='Maximum sequence length')
    arch_group.add_argument('--activation', type=str, default='swiglu',
                           choices=['gelu', 'relu', 'swiglu', 'silu'],
                           help='Activation function')
    
    # ============================================================================
    # MIXTURE OF EXPERTS (MoE)
    # ============================================================================
    moe_group = parser.add_argument_group('Mixture of Experts (MoE)')
    moe_group.add_argument('--enable_moe', action='store_true',
                          help='Enable Mixture of Experts architecture')
    moe_group.add_argument('--num_knowledge_experts', type=int, default=8,
                          help='Number of knowledge experts')
    moe_group.add_argument('--num_skill_experts', type=int, default=4,
                          help='Number of skill experts')
    moe_group.add_argument('--num_meta_experts', type=int, default=2,
                          help='Number of meta experts')
    moe_group.add_argument('--num_safety_experts', type=int, default=2,
                          help='Number of safety experts')
    moe_group.add_argument('--moe_top_k', type=int, default=2,
                          help='Number of experts to route to (top-k)')
    moe_group.add_argument('--expert_capacity', type=float, default=1.25,
                          help='Expert capacity factor')
    moe_group.add_argument('--load_balance_weight', type=float, default=0.01,
                          help='Weight for load balancing loss')
    moe_group.add_argument('--z_loss_weight', type=float, default=0.001,
                          help='Weight for router logit regularization')
    moe_group.add_argument('--importance_weight', type=float, default=0.01,
                          help='Weight for routing diversity loss')
    
    # ============================================================================
    # MULTIMODAL
    # ============================================================================
    mm_group = parser.add_argument_group('Multimodal Settings')
    mm_group.add_argument('--enable_multimodal', action='store_true',
                         help='Enable multimodal capabilities (text, image, audio)')
    mm_group.add_argument('--image_size', type=int, default=224,
                         help='Input image size')
    mm_group.add_argument('--patch_size', type=int, default=14,
                         help='Vision transformer patch size')
    mm_group.add_argument('--audio_sample_rate', type=int, default=16000,
                         help='Audio sample rate in Hz')
    
    # ============================================================================
    # ADVANCED FEATURES
    # ============================================================================
    adv_group = parser.add_argument_group('Advanced Features')
    adv_group.add_argument('--enable_dre', action='store_true',
                          help='Enable Dynamic Reasoning Engine')
    adv_group.add_argument('--enable_constitutional', action='store_true',
                          help='Enable Constitutional AI for safety')
    adv_group.add_argument('--enable_rlhf', action='store_true',
                          help='Enable Reinforcement Learning from Human Feedback')
    adv_group.add_argument('--dre_warmup_steps', type=int, default=1000,
                          help='Disable DRE for first N steps to stabilize training')
    adv_group.add_argument('--dre_force_path', type=str, default=None,
                          choices=['fast', 'standard', 'expert', 'deep', 'ultra_deep'],
                          help='Force specific DRE reasoning path for debugging')
    
    # ============================================================================
    # TRAINING HYPERPARAMETERS
    # ============================================================================
    train_group = parser.add_argument_group('Training Hyperparameters')
    train_group.add_argument('--batch_size', type=int, default=8,
                            help='Training batch size per device')
    train_group.add_argument('--gradient_accumulation_steps', type=int, default=4,
                            help='Number of gradient accumulation steps')
    train_group.add_argument('--learning_rate', type=float, default=3e-4,
                            help='Peak learning rate')
    train_group.add_argument('--weight_decay', type=float, default=0.01,
                            help='Weight decay for AdamW optimizer')
    train_group.add_argument('--adam_beta1', type=float, default=0.9,
                            help='Adam beta1 parameter')
    train_group.add_argument('--adam_beta2', type=float, default=0.999,
                            help='Adam beta2 parameter')
    train_group.add_argument('--warmup_steps', type=int, default=2000,
                            help='Number of warmup steps for learning rate')
    train_group.add_argument('--max_steps', type=int, default=100000,
                            help='Maximum training steps')
    train_group.add_argument('--num_epochs', type=int, default=3,
                            help='Number of training epochs')
    train_group.add_argument('--gradient_clipping', type=float, default=1.0,
                            help='Gradient clipping threshold (0 to disable)')
    
    # ============================================================================
    # REGULARIZATION
    # ============================================================================
    reg_group = parser.add_argument_group('Regularization')
    reg_group.add_argument('--dropout', type=float, default=0.1,
                          help='Dropout probability')
    reg_group.add_argument('--attention_dropout', type=float, default=0.1,
                          help='Attention dropout probability')
    
    # ============================================================================
    # OPTIMIZATION & PERFORMANCE
    # ============================================================================
    opt_group = parser.add_argument_group('Optimization & Performance')
    opt_group.add_argument('--use_flash_attention', action='store_true',
                          help='Use Flash Attention 2 for faster training')
    opt_group.add_argument('--gradient_checkpointing', action='store_true',
                          help='Enable gradient checkpointing to save memory')
    opt_group.add_argument('--use_amp', action='store_true',
                          help='Use Automatic Mixed Precision (AMP)')
    opt_group.add_argument('--amp_warmup_steps', type=int, default=0,
                          help='Disable AMP for first N steps to stabilize')
    
    # ============================================================================
    # DISTRIBUTED TRAINING
    # ============================================================================
    dist_group = parser.add_argument_group('Distributed Training')
    dist_group.add_argument('--distributed', action='store_true',
                           help='Enable distributed training')
    dist_group.add_argument('--use_4d_parallelism', action='store_true',
                           help='Enable 4D parallelism (data, tensor, pipeline, expert)')
    dist_group.add_argument('--data_parallel_size', type=int, default=1,
                           help='Data parallel size')
    dist_group.add_argument('--tensor_parallel_size', type=int, default=1,
                           help='Tensor parallel size')
    dist_group.add_argument('--pipeline_parallel_size', type=int, default=1,
                           help='Pipeline parallel size')
    dist_group.add_argument('--expert_parallel_size', type=int, default=1,
                           help='Expert parallel size (for MoE)')
    dist_group.add_argument('--zero_stage', type=int, default=0,
                           choices=[0, 1, 2, 3],
                           help='DeepSpeed ZeRO optimization stage')
    dist_group.add_argument('--deepspeed', type=str, default=None,
                           help='Path to DeepSpeed config JSON')
    dist_group.add_argument('--launcher', type=str, default='none',
                           choices=['none', 'deepspeed', 'accelerate', 'torchrun'],
                           help='Distributed launcher type')
    
    # ============================================================================
    # RLHF SETTINGS
    # ============================================================================
    rlhf_group = parser.add_argument_group('RLHF Settings')
    rlhf_group.add_argument('--rlhf_frequency', type=int, default=5,
                           help='Run RLHF every N epochs')
    rlhf_group.add_argument('--rlhf_iterations', type=int, default=100,
                           help='Number of RLHF iterations')
    rlhf_group.add_argument('--rlhf_steps_per_iteration', type=int, default=1000,
                           help='Steps per RLHF iteration')
    rlhf_group.add_argument('--ppo_epochs', type=int, default=4,
                           help='PPO epochs per update')
    rlhf_group.add_argument('--ppo_batch_size', type=int, default=32,
                           help='PPO batch size')
    
    # ============================================================================
    # DATASET CONFIGURATION
    # ============================================================================
    data_group = parser.add_argument_group('Dataset Configuration')
    data_group.add_argument('--dataset', type=str, default='custom',
                           choices=['wikitext', 'openwebtext', 'pile', 'c4', 
                                   'bookcorpus', 'dummy', 'custom'],
                           help='Dataset to use')
    data_group.add_argument('--mix_datasets', type=str, default=None,
                           help='Mix datasets with weights (e.g., "wikitext:0.5,c4:0.5")')
    data_group.add_argument('--dataset_subset', type=str, default=None,
                           help='Dataset subset/config name')
    data_group.add_argument('--data_path', type=str, default='./easy_dataset.json',
                           help='Path to custom dataset file')
    data_group.add_argument('--text_column', type=str, default='text',
                           help='Column name containing text data')
    data_group.add_argument('--tokenizer_name', type=str, default='gpt2',
                           help='Tokenizer to use (gpt2, llama, etc.)')
    data_group.add_argument('--max_samples', type=int, default=None,
                           help='Maximum number of samples to use')
    data_group.add_argument('--streaming', action='store_true',
                           help='Enable streaming mode for large datasets')
    data_group.add_argument('--train_samples', type=int, default=10000,
                           help='Number of training samples (for dummy dataset)')
    data_group.add_argument('--val_samples', type=int, default=1000,
                           help='Number of validation samples (for dummy dataset)')
    data_group.add_argument('--num_workers', type=int, default=2,
                           help='Number of data loading workers')
    data_group.add_argument('--use_synthetic_data', action='store_true',
                           help='Use synthetic data generation')
    data_group.add_argument('--synthetic_samples', type=int, default=5000,
                           help='Number of synthetic samples to generate')
    
    # ============================================================================
    # EVALUATION
    # ============================================================================
    eval_group = parser.add_argument_group('Evaluation')
    eval_group.add_argument('--eval_frequency', type=int, default=1,
                           help='Evaluate every N epochs')
    
    # ============================================================================
    # LOGGING & MONITORING
    # ============================================================================
    log_group = parser.add_argument_group('Logging & Monitoring')
    log_group.add_argument('--use_wandb', action='store_true',
                          help='Enable Weights & Biases logging')
    log_group.add_argument('--use_mlflow', action='store_true',
                          help='Enable MLflow experiment tracking')
    log_group.add_argument('--mlflow_tracking_uri', type=str, default='file:./mlruns',
                          help='MLflow tracking URI')
    log_group.add_argument('--mlflow_experiment', type=str, default='UltraThinking-Complete',
                          help='MLflow experiment name')
    log_group.add_argument('--run_name', type=str, default='complete_training',
                          help='Run name for logging')
    log_group.add_argument('--perf_log_interval', type=int, default=100,
                          help='Log performance metrics every N batches')
    
    # ============================================================================
    # CHECKPOINTING & OUTPUT
    # ============================================================================
    ckpt_group = parser.add_argument_group('Checkpointing & Output')
    ckpt_group.add_argument('--output_dir', type=str, default='./outputs/complete_training',
                           help='Output directory for checkpoints and logs')
    ckpt_group.add_argument('--init_from_model_dir', type=str, default=None,
                           help='Initialize from pretrained model directory')
    ckpt_group.add_argument('--resume_checkpoint', type=str, default=None,
                           help='Resume training from checkpoint file (.pt)')
    ckpt_group.add_argument('--save_checkpoint_every', type=int, default=1,
                           help='Save checkpoint every N epochs')
    ckpt_group.add_argument('--keep_last_n_checkpoints', type=int, default=3,
                           help='Keep only last N checkpoints to save space')
    ckpt_group.add_argument('--continuous', action='store_true',
                           help='Train indefinitely until interrupted')
    
    return parser


def main():
    """Main training function with comprehensive configuration"""
    
    # Parse arguments
    parser = create_complete_training_args()
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Print configuration summary
    print("=" * 80)
    print("ULTRATHINKING COMPLETE TRAINING CONFIGURATION")
    print("=" * 80)
    print("\n📊 MODEL ARCHITECTURE:")
    print(f"  • Hidden Size: {args.hidden_size}")
    print(f"  • Layers: {args.num_layers}")
    print(f"  • Attention Heads: {args.num_heads}")
    print(f"  • Sequence Length: {args.max_seq_length}")
    print(f"  • Activation: {args.activation}")
    
    print("\n🧠 ADVANCED FEATURES:")
    print(f"  • MoE Enabled: {args.enable_moe}")
    print(f"  • Dynamic Reasoning: {args.enable_dre}")
    print(f"  • Constitutional AI: {args.enable_constitutional}")
    print(f"  • RLHF: {args.enable_rlhf}")
    print(f"  • Multimodal: {args.enable_multimodal}")
    
    print("\n⚡ OPTIMIZATION:")
    print(f"  • Flash Attention: {args.use_flash_attention}")
    print(f"  • Gradient Checkpointing: {args.gradient_checkpointing}")
    print(f"  • Mixed Precision (AMP): {args.use_amp}")
    print(f"  • Gradient Clipping: {args.gradient_clipping}")
    
    print("\n📚 TRAINING:")
    print(f"  • Dataset: {args.dataset}")
    print(f"  • Data Path: {args.data_path}")
    print(f"  • Batch Size: {args.batch_size}")
    print(f"  • Gradient Accumulation: {args.gradient_accumulation_steps}")
    print(f"  • Learning Rate: {args.learning_rate}")
    print(f"  • Epochs: {args.num_epochs}")
    print(f"  • Warmup Steps: {args.warmup_steps}")
    
    print("\n💾 CHECKPOINTING:")
    print(f"  • Output Directory: {args.output_dir}")
    print(f"  • Save Every: {args.save_checkpoint_every} epoch(s)")
    print(f"  • Resume From: {args.resume_checkpoint or 'None'}")
    
    print("\n📊 LOGGING:")
    print(f"  • MLflow: {args.use_mlflow}")
    print(f"  • Weights & Biases: {args.use_wandb}")
    
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80 + "\n")
    
    # Import and run the actual training
    from train_ultrathink import UltraThinkTrainer
    import logging
    
    # Setup logging
    log_file = os.path.join(args.output_dir, 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting UltraThinking complete training pipeline")
    
    # Create trainer and run
    trainer = UltraThinkTrainer(args)
    
    # Start MLflow if enabled
    if args.use_mlflow and trainer.is_main_process():
        import mlflow
        try:
            mlflow.set_tracking_uri(args.mlflow_tracking_uri)
            mlflow.set_experiment(args.mlflow_experiment)
            mlflow.start_run(run_name=args.run_name)
            mlflow.log_params({k: str(v) for k, v in vars(args).items()})
            logger.info("MLflow tracking started")
        except Exception as e:
            logger.warning(f"Failed to start MLflow: {e}")
    
    # Run training
    try:
        results = trainer.train()
        logger.info("Training completed successfully!")
        logger.info(f"Final results: {results}")
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise
    finally:
        # End MLflow run
        if args.use_mlflow and trainer.is_main_process():
            try:
                import mlflow
                mlflow.end_run()
            except:
                pass
    
    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\n📁 Outputs saved to: {args.output_dir}")
    print(f"📊 Logs available at: {log_file}")
    print("\n")


if __name__ == "__main__":
    main()
