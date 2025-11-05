"""
Unified Distributed Training Launcher
Supports torchrun, DeepSpeed, and accelerate on Windows/Linux
"""
import os
import sys
import subprocess
import argparse
import platform
from pathlib import Path
from typing import List, Optional


def is_windows():
    """Check if running on Windows"""
    return platform.system() == 'Windows'


def build_torchrun_command(
    script: str,
    num_gpus: int,
    num_nodes: int = 1,
    node_rank: int = 0,
    master_addr: str = "localhost",
    master_port: int = 29500,
    script_args: Optional[List[str]] = None,
) -> List[str]:
    """Build torchrun command"""
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--standalone" if num_nodes == 1 else "",
        f"--nproc_per_node={num_gpus}",
    ]
    
    if num_nodes > 1:
        cmd.extend([
            f"--nnodes={num_nodes}",
            f"--node_rank={node_rank}",
            f"--master_addr={master_addr}",
            f"--master_port={master_port}",
        ])
    
    # Remove empty strings
    cmd = [c for c in cmd if c]
    
    cmd.append(script)
    
    if script_args:
        cmd.extend(script_args)
    
    return cmd


def build_deepspeed_command(
    script: str,
    num_gpus: int,
    num_nodes: int = 1,
    hostfile: Optional[str] = None,
    script_args: Optional[List[str]] = None,
) -> List[str]:
    """Build DeepSpeed command"""
    cmd = ["deepspeed"]
    
    if num_nodes == 1:
        cmd.append(f"--num_gpus={num_gpus}")
    else:
        if hostfile:
            cmd.append(f"--hostfile={hostfile}")
        else:
            cmd.append(f"--num_nodes={num_nodes}")
            cmd.append(f"--num_gpus={num_gpus}")
    
    cmd.append(script)
    
    if script_args:
        cmd.extend(script_args)
    
    return cmd


def build_accelerate_command(
    script: str,
    config_file: Optional[str] = None,
    num_processes: Optional[int] = None,
    script_args: Optional[List[str]] = None,
) -> List[str]:
    """Build accelerate command"""
    cmd = ["accelerate", "launch"]
    
    if config_file:
        cmd.extend(["--config_file", config_file])
    
    if num_processes:
        cmd.extend(["--num_processes", str(num_processes)])
    
    cmd.append(script)
    
    if script_args:
        cmd.extend(script_args)
    
    return cmd


def parse_args():
    """Parse launcher arguments"""
    parser = argparse.ArgumentParser(description="Distributed Training Launcher")
    
    # Launcher type
    parser.add_argument(
        "--launcher",
        type=str,
        default="torchrun",
        choices=["torchrun", "deepspeed", "accelerate"],
        help="Launcher to use"
    )
    
    # Script to run
    parser.add_argument(
        "--script",
        type=str,
        default="train_ultrathink.py",
        help="Training script to run"
    )
    
    # Distributed config
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs per node")
    parser.add_argument("--num_nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--node_rank", type=int, default=0, help="Node rank")
    parser.add_argument("--master_addr", type=str, default="localhost", help="Master address")
    parser.add_argument("--master_port", type=int, default=29500, help="Master port")
    
    # DeepSpeed specific
    parser.add_argument("--hostfile", type=str, default=None, help="DeepSpeed hostfile")
    
    # Accelerate specific
    parser.add_argument("--accelerate_config", type=str, default=None, help="Accelerate config file")
    
    # Environment variables
    parser.add_argument("--use_libuv", type=str, default="0", help="USE_LIBUV setting (0 or 1)")
    
    # Script arguments (everything after --)
    parser.add_argument("script_args", nargs=argparse.REMAINDER, help="Arguments for training script")
    
    return parser.parse_args()


def set_environment_variables(args):
    """Set environment variables for distributed training"""
    # Disable libuv on Windows if requested
    if is_windows() and args.use_libuv == "0":
        os.environ["USE_LIBUV"] = "0"
    
    # Set NCCL debug if needed
    if os.environ.get("NCCL_DEBUG") is None:
        os.environ["NCCL_DEBUG"] = "WARN"


def main():
    """Main launcher function"""
    args = parse_args()
    
    # Set environment variables
    set_environment_variables(args)
    
    # Clean script args (remove leading --)
    script_args = args.script_args
    if script_args and script_args[0] == "--":
        script_args = script_args[1:]
    
    # Build command based on launcher
    if args.launcher == "torchrun":
        cmd = build_torchrun_command(
            script=args.script,
            num_gpus=args.num_gpus,
            num_nodes=args.num_nodes,
            node_rank=args.node_rank,
            master_addr=args.master_addr,
            master_port=args.master_port,
            script_args=script_args,
        )
    elif args.launcher == "deepspeed":
        if is_windows():
            print("WARNING: DeepSpeed has limited Windows support. Consider using torchrun.")
        cmd = build_deepspeed_command(
            script=args.script,
            num_gpus=args.num_gpus,
            num_nodes=args.num_nodes,
            hostfile=args.hostfile,
            script_args=script_args,
        )
    elif args.launcher == "accelerate":
        cmd = build_accelerate_command(
            script=args.script,
            config_file=args.accelerate_config,
            num_processes=args.num_gpus * args.num_nodes if args.num_gpus else None,
            script_args=script_args,
        )
    else:
        raise ValueError(f"Unknown launcher: {args.launcher}")
    
    # Print command
    print(f"Launching with {args.launcher}:")
    print(" ".join(cmd))
    print()
    
    # Run command
    try:
        result = subprocess.run(cmd, check=True)
        sys.exit(result.returncode)
    except subprocess.CalledProcessError as e:
        print(f"Training failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)


if __name__ == "__main__":
    main()
