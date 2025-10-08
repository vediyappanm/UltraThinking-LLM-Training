"""
Enhanced Training Script with Real-Time Monitoring
Shows step-by-step progress with MoE and DRE metrics
"""

import subprocess
import sys
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

class EnhancedTrainingMonitor:
    def __init__(self):
        self.step = 0
        self.epoch = 1
        self.batch_in_step = 0
        
        # Metrics for current step
        self.dre_paths = defaultdict(int)
        self.dre_complexities = []
        self.dre_latencies = []
        self.moe_used = None
        self.moe_aux_loss = None
        self.moe_load_balance = None
        self.moe_z_loss = None
        
        # Training metrics
        self.current_loss = None
        self.current_ppl = None
        
    def reset_metrics(self):
        """Reset metrics for new step"""
        self.dre_paths.clear()
        self.dre_complexities.clear()
        self.dre_latencies.clear()
        self.moe_used = None
        self.moe_aux_loss = None
        self.moe_load_balance = None
        self.moe_z_loss = None
        self.batch_in_step = 0
        
    def process_line(self, line):
        """Process a log line"""
        
        # Configuration info at start
        if 'Training configuration:' in line:
            print("\n" + "="*90)
            print(" "*25 + "ULTRATHINKING LLM TRAINING")
            print("="*90)
            return
        
        # Model specs
        if any(x in line for x in ['hidden_size:', 'num_layers:', 'num_heads:', 'enable_moe:', 'enable_dre:']):
            if 'hidden_size:' in line:
                print(f"\n🏗️  MODEL ARCHITECTURE:")
            match = re.search(r'INFO -\s+(\w+):\s+(.+)', line)
            if match:
                key = match.group(1)
                value = match.group(2)
                if key in ['hidden_size', 'num_layers', 'num_heads', 'num_kv_heads', 
                          'intermediate_size', 'max_seq_length', 'activation',
                          'enable_moe', 'enable_dre', 'enable_constitutional']:
                    print(f"   {key:25s}: {value}")
            return
        
        # Dataset info
        if 'dataset:' in line and 'INFO' in line:
            match = re.search(r'dataset:\s+(.+)', line)
            if match:
                print(f"\n📚 DATASET: {match.group(1)}")
            return
        
        # Training start
        if 'Starting training...' in line or 'Epoch ' in line:
            if 'Epoch ' in line:
                self.epoch = int(re.search(r'Epoch (\d+)', line).group(1))
                print(f"\n{'='*90}")
                print(f"{'EPOCH ' + str(self.epoch):^90}")
                print(f"{'='*90}\n")
            return
        
        # DRE routing
        if 'DRE: Path=' in line:
            match = re.search(r'Path=(\w+), Complexity=([\d.]+), Latency=([\d.]+)ms', line)
            if match:
                path = match.group(1)
                complexity = float(match.group(2))
                latency = float(match.group(3))
                
                self.dre_paths[path] += 1
                self.dre_complexities.append(complexity)
                self.dre_latencies.append(latency)
                self.batch_in_step += 1
            return
        
        # MoE metrics
        if 'used_moe=' in line:
            match = re.search(r'used_moe=(\w+)', line)
            if match:
                self.moe_used = match.group(1) == 'True'
        
        if 'aux_total=' in line:
            match = re.search(r'aux_total=([\d.]+)', line)
            if match:
                self.moe_aux_loss = float(match.group(1))
        
        if 'Load balance:' in line:
            match = re.search(r'Load balance:\s*([\d.]+)', line)
            if match:
                self.moe_load_balance = float(match.group(1))
        
        if 'z-loss:' in line:
            match = re.search(r'z-loss:\s*([\d.]+)', line)
            if match:
                self.moe_z_loss = float(match.group(1))
        
        # Training step complete
        if '[train]' in line and 'avg_loss=' in line:
            match = re.search(r'avg_loss=([\d.]+)\s+avg_ppl=([\d.]+)', line)
            if match:
                self.current_loss = float(match.group(1))
                self.current_ppl = float(match.group(2))
                self.step += 1
                self.print_step_summary()
                self.reset_metrics()
            return
        
        # Validation progress
        if '[val_progress]' in line:
            match = re.search(r'batch=(\d+)\s+loss=([\d.]+)\s+ppl=([\d.]+)', line)
            if match:
                batch = int(match.group(1))
                val_loss = float(match.group(2))
                val_ppl = float(match.group(3))
                print(f"   📊 Validation Batch {batch:3d} | Loss: {val_loss:7.4f} | PPL: {val_ppl:10.2f}")
            return
    
    def print_step_summary(self):
        """Print formatted step summary"""
        print(f"\n{'─'*90}")
        print(f"⚡ STEP {self.step:4d} | {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'─'*90}")
        
        # Training metrics
        if self.current_loss is not None:
            print(f"\n📈 Training:")
            print(f"   Loss: {self.current_loss:.4f}  |  Perplexity: {self.current_ppl:.2f}")
        
        # DRE summary
        if self.dre_paths:
            total_routes = sum(self.dre_paths.values())
            print(f"\n🧠 Dynamic Reasoning Engine ({total_routes} batches):")
            
            # Path distribution
            for path, count in sorted(self.dre_paths.items(), key=lambda x: x[1], reverse=True):
                pct = (count / total_routes) * 100
                bar_length = int(pct / 2)  # 50 chars max
                bar = '█' * bar_length + '░' * (50 - bar_length)
                print(f"   {path.upper():8s} [{bar}] {count:3d} ({pct:5.1f}%)")
            
            # Stats
            if self.dre_complexities:
                avg_c = sum(self.dre_complexities) / len(self.dre_complexities)
                print(f"   Complexity: {avg_c:.3f} (min: {min(self.dre_complexities):.3f}, max: {max(self.dre_complexities):.3f})")
            
            if self.dre_latencies:
                avg_l = sum(self.dre_latencies) / len(self.dre_latencies)
                print(f"   Latency:    {avg_l:.1f}ms (min: {min(self.dre_latencies):.1f}ms, max: {max(self.dre_latencies):.1f}ms)")
        
        # MoE summary
        if self.moe_used is not None:
            status_icon = "✓" if self.moe_used else "✗"
            status_text = "Active" if self.moe_used else "Inactive"
            print(f"\n🔀 Mixture of Experts: {status_icon} {status_text}")
            
            if self.moe_used and any([self.moe_aux_loss, self.moe_load_balance, self.moe_z_loss]):
                if self.moe_aux_loss is not None:
                    print(f"   Aux Loss:      {self.moe_aux_loss:.4f}")
                if self.moe_load_balance is not None:
                    print(f"   Load Balance:  {self.moe_load_balance:.4f}")
                if self.moe_z_loss is not None:
                    print(f"   Z-Loss:        {self.moe_z_loss:.4f}")


def run_training_with_monitor(command):
    """Run training command and monitor output"""
    monitor = EnhancedTrainingMonitor()
    
    print("Starting ULTRATHINK training with enhanced monitoring...")
    print("="*90 + "\n")
    
    # Run the command and capture output
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    
    try:
        for line in process.stdout:
            # Process through monitor
            monitor.process_line(line.strip())
            
            # Also write to log file
            # with open('training_output.log', 'a') as f:
            #     f.write(line)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        process.terminate()
    
    process.wait()
    
    print("\n" + "="*90)
    print(f"Training completed with exit code: {process.returncode}")
    print("="*90)


if __name__ == "__main__":
    # Example: Modify this to match your training command
    if len(sys.argv) > 1:
        # Use command from arguments
        training_command = sys.argv[1:]
    else:
        # Default training command
        training_command = [
            sys.executable,  # Python executable
            "train_ultrathink.py",
            "--dataset", "c4",
            "--dataset_subset", "en",
            "--streaming",
            "--hidden_size", "512",
            "--num_layers", "6",
            "--num_heads", "8",
            "--num_kv_heads", "4",
            "--intermediate_size", "2048",
            "--max_seq_length", "256",
            "--activation", "swiglu",
            "--enable_moe",
            "--enable_dre",
            "--batch_size", "1",
            "--gradient_accumulation_steps", "16",
            "--learning_rate", "3e-4",
            "--use_amp",
            "--gradient_checkpointing",
            "--use_mlflow",
            "--run_name", "monitored_run",
            "--output_dir", "./outputs/monitored_run"
        ]
    
    run_training_with_monitor(training_command)
