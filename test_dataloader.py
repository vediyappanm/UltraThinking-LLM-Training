"""
Quick test script to verify DataLoader configuration works without hanging
"""
import torch
from torch.utils.data import DataLoader, Dataset
import time

class SimpleDataset(Dataset):
    def __init__(self, size=100, seq_len=512):
        self.size = size
        self.seq_len = seq_len
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return {
            'input_ids': torch.randint(0, 50257, (self.seq_len,)),
            'attention_mask': torch.ones(self.seq_len, dtype=torch.long),
            'labels': torch.randint(0, 50257, (self.seq_len,))
        }

def test_dataloader():
    print("=" * 60)
    print("Testing DataLoader Configuration")
    print("=" * 60)
    
    dataset = SimpleDataset(size=50)
    
    # Test with Windows-safe configuration
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,  # Safe for Windows
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
        prefetch_factor=None,
        timeout=0,
        drop_last=True
    )
    
    print(f"✓ DataLoader created successfully")
    print(f"  - Dataset size: {len(dataset)}")
    print(f"  - Batch size: 4")
    print(f"  - Expected batches: {len(dataset) // 4}")
    print(f"  - num_workers: 0 (Windows-safe)")
    print()
    
    # Test iteration
    print("Testing iteration...")
    start_time = time.time()
    
    try:
        iterator = iter(loader)
        print("✓ Iterator created successfully")
    except Exception as e:
        print(f"✗ Failed to create iterator: {e}")
        return
    
    batch_count = 0
    for i in range(min(10, len(loader))):
        try:
            batch = next(iterator)
            batch_count += 1
            
            # Verify batch structure
            assert 'input_ids' in batch
            assert 'attention_mask' in batch
            assert 'labels' in batch
            assert batch['input_ids'].shape[0] == 4
            
            if i == 0:
                print(f"✓ First batch fetched successfully")
                print(f"  - Batch shape: {batch['input_ids'].shape}")
            elif i == 1:
                print(f"✓ Second batch fetched successfully (CRITICAL TEST)")
            elif i % 3 == 0:
                print(f"✓ Batch {i+1} fetched successfully")
                
        except StopIteration:
            print(f"✓ Reached end of dataset at batch {batch_count}")
            break
        except Exception as e:
            print(f"✗ Error at batch {i}: {e}")
            return
    
    elapsed = time.time() - start_time
    print()
    print(f"✓ Successfully iterated through {batch_count} batches")
    print(f"✓ Total time: {elapsed:.2f}s")
    print(f"✓ Time per batch: {elapsed/batch_count:.3f}s")
    print()
    print("=" * 60)
    print("✓ ALL TESTS PASSED - DataLoader is working correctly!")
    print("=" * 60)

if __name__ == "__main__":
    test_dataloader()
