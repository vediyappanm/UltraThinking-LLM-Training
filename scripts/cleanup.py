"""Cleanup script to remove unwanted files and directories"""
import os
import shutil
from pathlib import Path


def remove_pycache(root_dir):
    """Remove all __pycache__ directories"""
    count = 0
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if '__pycache__' in dirnames:
            cache_path = Path(dirpath) / '__pycache__'
            print(f"Removing: {cache_path}")
            shutil.rmtree(cache_path, ignore_errors=True)
            count += 1
    return count


def remove_pyc_files(root_dir):
    """Remove all .pyc files"""
    count = 0
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.pyc') or filename.endswith('.pyo'):
                file_path = Path(dirpath) / filename
                print(f"Removing: {file_path}")
                file_path.unlink(missing_ok=True)
                count += 1
    return count


def remove_pytest_cache(root_dir):
    """Remove pytest cache directories"""
    count = 0
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if '.pytest_cache' in dirnames:
            cache_path = Path(dirpath) / '.pytest_cache'
            print(f"Removing: {cache_path}")
            shutil.rmtree(cache_path, ignore_errors=True)
            count += 1
    return count


def remove_coverage_files(root_dir):
    """Remove coverage files and directories"""
    count = 0
    root = Path(root_dir)
    
    # Remove .coverage file
    coverage_file = root / '.coverage'
    if coverage_file.exists():
        print(f"Removing: {coverage_file}")
        coverage_file.unlink()
        count += 1
    
    # Remove htmlcov directory
    htmlcov_dir = root / 'htmlcov'
    if htmlcov_dir.exists():
        print(f"Removing: {htmlcov_dir}")
        shutil.rmtree(htmlcov_dir, ignore_errors=True)
        count += 1
    
    return count


def remove_build_artifacts(root_dir):
    """Remove build artifacts"""
    count = 0
    root = Path(root_dir)
    
    # Remove build directories
    for dir_name in ['build', 'dist', '*.egg-info']:
        for path in root.glob(dir_name):
            if path.is_dir():
                print(f"Removing: {path}")
                shutil.rmtree(path, ignore_errors=True)
                count += 1
    
    return count


def remove_logs(root_dir, keep_structure=True):
    """Remove log files but optionally keep directory structure"""
    count = 0
    logs_dir = Path(root_dir) / 'logs'
    
    if logs_dir.exists():
        if keep_structure:
            # Only remove .log files
            for log_file in logs_dir.glob('*.log'):
                print(f"Removing: {log_file}")
                log_file.unlink()
                count += 1
        else:
            # Remove entire logs directory
            print(f"Removing: {logs_dir}")
            shutil.rmtree(logs_dir, ignore_errors=True)
            count += 1
    
    return count


def main():
    """Main cleanup function"""
    # Get project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    print("=" * 60)
    print("ULTRATHINK Cleanup Script")
    print("=" * 60)
    print(f"Project root: {project_root}")
    print()
    
    total_removed = 0
    
    # Remove __pycache__ directories
    print("Removing __pycache__ directories...")
    count = remove_pycache(project_root)
    print(f"  Removed {count} directories")
    total_removed += count
    print()
    
    # Remove .pyc files
    print("Removing .pyc/.pyo files...")
    count = remove_pyc_files(project_root)
    print(f"  Removed {count} files")
    total_removed += count
    print()
    
    # Remove pytest cache
    print("Removing pytest cache...")
    count = remove_pytest_cache(project_root)
    print(f"  Removed {count} directories")
    total_removed += count
    print()
    
    # Remove coverage files
    print("Removing coverage files...")
    count = remove_coverage_files(project_root)
    print(f"  Removed {count} items")
    total_removed += count
    print()
    
    # Remove build artifacts
    print("Removing build artifacts...")
    count = remove_build_artifacts(project_root)
    print(f"  Removed {count} items")
    total_removed += count
    print()
    
    # Remove logs (keep structure by default)
    print("Removing log files...")
    count = remove_logs(project_root, keep_structure=True)
    print(f"  Removed {count} items")
    total_removed += count
    print()
    
    print("=" * 60)
    print(f"Cleanup complete! Removed {total_removed} items total.")
    print("=" * 60)


if __name__ == "__main__":
    main()
