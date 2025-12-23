#!/usr/bin/env python
"""Setup script for nnU-Net integration"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def check_nnunet_installation():
    """Check if nnU-Net is installed and working"""
    try:
        import nnunet
        print("✓ nnU-Net Python package is installed")
        return True
    except ImportError:
        print("✗ nnU-Net Python package not found")
        return False


def check_nnunet_commands():
    """Check if nnU-Net command line tools are available"""
    commands = [
        'nnUNet_plan_and_preprocess',
        'nnUNet_train',
        'nnUNet_predict'
    ]
    
    available_commands = []
    for cmd in commands:
        try:
            result = subprocess.run([cmd, '-h'], 
                                  capture_output=True, 
                                  timeout=10)
            if result.returncode == 0:
                print(f"✓ {cmd} available")
                available_commands.append(cmd)
            else:
                print(f"✗ {cmd} not working properly")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print(f"✗ {cmd} not found")
    
    return len(available_commands) == len(commands)


def check_environment_variables():
    """Check nnU-Net environment variables"""
    required_vars = [
        'nnUNet_raw_data_base',
        'nnUNet_preprocessed', 
        'RESULTS_FOLDER'
    ]
    
    missing_vars = []
    for var in required_vars:
        if os.environ.get(var):
            print(f"✓ {var}: {os.environ[var]}")
        else:
            print(f"✗ {var}: Not set")
            missing_vars.append(var)
    
    return len(missing_vars) == 0


def install_nnunet(method='pip'):
    """Install nnU-Net"""
    print(f"Installing nnU-Net using {method}...")
    
    if method == 'pip':
        cmd = [sys.executable, '-m', 'pip', 'install', 'nnunet']
    elif method == 'source':
        print("Installing from source...")
        # Clone and install from source
        clone_cmd = ['git', 'clone', 'https://github.com/MIC-DKFZ/nnUNet.git']
        subprocess.run(clone_cmd, check=True)
        
        os.chdir('nnUNet')
        cmd = [sys.executable, '-m', 'pip', 'install', '-e', '.']
    else:
        raise ValueError(f"Unknown installation method: {method}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✓ nnU-Net installation completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ nnU-Net installation failed: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return False


def setup_environment_variables(base_dir=None):
    """Setup nnU-Net environment variables"""
    if base_dir is None:
        base_dir = Path.cwd()
    else:
        base_dir = Path(base_dir)
    
    # Create directories
    nnunet_dirs = {
        'nnUNet_raw_data_base': base_dir / 'nnUNet_raw_data',
        'nnUNet_preprocessed': base_dir / 'nnUNet_preprocessed',
        'RESULTS_FOLDER': base_dir / 'nnUNet_results'
    }
    
    print("Setting up nnU-Net directories...")
    for var_name, dir_path in nnunet_dirs.items():
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Created: {dir_path}")
        
        # Set environment variable for current session
        os.environ[var_name] = str(dir_path)
        print(f"Set {var_name}={dir_path}")
    
    # Generate shell commands for permanent setup
    print("\n" + "="*60)
    print("To make these environment variables permanent, add these lines to your shell profile:")
    print("(.bashrc, .zshrc, etc.)")
    print("="*60)
    
    for var_name, dir_path in nnunet_dirs.items():
        print(f'export {var_name}="{dir_path}"')
    
    print("="*60)
    
    return True


def create_project_directories():
    """Create necessary project directories"""
    directories = [
        'experiments/nnunet',
        'experiments/multi_unet', 
        'experiments/predictions'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created project directory: {directory}")


def verify_setup():
    """Verify complete nnU-Net setup"""
    print("\n" + "="*50)
    print("VERIFYING nnU-Net SETUP")
    print("="*50)
    
    checks = [
        ("Python package", check_nnunet_installation()),
        ("Command line tools", check_nnunet_commands()),
        ("Environment variables", check_environment_variables())
    ]
    
    all_passed = all(check[1] for check in checks)
    
    if all_passed:
        print("\n✓ nnU-Net setup is complete and working!")
        print("\nNext steps:")
        print("1. Prepare dataset: python scripts/prepare_nnunet_dataset.py")
        print("2. Train with multi-architecture: python scripts/train_multiple_unets.py --use-nnunet")
        print("3. Or train standalone: python scripts/train_nnunet_standalone.py")
    else:
        print("\n✗ nnU-Net setup incomplete. Please address the issues above.")
        failed_checks = [check[0] for check in checks if not check[1]]
        print(f"Failed checks: {', '.join(failed_checks)}")
    
    return all_passed


def main():
    parser = argparse.ArgumentParser(description='Setup nnU-Net for the project')
    parser.add_argument('--install', action='store_true',
                        help='Install nnU-Net if not already installed')
    parser.add_argument('--install-method', choices=['pip', 'source'], default='pip',
                        help='Installation method (default: pip)')
    parser.add_argument('--setup-env', action='store_true',
                        help='Setup environment variables and directories')
    parser.add_argument('--base-dir', type=str,
                        help='Base directory for nnU-Net data (default: current directory)')
    parser.add_argument('--verify-only', action='store_true',
                        help='Only verify existing setup')
    
    args = parser.parse_args()
    
    print("nnU-Net Setup for Pulmonary Artery Segmentation")
    print("=" * 50)
    
    if args.verify_only:
        verify_setup()
        return
    
    # Check current status
    nnunet_installed = check_nnunet_installation()
    env_vars_set = check_environment_variables()
    
    # Install nnU-Net if requested or if not installed
    if args.install or not nnunet_installed:
        if not install_nnunet(args.install_method):
            print("Installation failed. Please check error messages above.")
            sys.exit(1)
    
    # Setup environment if requested or if not set
    if args.setup_env or not env_vars_set:
        setup_environment_variables(args.base_dir)
    
    # Create project directories
    create_project_directories()
    
    # Final verification
    print("\n" + "="*50)
    print("FINAL VERIFICATION")
    print("="*50)
    success = verify_setup()
    
    if success:
        print("\n🎉 nnU-Net is ready to use!")
        
        # Show example commands
        print("\nExample usage:")
        print("# Prepare dataset")
        print("python scripts/prepare_nnunet_dataset.py --dataset DATASET/Parse_dataset")
        print()
        print("# Train with all architectures including nnU-Net")
        print("python scripts/train_multiple_unets.py --use-nnunet")
        print()
        print("# Train only nnU-Net")
        print("python scripts/train_nnunet_standalone.py --task-id 501")
        print()
        print("# Generate predictions")
        print("python scripts/generate_nnunet_predictions.py --dataset DATASET/Parse_dataset --output-dir predictions/nnunet")
        
    else:
        print("\n❌ Setup incomplete. Please resolve the issues and try again.")
        sys.exit(1)


if __name__ == '__main__':
    main()