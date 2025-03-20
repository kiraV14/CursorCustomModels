#!/usr/bin/env python3
import subprocess
import datetime
import os
import sys
import importlib.util

def ensure_dependencies():
    """Check and install required dependencies if they're missing."""
    required_packages = []  # We're only using standard library modules
    
    # This function is included for future extensibility
    # If we add non-standard dependencies later, this will handle them
    for package in required_packages:
        if importlib.util.find_spec(package) is None:
            print(f"Installing required package: {package}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"Successfully installed {package}")

def run_command(command):
    """Run a shell command and return its output."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {command}")
        print(f"Error message: {e.stderr}")
        sys.exit(1)

def check_git_installed():
    """Check if git is installed on the system."""
    try:
        run_command("git --version")
        return True
    except:
        print("Error: Git is not installed or not in PATH.")
        print("Please install Git before running this script.")
        sys.exit(1)

def git_operations():
    # Check if git is installed
    check_git_installed()
    
    # Check if we're in a git repository
    if not os.path.exists('.git'):
        print("Error: Not a git repository. Please run this script from a git repository root.")
        sys.exit(1)
    
    # Get current timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Git add all changes
    print("Adding all changes...")
    run_command("git add .")
    
    # Check if there are changes to commit
    status = run_command("git status --porcelain")
    if not status:
        print("No changes to commit. Repository is clean.")
        return
    
    # Git commit with timestamp
    commit_message = f"Automatic commit at {timestamp}"
    print(f"Committing with message: '{commit_message}'")
    run_command(f'git commit -m "{commit_message}"')
    
    # Get current branch name
    branch = run_command("git rev-parse --abbrev-ref HEAD")
    
    # Check if remote exists
    remote_exists = False
    try:
        run_command("git remote get-url origin")
        remote_exists = True
    except:
        print("Warning: No remote named 'origin' found. Skipping push operation.")
    
    # Git push to remote if it exists
    if remote_exists:
        print(f"Pushing to origin/{branch}...")
        try:
            run_command(f"git push -u origin {branch}")
            print("Git operations completed successfully!")
        except:
            print(f"Failed to push to origin/{branch}. You may need to set up the remote branch.")
    else:
        print("Git commit completed successfully! (Push was skipped)")

if __name__ == "__main__":
    print("=== Git Auto-Commit and Push Tool ===")
    
    # Ensure all dependencies are installed
    ensure_dependencies()
    
    # Run git operations
    git_operations() 