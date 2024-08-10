import subprocess
import sys

def install(package):
    """Install a package using pip."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def install_missing_packages(packages):
    """Install missing packages from the provided list."""
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing missing package: {package}")
            install(package)
        else:
            print(f"Package '{package}' is already installed.")

if __name__ == "__main__":
    packages_to_check = ["numpy", "pandas", "matplotlib"]  # Add your required packages here
    install_missing_packages(packages_to_check)