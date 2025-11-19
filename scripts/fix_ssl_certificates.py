#!/usr/bin/env python3
"""Fix SSL certificate issues for Confident AI uploads on macOS."""

import os
import sys
import subprocess
from pathlib import Path

def install_certifi():
    """Install or upgrade certifi package."""
    print("Installing/upgrading certifi package...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "certifi"])
        print("✅ certifi installed/upgraded successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install certifi: {e}")
        return False

def find_python_installer():
    """Find the Install Certificates.command script."""
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    possible_paths = [
        f"/Applications/Python {python_version}/Install Certificates.command",
        f"/Applications/Python {sys.version_info.major}.{sys.version_info.minor}/Install Certificates.command",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    # Try to find Python installation directory
    python_dir = Path(sys.executable).parent.parent
    installer = python_dir / "Install Certificates.command"
    if installer.exists():
        return str(installer)
    
    return None

def run_certificate_installer():
    """Run the macOS certificate installer if available."""
    installer_path = find_python_installer()
    
    if installer_path:
        print(f"Found certificate installer at: {installer_path}")
        print("Running certificate installer...")
        try:
            subprocess.check_call(["bash", installer_path])
            print("✅ Certificate installer completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Certificate installer failed: {e}")
            return False
    else:
        print("⚠️  Could not find Install Certificates.command script")
        print("   This is normal if Python was installed via Homebrew or pyenv")
        return False

def set_ssl_cert_file():
    """Set SSL_CERT_FILE environment variable."""
    try:
        import certifi
        cert_file = certifi.where()
        os.environ["SSL_CERT_FILE"] = cert_file
        print(f"✅ Set SSL_CERT_FILE to: {cert_file}")
        print(f"   Add this to your .env file: SSL_CERT_FILE={cert_file}")
        return True
    except ImportError:
        print("❌ certifi not installed")
        return False

def main():
    """Main function to fix SSL certificates."""
    print("=" * 60)
    print("SSL Certificate Fix for Confident AI")
    print("=" * 60)
    print()
    
    # Step 1: Install certifi
    if not install_certifi():
        print("\n⚠️  Continuing anyway...")
    
    print()
    
    # Step 2: Run macOS certificate installer (if available)
    installer_success = run_certificate_installer()
    
    print()
    
    # Step 3: Set SSL_CERT_FILE
    if set_ssl_cert_file():
        print()
        print("=" * 60)
        print("✅ SSL certificates configured!")
        print("=" * 60)
        print()
        print("Next steps:")
        print("1. Add SSL_CERT_FILE to your .env file (see above)")
        print("2. Set CONFIDENT_API_KEY in your .env file")
        print("3. Restart your terminal/IDE")
        print("4. Run evaluation again")
    else:
        print()
        print("=" * 60)
        print("⚠️  Manual configuration may be required")
        print("=" * 60)
        print()
        print("If SSL errors persist, try:")
        print("1. Install certificates manually:")
        print("   /Applications/Python\\ 3.12/Install\\ Certificates.command")
        print("2. Or set SSL_CERT_FILE manually in .env:")
        print("   SSL_CERT_FILE=$(python3 -m certifi)")

if __name__ == "__main__":
    main()

