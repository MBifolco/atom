#!/bin/bash
# Upgrade ROCm 6.3 → 7.1 for JAX GPU Support
# Run with: sudo bash upgrade_rocm_to_7.sh

set -e  # Exit on error

echo "=========================================="
echo "  ROCm 6.3 → 7.1 Upgrade Script"
echo "=========================================="
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
   echo "❌ Please run with sudo: sudo bash upgrade_rocm_to_7.sh"
   exit 1
fi

echo "📋 Current ROCm version:"
cat /opt/rocm/.info/version 2>/dev/null || echo "Not found"
echo ""

echo "⚠️  This will:"
echo "  1. Uninstall ROCm 6.3"
echo "  2. Install ROCm 7.1"
echo "  3. May require a reboot"
echo ""
read -p "Continue? (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo "=========================================="
echo "Step 1: Uninstalling ROCm 6.3..."
echo "=========================================="

# Uninstall all ROCm packages
apt autoremove --purge -y rocm || echo "rocm meta-package not found"
apt autoremove --purge -y 'rocm-*' || echo "No rocm-* packages found"

# Remove repository list
rm -f /etc/apt/sources.list.d/rocm.list
rm -f /etc/apt/sources.list.d/amdgpu.list

# Clean apt cache
apt clean all

echo "✅ ROCm 6.3 uninstalled"
echo ""

echo "=========================================="
echo "Step 2: Adding ROCm 7.1 repository..."
echo "=========================================="

# Ensure keyrings directory exists
mkdir -p /etc/apt/keyrings

# Download ROCm GPG key if not present
if [ ! -f /etc/apt/keyrings/rocm.gpg ]; then
    echo "Downloading ROCm GPG key..."
    wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor -o /etc/apt/keyrings/rocm.gpg
fi

# Add ROCm 7.1 repository
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/7.1 jammy main" > /etc/apt/sources.list.d/rocm.list

# Add AMDGPU repository (needed for drivers)
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/amdgpu/latest/ubuntu jammy main" > /etc/apt/sources.list.d/amdgpu.list

echo "✅ ROCm 7.1 repository added"
echo ""

echo "=========================================="
echo "Step 3: Updating package lists..."
echo "=========================================="

apt update

echo "✅ Package lists updated"
echo ""

echo "=========================================="
echo "Step 4: Installing ROCm 7.1..."
echo "=========================================="
echo "⏳ This may take 5-10 minutes..."
echo ""

# Install ROCm 7.1
apt install -y rocm

echo "✅ ROCm 7.1 installed"
echo ""

echo "=========================================="
echo "Step 5: Verifying installation..."
echo "=========================================="

# Verify ROCm installation
echo "ROCm version:"
cat /opt/rocm/.info/version

echo ""
echo "GPU detected:"
/opt/rocm/bin/rocm-smi --showproductname

echo ""
echo "=========================================="
echo "  ✅ ROCm 7.1 Upgrade Complete!"
echo "=========================================="
echo ""
echo "📝 Next steps:"
echo "  1. Test JAX GPU detection:"
echo "     ~/.pyenv/versions/3.11.10/bin/python3 -c \"import jax; print(jax.devices())\""
echo ""
echo "  2. If it shows [RocmDevice(id=0)], GPU is working!"
echo ""
echo "⚠️  If you have issues, a reboot may be required:"
echo "     sudo reboot"
echo ""
