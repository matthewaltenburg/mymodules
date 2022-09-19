#!/bin/bash

set -e

# Up date the system
echo "Up dating the system"
apt update -y
apt upgrade -y

# Add system packages
apt install vim -y

# Install pip packages
pip3 install torch torchvision torchaudio numpy matplotlib

# Updaate pip
pip install --upgrade pip
pip3 list --outdated --format=freeze | grep -v '^\-e' | cut -d = -f 1 | xargs -n1 pip3 install -U 

echo "All Done"
