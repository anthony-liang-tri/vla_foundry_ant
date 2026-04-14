#!/bin/bash
# Deploy VLA Foundry Docs to EC2
#
# Prerequisites:
# - An EC2 instance with a security group allowing inbound port 8081 (HTTP)
# - mkdocs + mkdocs-material installed locally (pip install -r docs/requirements.txt)
#
# Usage:
#   ./docs/deploy_ec2.sh [--setup]    # Deploy (with optional first-time setup)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Configuration - edit these for your setup
EC2_HOST="ec2-user@10.242.10.206"
SSH_KEY="~/.ssh/vla_foundry.pem"
REMOTE_DIR="/home/ec2-user/docs-site"

SSH_CMD="ssh -i $SSH_KEY $EC2_HOST"
SCP_CMD="scp -i $SSH_KEY"

# First-time setup
if [ "$1" == "--setup" ]; then
    echo "=== Setting up EC2 instance ==="
    ssh -i $SSH_KEY $EC2_HOST bash -s << 'EOF'
        # Install Python
        sudo yum update -y
        sudo yum install -y python3

        # Create directory
        mkdir -p ~/docs-site

        # Create systemd service - simple Python HTTP server
        sudo tee /etc/systemd/system/docs.service << 'SERVICE'
[Unit]
Description=VLA Foundry Documentation
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/home/ec2-user/docs-site
ExecStart=/usr/bin/python3 -m http.server 8081
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
SERVICE

        sudo systemctl daemon-reload
        sudo systemctl enable docs
        echo "Setup complete!"
EOF
fi

# Build docs locally
echo "=== Building docs ==="
cd "$REPO_DIR"
mkdocs build --strict
echo "Built $(find site -name '*.html' | wc -l) pages"

# Deploy site/ directory
echo "=== Deploying docs to EC2 ==="
# rsync is faster for incremental updates; fall back to scp if unavailable
if command -v rsync &> /dev/null; then
    rsync -avz --delete -e "ssh -i $SSH_KEY" "$REPO_DIR/site/" "$EC2_HOST:$REMOTE_DIR/"
else
    $SSH_CMD "rm -rf $REMOTE_DIR/*"
    $SCP_CMD -r "$REPO_DIR/site/"* "$EC2_HOST:$REMOTE_DIR/"
fi

# Restart service
echo "=== Restarting service ==="
$SSH_CMD "sudo systemctl restart docs"

# Get status
echo ""
echo "=== Deployment complete ==="
$SSH_CMD "sudo systemctl status docs --no-pager" || true

# Extract IP from host
IP="${EC2_HOST#*@}"

echo ""
echo "Docs available at: http://$IP:8081"
