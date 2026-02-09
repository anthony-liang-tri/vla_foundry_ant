#!/bin/bash
# Deploy VLA Foundry Dashboard to EC2
#
# Prerequisites:
# - An EC2 instance role with DynamoDB read access
# - Security group allowing inbound port 80 (HTTP)
#
# Usage:
#   ./deploy_ec2.sh [--setup]    # Deploy (with optional first-time setup)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configuration - edit these for your setup
EC2_HOST="ec2-user@10.242.10.206"
SSH_KEY="~/.ssh/jean_vla_foundry.pem"
REMOTE_DIR="/home/ec2-user/dashboard"

SSH_CMD="ssh -i $SSH_KEY $EC2_HOST"
SCP_CMD="scp -i $SSH_KEY"

# First-time setup
if [ "$1" == "--setup" ]; then
    echo "=== Setting up EC2 instance ==="
    ssh -i $SSH_KEY $EC2_HOST bash -s << 'EOF'
        # Install Python and dependencies
        sudo yum update -y
        sudo yum install -y python3 python3-pip

        # Install Python packages
        pip3 install --user fastapi uvicorn boto3 numpy scipy

        # Create directory
        mkdir -p ~/dashboard

        # Create systemd service
        sudo tee /etc/systemd/system/dashboard.service << 'SERVICE'
[Unit]
Description=VLA Foundry Dashboard
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/home/ec2-user/dashboard
ExecStart=/usr/bin/python3 /home/ec2-user/dashboard/server.py --host 0.0.0.0 --port 8080
Restart=always
RestartSec=5
Environment=PATH=/home/ec2-user/.local/bin:/usr/bin

[Install]
WantedBy=multi-user.target
SERVICE

        sudo systemctl daemon-reload
        sudo systemctl enable dashboard
        echo "Setup complete!"
EOF
fi

# Deploy files
echo "=== Deploying dashboard files ==="
$SCP_CMD "$SCRIPT_DIR/server.py" "$EC2_HOST:$REMOTE_DIR/"
$SCP_CMD "$SCRIPT_DIR/index.html" "$EC2_HOST:$REMOTE_DIR/"

# Deploy leaderboard files
echo "=== Deploying leaderboard files ==="
$SSH_CMD "mkdir -p $REMOTE_DIR/leaderboard"
$SCP_CMD "$SCRIPT_DIR/leaderboard/index.html" "$EC2_HOST:$REMOTE_DIR/leaderboard/"
$SCP_CMD "$SCRIPT_DIR/leaderboard/style.css" "$EC2_HOST:$REMOTE_DIR/leaderboard/"
$SCP_CMD "$SCRIPT_DIR/leaderboard/app.js" "$EC2_HOST:$REMOTE_DIR/leaderboard/"

# Restart service
echo "=== Restarting service ==="
$SSH_CMD "sudo systemctl restart dashboard"

# Get status
echo ""
echo "=== Deployment complete ==="
$SSH_CMD "sudo systemctl status dashboard --no-pager" || true

# Extract IP from host
IP="${EC2_HOST#*@}"

echo ""
echo "Dashboard should be available at: http://$IP:8080"
echo "Leaderboard should be available at: http://$IP:8080/leaderboard"
