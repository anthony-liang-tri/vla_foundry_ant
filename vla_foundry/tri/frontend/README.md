# VLA Foundry Dashboard & Leaderboard

A web dashboard for monitoring VLA Foundry training runs and evaluation results, backed by DynamoDB.

## Usage

The dashboard and leaderboard are hosted on an EC2 instance at:
[http://10.242.10.206:8080](http://10.242.10.206:8080)

And the leaderboard at:
[http://10.242.10.206:8080/leaderboard](http://10.242.10.206:8080/leaderboard)


To upload evaluation results, you can use the following commands:
```bash
# Upload evaluation results
python leaderboard_cli.py upload task_list.txt --campaign-name my_campaign

# List all entries
python leaderboard_cli.py list

# Delete entries
python leaderboard_cli.py delete --campaign-name my_campaign --force
```

To update the frontend, get the jean_vla_foundry.pem file and place it in ~/.ssh/.
Make your changes, and then use the following command to deploy the changes:
```bash
./deploy_ec2.sh
```

## Architecture Overview

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Training Jobs  │────▶│                 │◀────│   EC2 Server    │
│  (db_logger.py) │     │    DynamoDB     │     │   (read-only)   │
└─────────────────┘     │    (Tables)     │     └─────────────────┘
                        │                 │              │
┌─────────────────┐     │                 │              ▼
│  CLI Upload     │────▶│                 │     ┌─────────────────┐
│  (leaderboard_  │     └─────────────────┘     │  Web Dashboard  │
│   cli.py)       │                             │  (HTML/JS)      │
└─────────────────┘                             └─────────────────┘
```

## Components

- **DynamoDB Tables**: Store training run metadata and evaluation results
- **EC2 Server**: Hosts read-only FastAPI backend that queries DynamoDB for the web frontend
- **Web Frontend**: HTML/JS dashboard for visualization
- **db_logger.py**: Python module that logs training progress to DynamoDB
- **leaderboard_cli.py**: CLI tool that uploads evaluation results directly to DynamoDB (using boto3)

---

## Step 1: Create DynamoDB Tables

### Option A: Using the setup script

```bash
cd vla_foundry/tri/frontend
python setup_dynamodb.py
```

### Option B: Manual creation via AWS Console or CLI

Create the following tables in `us-west-2` (or your preferred region):

#### Table 1: `vla_foundry_models`
- **Partition Key**: `uuid` (String)
- **Purpose**: Stores training run metadata

```bash
aws dynamodb create-table \
    --table-name vla_foundry_models \
    --attribute-definitions AttributeName=uuid,AttributeType=S \
    --key-schema AttributeName=uuid,KeyType=HASH \
    --billing-mode PAY_PER_REQUEST \
    --region us-west-2
```

#### Table 2: `vla_foundry_datasets`
- **Partition Key**: `uuid` (String)
- **Purpose**: Stores dataset preprocessing metadata

```bash
aws dynamodb create-table \
    --table-name vla_foundry_datasets \
    --attribute-definitions AttributeName=uuid,AttributeType=S \
    --key-schema AttributeName=uuid,KeyType=HASH \
    --billing-mode PAY_PER_REQUEST \
    --region us-west-2
```

#### Table 3: `vla_foundry_leaderboard_runs`
- **Partition Key**: `run_id` (Number)
- **Purpose**: Stores evaluation run metadata

```bash
aws dynamodb create-table \
    --table-name vla_foundry_leaderboard_runs \
    --attribute-definitions AttributeName=run_id,AttributeType=N \
    --key-schema AttributeName=run_id,KeyType=HASH \
    --billing-mode PAY_PER_REQUEST \
    --region us-west-2
```

#### Table 4: `vla_foundry_leaderboard_evaluations`
- **Partition Key**: `eval_id` (String)
- **Purpose**: Stores individual evaluation results

```bash
aws dynamodb create-table \
    --table-name vla_foundry_leaderboard_evaluations \
    --attribute-definitions AttributeName=eval_id,AttributeType=S \
    --key-schema AttributeName=eval_id,KeyType=HASH \
    --billing-mode PAY_PER_REQUEST \
    --region us-west-2
```

---

## Step 2: Set Up EC2 Instance

### 2.1 Launch an EC2 Instance

- **AMI**: Amazon Linux 2023 or Amazon Linux 2
- **Instance Type**: `t3.micro` is sufficient for light usage
- **Security Group**: Allow inbound HTTP (port 8080) from your network
- **IAM Role**: Attach a role with DynamoDB access

Example IAM policy for the EC2 role (read-only):
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "dynamodb:Scan",
                "dynamodb:Query",
                "dynamodb:GetItem",
                "dynamodb:BatchGetItem"
            ],
            "Resource": [
                "arn:aws:dynamodb:us-west-2:*:table/vla_foundry_*"
            ]
        }
    ]
}
```

Note: The CLI (`leaderboard_cli.py`) writes directly to DynamoDB using your local AWS credentials,
so you need write permissions on your local machine but the EC2 server only needs read access.

### 2.2 Initial Server Setup

SSH into your EC2 instance and run:

```bash
# Install dependencies
sudo yum update -y
sudo yum install -y python3 python3-pip

# Install Python packages
pip3 install --user fastapi uvicorn boto3

# Create dashboard directory
mkdir -p ~/dashboard
mkdir -p ~/dashboard/leaderboard
```

### 2.3 Create systemd Service

```bash
sudo tee /etc/systemd/system/dashboard.service << 'EOF'
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
EOF

sudo systemctl daemon-reload
sudo systemctl enable dashboard
```

---

## Step 3: Deploy the Dashboard

### 3.1 Configure Deployment Script

Edit `deploy_ec2.sh` with your EC2 details:

```bash
EC2_HOST="ec2-user@YOUR_EC2_IP"
SSH_KEY="~/.ssh/your_key.pem"
```

### 3.2 Deploy

```bash
cd vla_foundry/tri/frontend
./deploy_ec2.sh
```

This copies the following files to EC2:
- `server.py` - FastAPI backend
- `index.html` - Main dashboard
- `leaderboard/` - Leaderboard frontend files

### 3.3 Verify

Open in browser:
- Dashboard: `http://YOUR_EC2_IP:8080`
- Leaderboard: `http://YOUR_EC2_IP:8080/leaderboard`

---

## Step 4: Integrate with Training

Training runs automatically log to DynamoDB via `vla_foundry/db_logger.py`.

### How it works:

1. **On training start**: `log_job_start()` creates a record with status="running"
2. **After each checkpoint**: `log_checkpoint()` updates progress
3. **On completion**: `log_completion()` sets status="completed"
4. **On crash**: `atexit` handler calls `log_crash()` sets status="crashed"

### Fields logged:

| Field | Description |
|-------|-------------|
| `uuid` | Unique run identifier |
| `status` | running, completed, or crashed |
| `datetime` | Last update timestamp |
| `datetime_job_started` | When training started |
| `checkpoint_number_current` | Current checkpoint |
| `checkpoint_number_total` | Total checkpoints |
| `sample_number_current` | Samples processed |
| `sample_number_total` | Total samples target |
| `cfg` | Full training config (YAML) |
| `wandb_url` | Link to W&B run |
| `git_commit_hash` | Code version |
| `git_branch` | Git branch |
| `created_by` | Username |
| `hostname` | Training host |

---

## Step 5: Upload Evaluation Results

Use `leaderboard_cli.py` to upload evaluation results directly to DynamoDB.
The CLI uses boto3 to write directly to the database (no EC2 server needed for writes).

### Upload Results

```bash
cd vla_foundry/tri/frontend

# Upload from a task list file
python leaderboard_cli.py upload task_list.txt --campaign-name my_campaign

# Upload multiple files
python leaderboard_cli.py upload file1.txt file2.txt --campaign-name my_campaign

# Upload without downloading configs (faster)
python leaderboard_cli.py upload task_list.txt --campaign-name my_campaign --skip-s3 --no-link

# List all leaderboard entries
python leaderboard_cli.py list

# Delete entries by filter
python leaderboard_cli.py delete --campaign-name my_campaign --force
python leaderboard_cli.py delete --ablation my_ablation
```

### Task List Format

Task list files should contain tuples of (scenario_name, task_name, s3_path):

```python
("MyTask_Ablation1", "BimanualPutFruitInBowl", "s3://bucket/path/to/checkpoint"),
("MyTask_Ablation2", "BimanualPutFruitInBowl", "s3://bucket/path/to/checkpoint2"),
```

---

## File Structure

```
vla_foundry/tri/frontend/
├── README.md              # This file
├── server.py              # FastAPI server (dashboard + leaderboard APIs)
├── index.html             # Main dashboard frontend
├── deploy_ec2.sh          # Deployment script
├── setup_dynamodb.py      # DynamoDB table creation
├── leaderboard_cli.py     # CLI for uploading evaluation results
└── leaderboard/
    ├── index.html         # Leaderboard frontend
    ├── style.css          # Leaderboard styles
    └── app.js             # Leaderboard JavaScript
```

---

## API Endpoints

### Dashboard APIs

| Endpoint | Description |
|----------|-------------|
| `GET /` | Main dashboard HTML |
| `GET /api/models` | List all training runs |
| `GET /api/datasets` | List all datasets |

### Leaderboard APIs

| Endpoint | Description |
|----------|-------------|
| `GET /leaderboard` | Leaderboard HTML |
| `GET /api/leaderboard` | All evaluations with run metadata |
| `GET /api/campaigns` | List of campaign names |
| `GET /api/tasks` | List of task names |
| `GET /api/ablations` | List of ablation names |
| `GET /api/config/{run_id}` | Get config for a run |
| `GET /api/stats` | Summary statistics |

Note: Write operations are done directly via `leaderboard_cli.py` which writes to DynamoDB using boto3.

---

## Customization

### Changing the AWS Region

1. Update `REGION` in:
   - `server.py`
   - `setup_dynamodb.py`
   - `db_logger.py`

2. Update IAM policy ARNs

### Changing Table Names

Update the `*_TABLE` constants in:
- `server.py`
- `setup_dynamodb.py`
- `db_logger.py`

---

## Troubleshooting

### Dashboard shows no data

1. Check EC2 has IAM role with DynamoDB access
2. Verify tables exist: `aws dynamodb list-tables --region us-west-2`
3. Check server logs: `sudo journalctl -u dashboard -f`

### Training runs not appearing

1. Verify `db_logger` is enabled in training config
2. Check training host has AWS credentials
3. Look for "Logged training job start to DynamoDB" in training logs

### Runs stuck as "Running"

- Old runs without `status` field: Dashboard uses 30-min stale heuristic
- New runs: Check if training crashed before `atexit` handler ran (e.g., SIGKILL)

### Leaderboard HTTP 500

1. Check EC2 IAM role includes leaderboard tables
2. Check server logs: `sudo journalctl -u dashboard -f`
