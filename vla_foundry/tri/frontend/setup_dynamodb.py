#!/usr/bin/env python3
"""
Setup DynamoDB tables for the VLA Foundry Dashboard and Leaderboard.

This script creates the required DynamoDB tables if they don't exist.

Run: python setup_dynamodb.py
"""

import boto3
from botocore.exceptions import ClientError

REGION = "us-west-2"

# Table definitions
TABLES = [
    {
        "name": "vla_foundry_models",
        "key_schema": [{"AttributeName": "uuid", "KeyType": "HASH"}],
        "attribute_definitions": [{"AttributeName": "uuid", "AttributeType": "S"}],
    },
    {
        "name": "vla_foundry_datasets",
        "key_schema": [{"AttributeName": "uuid", "KeyType": "HASH"}],
        "attribute_definitions": [{"AttributeName": "uuid", "AttributeType": "S"}],
    },
    {
        "name": "vla_foundry_leaderboard_runs",
        "key_schema": [{"AttributeName": "run_id", "KeyType": "HASH"}],
        "attribute_definitions": [{"AttributeName": "run_id", "AttributeType": "N"}],
    },
    {
        "name": "vla_foundry_leaderboard_evaluations",
        "key_schema": [{"AttributeName": "eval_id", "KeyType": "HASH"}],
        "attribute_definitions": [
            {"AttributeName": "eval_id", "AttributeType": "S"},
        ],
    },
    {
        "name": "vla_foundry_sample_results",
        "key_schema": [{"AttributeName": "sample_id", "KeyType": "HASH"}],
        "attribute_definitions": [
            {"AttributeName": "sample_id", "AttributeType": "S"},
            {"AttributeName": "eval_id", "AttributeType": "S"},
        ],
        "global_secondary_indexes": [
            {
                "IndexName": "eval_id-index",
                "KeySchema": [{"AttributeName": "eval_id", "KeyType": "HASH"}],
                "Projection": {"ProjectionType": "ALL"},
            }
        ],
    },
]


def create_table(dynamodb, table_config):
    """Create a DynamoDB table."""
    table_name = table_config["name"]

    try:
        # Check if table exists
        dynamodb.describe_table(TableName=table_name)
        print(f"Table {table_name} already exists.")
        return
    except ClientError as e:
        if e.response["Error"]["Code"] != "ResourceNotFoundException":
            raise

    print(f"Creating table {table_name}...")

    params = {
        "TableName": table_name,
        "KeySchema": table_config["key_schema"],
        "AttributeDefinitions": table_config["attribute_definitions"],
        "BillingMode": "PAY_PER_REQUEST",  # On-demand capacity
    }

    # Add GSIs if defined
    if "global_secondary_indexes" in table_config:
        params["GlobalSecondaryIndexes"] = table_config["global_secondary_indexes"]

    dynamodb.create_table(**params)

    # Wait for table to be created
    waiter = dynamodb.get_waiter("table_exists")
    waiter.wait(TableName=table_name)
    print(f"Table {table_name} created successfully.")


def main():
    dynamodb = boto3.client("dynamodb", region_name=REGION)

    print(f"Setting up DynamoDB tables in {REGION}...")
    print("=" * 50)

    for table_config in TABLES:
        create_table(dynamodb, table_config)

    print("=" * 50)
    print("All tables ready!")


if __name__ == "__main__":
    main()
