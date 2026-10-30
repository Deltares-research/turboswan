"""MinIO / S3 helpers for cluster notebooks."""

from __future__ import annotations

import io
import json
import os
import time
from pathlib import Path

import boto3
import pandas as pd
from botocore.client import Config

_s3 = None


def s3_client():
    global _s3
    if _s3 is None:
        _s3 = boto3.client(
            "s3",
            endpoint_url=os.environ["S3_ENDPOINT"],
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
            config=Config(
                signature_version="s3v4",
                s3={"addressing_style": "path"},
            ),
            region_name="eu-west-1",
        )
    return _s3


def _object_key(root: str, suffix: str) -> str:
    return f"{root.rstrip('/')}/{suffix}"


def read_csv_key(
    bucket: str, root: str, suffix: str, **read_csv_kwargs
) -> pd.DataFrame:
    body = s3_client().get_object(Bucket=bucket, Key=_object_key(root, suffix))["Body"].read()
    return pd.read_csv(io.BytesIO(body), **read_csv_kwargs)


def read_json_prefix(bucket: str, prefix: str) -> dict[str, dict]:
    """Load all ``*.json`` objects under an S3 prefix. Keys are basenames."""
    result: dict[str, dict] = {}
    for page in s3_client().get_paginator("list_objects_v2").paginate(
        Bucket=bucket, Prefix=prefix.rstrip("/") + "/"
    ):
        for obj in page.get("Contents", []):
            name = Path(obj["Key"]).name
            if not name.endswith(".json"):
                continue
            body = s3_client().get_object(Bucket=bucket, Key=obj["Key"])["Body"].read()
            result[name] = json.loads(body)
    return result


def list_csv_keys(bucket: str, prefix: str) -> list[str]:
    """List ``*.csv`` object keys under an S3 prefix, relative to the prefix."""
    root = prefix.rstrip("/")
    keys = []
    for page in s3_client().get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=root + "/"):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".csv"):
                keys.append(obj["Key"][len(root) + 1 :])
    return sorted(keys)


def download_key(bucket: str, root: str, suffix: str, dest_path: str | Path) -> Path:
    """Download an S3 object to a local path. Returns dest_path as a Path.

    For formats pandas/json can't read straight from bytes (e.g. HDF5-based
    JLD2, NetCDF) — those libraries need a real file on disk.
    """
    dest = Path(dest_path)
    s3_client().download_file(bucket, _object_key(root, suffix), str(dest))
    return dest


def write_csv_key(df: pd.DataFrame, bucket: str, root: str, suffix: str) -> str:
    """Write a DataFrame as CSV to S3. Returns the full s3:// URI."""
    key = _object_key(root, suffix)
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    s3_client().put_object(
        Bucket=bucket,
        Key=key,
        Body=buf.getvalue(),
        ContentType="text/csv",
    )
    return f"s3://{bucket}/{key}"


class NotebookLogger:
    """Simple elapsed-time logger for Papermill notebooks."""

    def __init__(self, prefix: str) -> None:
        self.prefix = prefix
        self._t0 = time.perf_counter()

    def __call__(self, msg: str) -> None:
        elapsed = time.perf_counter() - self._t0
        print(f"[{self.prefix}] [{elapsed:6.1f}s] {msg}", flush=True)
