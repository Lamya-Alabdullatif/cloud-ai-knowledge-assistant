"""Thin, well-typed wrapper around boto3's S3 client.

All access is private: the bucket is expected to block public access (see
README -> AWS Setup) and this app never sets a public-read ACL. Objects are
addressed by a generated key of the form `documents/{document_id}/{filename}`
so multiple documents can never collide.
"""
from __future__ import annotations

import logging
from typing import Optional

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import BotoCoreError, ClientError

from app.config import Settings
from app.exceptions import StorageError

logger = logging.getLogger("app")


class S3Client:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._bucket = settings.S3_BUCKET_NAME
        self._client = boto3.client(
            "s3",
            region_name=settings.AWS_REGION,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID or None,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY or None,
            config=BotoConfig(retries={"max_attempts": 3, "mode": "standard"}),
        )

    def object_key(self, document_id: str, filename: str) -> str:
        return f"documents/{document_id}/{filename}"

    def upload_bytes(self, key: str, data: bytes, content_type: str = "application/pdf") -> None:
        try:
            self._client.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=data,
                ContentType=content_type,
                ServerSideEncryption="AES256",
            )
            logger.info("Uploaded object to S3: s3://%s/%s (%d bytes)", self._bucket, key, len(data))
        except (BotoCoreError, ClientError) as exc:
            logger.error("S3 upload failed for key=%s: %s", key, exc)
            raise StorageError(f"Failed to upload file to S3: {exc}") from exc

    def get_bytes(self, key: str) -> bytes:
        try:
            response = self._client.get_object(Bucket=self._bucket, Key=key)
            return response["Body"].read()
        except (BotoCoreError, ClientError) as exc:
            logger.error("S3 download failed for key=%s: %s", key, exc)
            raise StorageError(f"Failed to download file from S3: {exc}") from exc

    def delete_object(self, key: str) -> None:
        try:
            self._client.delete_object(Bucket=self._bucket, Key=key)
            logger.info("Deleted object from S3: s3://%s/%s", self._bucket, key)
        except (BotoCoreError, ClientError) as exc:
            logger.error("S3 delete failed for key=%s: %s", key, exc)
            raise StorageError(f"Failed to delete file from S3: {exc}") from exc

    def object_exists(self, key: str) -> bool:
        try:
            self._client.head_object(Bucket=self._bucket, Key=key)
            return True
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in ("404", "NoSuchKey"):
                return False
            logger.error("S3 head_object failed for key=%s: %s", key, exc)
            raise StorageError(f"Failed to check file existence in S3: {exc}") from exc


_s3_singleton: Optional[S3Client] = None


def get_s3_client(settings: Settings) -> S3Client:
    global _s3_singleton
    if _s3_singleton is None:
        _s3_singleton = S3Client(settings)
    return _s3_singleton
