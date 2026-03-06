# data/registry/hub_pusher.py

from huggingface_hub import HfApi, upload_folder, upload_file
from pathlib import Path

import logging

logger = logging.getLogger(__name__)

class HFHubPusher:
    """
    Pushes tokenizer and shard files to HuggingFace Hub.
    Acts as the artifact registry for all pipeline outputs.
    """

    def __init__(self, repo_id: str, token: str):
        self.repo_id = repo_id
        self.token = token
        self.api = HfApi()

    @property
    def repo_url(self) -> str:
        return f"https://huggingface.co/datasets/{self.repo_id}"

    def create_repo(self, repo_type: str = "dataset") -> None:
        """
        Creates a new repository on HF Hub.
        """
        logger.info(f"Creating repository: {self.repo_id}")
        self.api.create_repo(
            repo_id=self.repo_id,
            token=self.token,
            repo_type=repo_type,
            exist_ok=True,
            private=False
        )

    def push_tokenizer(self, tokenizer_path: str) -> str:
        """
        Uploads tokenizer.json to HF Hub.
        Returns the URL of the uploaded file.
        """
        logger.info(f"Pushing tokenizer from {tokenizer_path} to {self.repo_id}")
        commit_info = upload_file(
            path_or_fileobj=tokenizer_path,
            path_in_repo="tokenizer/tokenizer.json",
            repo_id=self.repo_id,
            token=self.token,
            repo_type="dataset"
        )
        url = f"{self.repo_url}/blob/main/tokenizer/tokenizer.json"
        logger.info(f"Tokenizer pushed: {url}")
        return url

    def push_shards(self, shards_dir: str) -> None:
        """Uploads all .bin shards from a directory."""
        logger.info(f"Pushing shards to: {self.repo_id}")
        upload_folder(
            folder_path=shards_dir,
            repo_id=self.repo_id,
            token=self.token,
            repo_type="dataset",
            path_in_repo="shards/",
            ignore_patterns=["*.py", "*.yaml"]
        )
        logger.info(f"Shards pushed to: {self.repo_url}")

    def push_file(self, local_path: str, repo_path: str) -> str:
        """
        Uploads a single file to HF Hub.
        Returns the URL of the uploaded file.
        """
        logger.info(f"Pushing file from {local_path} to {self.repo_id}")
        commit_info = upload_file(
            path_or_fileobj=local_path,
            path_in_repo=repo_path,
            repo_id=self.repo_id,
            token=self.token,
            repo_type="dataset"
        )
        url = f"{self.repo_url}/blob/main/{repo_path}"
        logger.info(f"File pushed: {url}")
        return url