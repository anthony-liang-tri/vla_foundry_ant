import argparse
import contextlib
import logging
import os
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse

import boto3
import ray
import requests
from botocore.config import Config
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

from vla_foundry.file_utils import file_exists, parse_s3_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DirectoryListingParser(HTMLParser):
    """Parse href values from <a> tags in Apache/nginx directory listings."""

    def __init__(self):
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            for name, value in attrs:
                if name == "href" and value is not None:
                    self.links.append(value)


def crawl_http_directory(
    base_url: str,
    session: requests.Session,
    relative_prefix: str = "",
    max_depth: int = 5,
) -> list[tuple[str, str]]:
    """
    Recursively crawl an HTTP directory listing and return downloadable file URLs.

    Returns:
        List of (download_url, relative_path) tuples.
    """
    if max_depth <= 0:
        return []

    # Ensure base_url ends with /
    if not base_url.endswith("/"):
        base_url += "/"

    response = session.get(base_url, timeout=30)
    response.raise_for_status()

    parser = DirectoryListingParser()
    parser.feed(response.text)

    files: list[tuple[str, str]] = []
    base_parsed = urlparse(base_url)

    for link in parser.links:
        # Skip parent directory, query strings, and absolute/external URLs
        if link in ("../", "..") or "?" in link:
            continue
        link_parsed = urlparse(link)
        if link_parsed.scheme and link_parsed.netloc and link_parsed.netloc != base_parsed.netloc:
            continue
        # Skip absolute paths that point outside the base
        if link.startswith("/"):
            continue

        full_url = urljoin(base_url, link)
        rel_path = relative_prefix + link

        if link.endswith("/"):
            # Recurse into subdirectory
            files.extend(crawl_http_directory(full_url, session, rel_path, max_depth - 1))
        else:
            files.append((full_url, rel_path))

    return files


@ray.remote
def download_and_upload_file(*args, **kwargs):
    # This makes it usable outside of Ray setups
    return _download_and_upload_file(*args, **kwargs)


def _download_and_upload_file(
    url: str,
    relative_key: str,
    s3_bucket: str,
    s3_output_dir: str,
    local_temp_dir: str,
    max_retries: int,
    backoff_factor: float,
    download_timeout: int,
) -> tuple[bool, dict | None]:
    """
    Download a file from HTTP and upload to S3.
    Returns (success, error_info).
    """
    # Check if already uploaded to S3
    s3_key = f"{s3_output_dir}/{relative_key}"
    if file_exists(f"s3://{s3_bucket}/{s3_key}"):
        print(f"Skipping {relative_key} (already exists on S3)")
        return True, None

    # Setup local paths
    temp_dir = Path(local_temp_dir)
    temp_path = temp_dir / relative_key
    temp_path.parent.mkdir(parents=True, exist_ok=True)

    # Setup retry configuration
    retry_config = Retry(
        total=max_retries,
        connect=max_retries,
        read=max_retries,
        status=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods={"GET"},
        respect_retry_after_header=True,
    )
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry_config)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    try:
        # Download file
        with session.get(url, stream=True, timeout=download_timeout) as response:
            response.raise_for_status()
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

        print(f"Downloaded {relative_key}")

        # Upload to S3
        boto3_config = Config(max_pool_connections=50, retries={"mode": "adaptive", "max_attempts": 3})
        s3_client = boto3.client("s3", config=boto3_config)
        s3_client.upload_file(str(temp_path), s3_bucket, s3_key)
        print(f"Uploaded s3://{s3_bucket}/{s3_key}")

        # Clean up local file
        with contextlib.suppress(OSError):
            os.remove(temp_path)

        return True, None

    except requests.HTTPError as http_error:
        error_info = {"url": url, "filename": relative_key, "error": str(http_error)}
        print(f"HTTP error downloading {url}: {http_error}")
        return False, error_info
    except requests.RequestException as req_error:
        error_info = {"url": url, "filename": relative_key, "error": str(req_error)}
        print(f"Network error downloading {url}: {req_error}")
        return False, error_info
    except Exception as unknown_error:
        error_info = {"url": url, "filename": relative_key, "error": str(unknown_error)}
        print(f"Unexpected error downloading {url}: {unknown_error}")
        return False, error_info


class HttpDatasetDownloader:
    def __init__(
        self,
        source_url: str,
        s3_bucket: str,
        s3_output_dir: str,
        local_temp_dir: str,
        max_retries: int = 10,
        backoff_factor: float = 3.0,
        download_timeout: int = 300,
    ):
        self.source_url = source_url
        self.s3_bucket = s3_bucket
        self.s3_output_dir = s3_output_dir
        self.local_temp_dir = local_temp_dir
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.download_timeout = download_timeout
        self.temp_dir = Path(local_temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.failed_files: list[dict] = []

    def discover_files(self) -> list[tuple[str, str]]:
        """Crawl the HTTP directory listing and return (url, relative_path) pairs."""
        session = requests.Session()
        files = crawl_http_directory(self.source_url, session)
        logger.info(f"Discovered {len(files)} files at {self.source_url}")
        return files

    def download_dataset(self) -> None:
        """Download all files in parallel using Ray and upload to S3."""
        start_time = datetime.now()
        logger.info(f"Starting download from {self.source_url}")
        logger.info(f"Target: s3://{self.s3_bucket}/{self.s3_output_dir}")

        files = self.discover_files()
        if not files:
            logger.warning("No files discovered. Nothing to download.")
            return

        # Submit Ray tasks
        futures = []
        for url, relative_key in files:
            future = download_and_upload_file.remote(
                url=url,
                relative_key=relative_key,
                s3_bucket=self.s3_bucket,
                s3_output_dir=self.s3_output_dir,
                local_temp_dir=self.local_temp_dir,
                max_retries=self.max_retries,
                backoff_factor=self.backoff_factor,
                download_timeout=self.download_timeout,
            )
            futures.append(future)

        # Track progress with ray.wait + tqdm
        remaining = list(futures)
        successful = 0
        with tqdm(total=len(futures), desc="Downloading") as pbar:
            while remaining:
                done, remaining = ray.wait(remaining, num_returns=min(10, len(remaining)))
                results = ray.get(done)
                for success, error_info in results:
                    if success:
                        successful += 1
                    elif error_info:
                        self.failed_files.append(error_info)
                    pbar.update(1)

        failed = len(files) - successful
        end_time = datetime.now()
        duration = end_time - start_time

        logger.info(f"Completed: {successful}/{len(files)} files succeeded, {failed} failed")
        logger.info(f"Duration: {duration}")

        if failed > 0:
            self.log_failed_files()

        # Cleanup temp directory
        if self.temp_dir.exists():
            with contextlib.suppress(OSError):
                self.temp_dir.rmdir()

    def log_failed_files(self) -> None:
        """Write failed files to a timestamped log."""
        if not self.failed_files:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        failed_log_path = self.temp_dir / f"failed_files_{timestamp}.log"

        with open(failed_log_path, "w") as f:
            f.write(f"Failed files for: {self.source_url}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Total failed: {len(self.failed_files)}\n\n")

            for failed_file in self.failed_files:
                f.write(f"URL: {failed_file['url']}\n")
                f.write(f"Filename: {failed_file['filename']}\n")
                f.write(f"Error: {failed_file['error']}\n")
                f.write("-" * 80 + "\n")

        logger.info(f"Failed files logged to: {failed_log_path}")

    def retry_failed_files(self) -> None:
        """Sequentially retry failed files after ray.shutdown()."""
        if not self.failed_files:
            return

        logger.info(f"Retrying {len(self.failed_files)} failed downloads sequentially")
        still_failed = []
        for failed_file in self.failed_files:
            success, error_info = _download_and_upload_file(
                url=failed_file["url"],
                relative_key=failed_file["filename"],
                s3_bucket=self.s3_bucket,
                s3_output_dir=self.s3_output_dir,
                local_temp_dir=self.local_temp_dir,
                max_retries=self.max_retries,
                backoff_factor=self.backoff_factor,
                download_timeout=self.download_timeout,
            )
            if not success and error_info:
                still_failed.append(error_info)

        if still_failed:
            logger.warning(f"{len(still_failed)} files still failed after retry")
            self.failed_files = still_failed
            self.log_failed_files()
        else:
            logger.info("All retries succeeded")
            self.failed_files = []


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download dataset from HTTP directory listing to S3")
    parser.add_argument(
        "--source-url",
        type=str,
        default="https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/",
        help="HTTP directory listing URL to crawl",
    )
    parser.add_argument(
        "--s3-output-path",
        type=str,
        required=True,
        help="S3 destination path",
    )
    parser.add_argument(
        "--local-temp-dir",
        type=str,
        default="/tmp/bridge_v2_download",
        help="Local temp directory for downloads",
    )
    parser.add_argument("--max-retries", type=int, default=10, help="Number of retries for failed downloads")
    parser.add_argument("--backoff-factor", type=float, default=3.0, help="Backoff factor for retries")
    parser.add_argument("--download-timeout", type=int, default=300, help="Download timeout in seconds")
    parser.add_argument("--ray-address", type=str, default=None, help="Ray cluster address")
    parser.add_argument("--ray-num-cpus", type=int, default=None, help="Number of CPUs for Ray")

    args = parser.parse_args()

    # Initialize Ray
    if args.ray_address:
        ray.init(address=args.ray_address)
        logger.info(f"Connected to Ray cluster at {args.ray_address}")
    else:
        ray.init(num_cpus=args.ray_num_cpus)
        logger.info(f"Started local Ray with num_cpus={args.ray_num_cpus}")

    s3_bucket, s3_output_dir = parse_s3_path(args.s3_output_path)

    downloader = HttpDatasetDownloader(
        source_url=args.source_url,
        s3_bucket=s3_bucket,
        s3_output_dir=s3_output_dir,
        local_temp_dir=args.local_temp_dir,
        max_retries=args.max_retries,
        backoff_factor=args.backoff_factor,
        download_timeout=args.download_timeout,
    )

    downloader.download_dataset()

    ray.shutdown()

    # Retry failed downloads sequentially
    downloader.retry_failed_files()
