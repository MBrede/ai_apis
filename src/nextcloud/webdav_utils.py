"""
WebDAV client setup and small download helper, shared by sync.py and
sync_llm.py (and everything else in this package).
"""

import os
from pathlib import Path

from webdav3.client import Client


def _make_webdav_client() -> tuple[Client, list[str]]:
    """Create a WebDAV client from environment variables.

    NEXTCLOUD_FOLDER may list several folders, comma-separated (e.g.
    ``"transcription,/Shared/1 Projects/MBS Benkana/transcription"``) — each
    is scanned independently by callers. A single folder (no comma) produces
    a 1-element list, identical to the pre-multi-folder behavior.

    Returns:
        Tuple of (client, list of remote root paths, one per folder).

    Raises:
        KeyError: If a required environment variable is missing.
    """
    url = os.environ["NEXTCLOUD_URL"].rstrip("/")
    user = os.environ["NEXTCLOUD_USER"]
    password = os.environ["NEXTCLOUD_PASSWORD"]
    folders = [f.strip().strip("/") for f in os.environ["NEXTCLOUD_FOLDER"].split(",") if f.strip()]
    # NEXTCLOUD_DAV_USER: the internal Nextcloud username used in the WebDAV path.
    # Defaults to NEXTCLOUD_USER. Set this when the login credential (email) differs
    # from the internal username, e.g. NEXTCLOUD_USER=user@example.com but
    # WebDAV path uses /remote.php/dav/files/username/.
    dav_user = os.environ.get("NEXTCLOUD_DAV_USER") or user
    client = Client(
        {
            "webdav_hostname": url,
            "webdav_login": user,
            "webdav_password": password,
        }
    )
    remote_roots = [f"/remote.php/dav/files/{dav_user}/{folder}" for folder in folders]
    return client, remote_roots


def _download_text(client: Client, remote_path: str, tmp_dir: Path) -> str:
    """Download a small remote text file and return its contents.

    Shared by sync.py (sidecar `stt:` parsing) and sync_llm.py (sidecar
    parsing, prompt templates, transcripts).

    Args:
        client: WebDAV client.
        remote_path: Remote file path to download.
        tmp_dir: Local scratch directory for the temporary download.

    Returns:
        Decoded UTF-8 file contents.
    """
    local_path = tmp_dir / Path(remote_path).name
    client.download_sync(remote_path=remote_path, local_path=str(local_path))
    try:
        return local_path.read_text(encoding="utf-8")
    finally:
        local_path.unlink(missing_ok=True)
