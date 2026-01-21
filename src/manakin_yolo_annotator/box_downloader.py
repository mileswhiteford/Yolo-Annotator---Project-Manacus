"""
Utility helpers for pulling annotated videos from Box and preparing local audio clips.

This module adapts the original Colab-only BoxNavigator notebook cell into a reusable
Python component that works anywhere (Colab, local dev machine, etc.). Paths default
to the current working directory but can be overridden when instantiating BoxNavigator.
"""

from __future__ import annotations

import json
import os
import time
import shutil
import requests
from typing import Optional, Dict, List, Tuple, TYPE_CHECKING

from moviepy.video.io.VideoFileClip import VideoFileClip

if TYPE_CHECKING:  # for type checkers, without forcing runtime dependency
    from IPython.display import Audio as _IPyAudio, display as _ipy_display, clear_output as _ipy_clear_output

try:  # Optional; available in notebooks but not required elsewhere
    from IPython.display import Audio, display, clear_output  # type: ignore
except ImportError:  # pragma: no cover - fallback when IPython is missing
    Audio = None

    def display(*args, **kwargs):
        for arg in args:
            print(arg)

    def clear_output(*args, **kwargs):
        pass


class BoxNavigator:
    """
    Lightweight Box client powered by the REST API + Developer Token.

    - Credentials live under ``<system_files_dir>/box_credentials.txt``
    - Files download into ``<download_dir>``
    - Provide ``base_dir`` to control defaults when explicit dirs are not set
    """

    def __init__(
        self,
        base_dir: Optional[str] = None,
        system_files_dir: Optional[str] = None,
        download_dir: Optional[str] = None,
    ) -> None:
        self.home_dir = os.path.abspath(base_dir or os.getcwd())
        self.system_files_dir = os.path.abspath(system_files_dir) if system_files_dir else os.path.join(
            self.home_dir, "system_files"
        )
        self.download_dir = os.path.abspath(download_dir) if download_dir else os.path.join(
            self.home_dir, "videos"
        )

        os.makedirs(self.home_dir, exist_ok=True)
        os.makedirs(self.download_dir, exist_ok=True)

        if os.path.exists(self._cred_file_path()):
            self.load_credentials_from_file()
        else:
            self.setup_credentials()

        self._build_session()

    # ---------- Credentials ----------
    def _cred_file_path(self) -> str:
        return os.path.join(self.system_files_dir, "box_credentials.txt")

    def _index_file_path(self, folder_id: str) -> str:
        safe_id = "".join(ch for ch in folder_id if ch.isalnum() or ch in ("-", "_"))
        return os.path.join(self.system_files_dir, f"box_index_{safe_id}.json")

    def load_credentials_from_file(self) -> None:
        os.makedirs(self.system_files_dir, exist_ok=True)
        with open(self._cred_file_path(), "r", encoding="utf-8") as file:
            lines = [ln.strip() for ln in file.readlines()]

        self.client_id = lines[0] if len(lines) >= 1 else ""
        self.client_secret = lines[1] if len(lines) >= 2 else ""
        self.access_token = lines[2] if len(lines) >= 3 else ""
        if not self.access_token:
            self.access_token = lines[0] if lines else ""

        if not self.access_token:
            self.access_token = input("Enter your Box Developer Token: ").strip()
            with open(self._cred_file_path(), "w", encoding="utf-8") as f:
                f.write(f"{self.client_id}\n{self.client_secret}\n{self.access_token}")

    def setup_credentials(self) -> None:
        os.makedirs(self.system_files_dir, exist_ok=True)
        print("Login to https://tulane.app.box.com/developers/console and create a Developer Token.")
        self.client_id = input("Enter your Box Client ID (optional, press Enter to skip): ").strip()
        self.client_secret = input("Enter your Box Client Secret (optional, press Enter to skip): ").strip()
        self.access_token = input("Enter your Box Developer Token: ").strip()

        with open(self._cred_file_path(), "w", encoding="utf-8") as file:
            file.write(f"{self.client_id}\n{self.client_secret}\n{self.access_token}")

    def _build_session(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {self.access_token}"})
        self.base_url = "https://api.box.com/2.0"

    def _update_token(self) -> None:
        self.access_token = input("Your Box Developer Token seems expired. Enter a new token: ").strip()
        with open(self._cred_file_path(), "w", encoding="utf-8") as f:
            f.write(f"{self.client_id}\n{self.client_secret}\n{self.access_token}")
        self._build_session()

    def _request(self, method: str, url: str, **kwargs):
        resp = self.session.request(method, url, **kwargs)
        if resp.status_code == 401:
            self._update_token()
            resp = self.session.request(method, url, **kwargs)
        resp.raise_for_status()
        return resp

    # ---------- Box operations ----------
    def search(self, video_name: str) -> Optional[Dict[str, str]]:
        params = {
            "query": video_name,
            "type": "file",
            "content_types": "name",
            "limit": 25,
        }
        url = f"{self.base_url}/search"
        try:
            data = self._request("GET", url, params=params).json()
        except requests.HTTPError as exc:
            print(f"Search error: {exc}")
            return None

        entries = data.get("entries", [])
        for item in entries:
            if item.get("type") == "file" and item.get("name") == video_name:
                return {"id": item.get("id"), "name": item.get("name"), "type": "file"}
        return None

    def list_files_recursive(self, folder_id: str) -> List[Dict[str, str]]:
        files: List[Dict[str, str]] = []
        stack = [folder_id]
        while stack:
            current_id = stack.pop()
            print(f"[box] Listing folder {current_id} ...")
            offset = 0
            limit = 1000
            while True:
                url = f"{self.base_url}/folders/{current_id}/items"
                params = {
                    "limit": limit,
                    "offset": offset,
                    "fields": "type,id,name",
                }
                try:
                    data = self._request("GET", url, params=params).json()
                except requests.HTTPError as exc:
                    print(f"[warn] Folder list failed for {current_id}: {exc}")
                    break

                entries = data.get("entries", [])
                total_count = data.get("total_count", 0)
                print(f"[box] Folder {current_id}: {offset}-{offset + len(entries)} of {total_count}")
                for item in entries:
                    if item.get("type") == "folder":
                        stack.append(item.get("id"))
                    elif item.get("type") == "file":
                        files.append({"id": item.get("id"), "name": item.get("name"), "type": "file"})

                offset += len(entries)
                if offset >= total_count or not entries:
                    break
        return files

    def load_box_index(self, folder_id: str) -> Optional[List[Dict[str, str]]]:
        path = self._index_file_path(folder_id)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return None
        return data.get("files")

    def build_box_index(self, folder_id: str) -> List[Dict[str, str]]:
        files = self.list_files_recursive(folder_id)
        payload = {
            "folder_id": folder_id,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "files": files,
        }
        os.makedirs(self.system_files_dir, exist_ok=True)
        path = self._index_file_path(folder_id)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        return files

    def search_videos_in_folder(self, folder_id: str) -> List[Dict[str, str]]:
        files: List[Dict[str, str]] = []
        offset = 0
        limit = 200
        while True:
            url = f"{self.base_url}/search"
            params = {
                "query": "mp4",
                "type": "file",
                "file_extensions": "mp4",
                "ancestor_folder_ids": folder_id,
                "limit": limit,
                "offset": offset,
                "fields": "type,id,name",
            }
            try:
                data = self._request("GET", url, params=params).json()
            except requests.HTTPError as exc:
                print(f"[warn] Search failed for folder {folder_id}: {exc}")
                break

            entries = data.get("entries", [])
            total_count = data.get("total_count", 0)
            print(f"[box] Search results: {offset}-{offset + len(entries)} of {total_count}")
            for item in entries:
                if item.get("type") == "file":
                    files.append({"id": item.get("id"), "name": item.get("name"), "type": "file"})

            offset += len(entries)
            if offset >= total_count or not entries:
                break
        return files

    def search_videos_in_folder_page(
        self, folder_id: str, limit: int = 200, offset: int = 0, query: str = "mp4"
    ) -> Tuple[List[Dict[str, str]], int]:
        url = f"{self.base_url}/search"
        params = {
            "query": query,
            "type": "file",
            "file_extensions": "mp4",
            "ancestor_folder_ids": folder_id,
            "limit": limit,
            "offset": offset,
            "fields": "type,id,name",
        }
        data = self._request("GET", url, params=params).json()
        entries = data.get("entries", [])
        total_count = data.get("total_count", 0)
        files = [
            {"id": item.get("id"), "name": item.get("name"), "type": "file"}
            for item in entries
            if item.get("type") == "file"
        ]
        return files, total_count

    def download_vid(self, video_name: str) -> Optional[str]:
        video_path = os.path.join(self.download_dir, video_name)
        if os.path.exists(video_path):
            print(f"Video '{video_name}' already exists at: {video_path}")
            return video_path

        result = self.search(video_name)
        if result and result["type"] == "file" and video_name.lower().endswith(".mp4"):
            file_id = result["id"]
            content_url = f"{self.base_url}/files/{file_id}/content"
            print(f"Downloading '{video_name}' to: {video_path} ...")
            try:
                with self._request("GET", content_url, stream=True) as response:
                    with open(video_path, "wb") as file:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                file.write(chunk)
                print(f"Successfully saved '{video_name}'.")
                return video_path
            except requests.HTTPError as exc:
                print(f"Download failed: {exc}")
                return None
        else:
            print(f"Video '{video_name}' not found in Box by exact name, or is not an mp4 file.")
            return None

__all__ = ["BoxNavigator"]
