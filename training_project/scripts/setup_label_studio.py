#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""Create or update the Label Studio projects for this repo via the API.

For each project (schematics = page dataset, subcircuits = crop dataset):
- create it, or update its labeling config (labelstudio/<name>.xml)
- add a Local Files storage covering training_data/ -- without it Label Studio
  answers /data/local-files/ requests with 404 ("There was an issue loading
  URL from $image value"), even when the env vars are set. Not synced: it only
  grants access; tasks come from our own JSON.
- connect the ML backend (labelstudio/ml_backend.py) if it's running
- import the dataset's review_queue/label_studio_tasks.json, skipping images
  that already have a task in the project

Safe to run repeatedly. Needs Label Studio running (scripts/start_label_studio.sh)
and a personal access token (Account & Settings -> Personal Access Token) in
LABEL_STUDIO_API_KEY, e.g. in the repo's gitignored .envrc.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import argcomplete
import requests

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from config.settings import Config

PROJECTS = {
    "schematics": {
        "title": "Schematics (pages)",
        "config": None,  # config/config.yaml
        "label_config": PROJECT_ROOT / "labelstudio" / "schematics.xml",
        "description": "Schematic boxes on full datasheet pages (stage 1)",
    },
    "subcircuits": {
        "title": "Subcircuits (crops)",
        "config": "config/subcircuits.yaml",
        "label_config": PROJECT_ROOT / "labelstudio" / "subcircuits.xml",
        "description": "Functional blocks inside schematic crops (stage 2)",
    },
}
STORAGE_TITLE = "training_data (read-only access, don't sync)"
ML_BACKEND_TITLE = "YOLO (ml_backend.py)"


class LabelStudio:
    def __init__(self, url: str, api_key: str):
        self.url = url.rstrip("/")
        self.session = requests.Session()
        self.session.headers["Authorization"] = self._auth_header(api_key)

    def _auth_header(self, api_key: str) -> str:
        # Personal access tokens (default since 1.18) are JWT refresh tokens that
        # have to be exchanged for a short-lived access token. Legacy tokens are
        # sent as-is.
        response = requests.post(f"{self.url}/api/token/refresh/", json={"refresh": api_key})
        if response.ok and "access" in response.json():
            return f"Bearer {response.json()['access']}"
        return f"Token {api_key}"

    def request(self, method: str, path: str, **kwargs):
        response = self.session.request(method, f"{self.url}{path}", **kwargs)
        if not response.ok:
            raise RuntimeError(f"{method} {path} -> {response.status_code}: {response.text[:300]}")
        return response.json() if response.content else None

    def list_all(self, path: str, params=None) -> list:
        """Handles both plain lists and paginated {"results": [...], "next": ...} responses."""
        data = self.request("GET", path, params=params)
        if isinstance(data, list):
            return data
        items = list(data.get("results", []))
        while data.get("next"):
            data = self.session.get(data["next"]).json()
            items += data.get("results", [])
        return items

    def task_images(self, project_id: int) -> set[str]:
        """data.image of every task in a project (this endpoint 404s past the last page)."""
        images, page = set(), 1
        while True:
            response = self.session.get(
                f"{self.url}/api/projects/{project_id}/tasks/",
                params={"page": page, "page_size": 100},
            )
            if response.status_code == 404:
                return images
            response.raise_for_status()
            tasks = response.json()
            if not tasks:
                return images
            images |= {t["data"].get("image") for t in tasks}
            page += 1


def setup_project(ls: LabelStudio, key: str, backend_url: str):
    spec = PROJECTS[key]
    config = Config(config_file=spec["config"]) if spec["config"] else Config()
    label_config = spec["label_config"].read_text()
    print(f"\n== {spec['title']}")

    # 1. Project
    project = next(
        (p for p in ls.list_all("/api/projects/", {"page_size": 100}) if p["title"] == spec["title"]),
        None,
    )
    if project is None:
        project = ls.request(
            "POST",
            "/api/projects/",
            json={
                "title": spec["title"],
                "description": spec["description"],
                "label_config": label_config,
            },
        )
        print(f"created project #{project['id']}")
    elif _normalize_xml(project.get("label_config", "")) != _normalize_xml(label_config):
        ls.request("PATCH", f"/api/projects/{project['id']}/", json={"label_config": label_config})
        print(f"updated labeling config of project #{project['id']}")
    else:
        print(f"project #{project['id']} is up to date")
    project_id = project["id"]

    # 2. Local Files storage (must be a subfolder of the document root, i.e. the repo)
    storage_path = str(config.PROJECT_ROOT / "training_data")
    storages = ls.list_all("/api/storages/localfiles/", {"project": project_id})
    if not any(s.get("path") == storage_path for s in storages):
        ls.request(
            "POST",
            "/api/storages/localfiles/",
            json={
                "project": project_id,
                "title": STORAGE_TITLE,
                "path": storage_path,
                "use_blob_urls": False,
            },
        )
        print(f"added Local Files storage {storage_path}")
    else:
        print("Local Files storage already set up")

    # 3. ML backend
    backends = ls.list_all("/api/ml/", {"project": project_id})
    if any(b.get("url", "").rstrip("/") == backend_url.rstrip("/") for b in backends):
        print("ML backend already connected")
    elif _backend_up(backend_url):
        ls.request(
            "POST",
            "/api/ml/",
            json={"project": project_id, "url": backend_url, "title": ML_BACKEND_TITLE},
        )
        print(f"connected ML backend {backend_url}")
    else:
        print(
            f"ML backend not reachable at {backend_url}, skipped. Start it with\n"
            "  uv run python training_project/labelstudio/ml_backend.py\n"
            "and run this script again."
        )

    # 4. Tasks from the review queue
    tasks_path = config.TRAINING_DATA_PATH / "review_queue" / "label_studio_tasks.json"
    if not tasks_path.exists():
        print(f"no review queue at {tasks_path}")
        return
    queued = json.loads(tasks_path.read_text())
    existing = ls.task_images(project_id)
    new_tasks = [t for t in queued if t["data"]["image"] not in existing]
    if new_tasks:
        ls.request("POST", f"/api/projects/{project_id}/import", json=new_tasks)
    print(f"imported {len(new_tasks)} new task(s), {len(queued) - len(new_tasks)} already there")


def _normalize_xml(xml: str) -> str:
    """Label Studio reformats saved configs, so compare without whitespace between tags."""
    return re.sub(r">\s+<", "><", xml.strip())


def _backend_up(url: str) -> bool:
    try:
        return requests.get(f"{url.rstrip('/')}/health", timeout=2).ok
    except requests.RequestException:
        return False


def main():
    parser = argparse.ArgumentParser(description="Set up the Label Studio projects via the API")
    parser.add_argument("--project", choices=[*PROJECTS, "all"], default="all")
    parser.add_argument(
        "--url",
        default=os.environ.get("LABEL_STUDIO_URL", "http://localhost:8080"),
        help="Label Studio URL (default: $LABEL_STUDIO_URL or http://localhost:8080)",
    )
    parser.add_argument(
        "--backend-url", default="http://localhost:9090", help="ML backend URL"
    )

    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    api_key = os.environ.get("LABEL_STUDIO_API_KEY")
    if not api_key:
        print(
            "LABEL_STUDIO_API_KEY is not set. Create a personal access token in Label Studio\n"
            "(Account & Settings -> Personal Access Token) and add\n"
            '  export LABEL_STUDIO_API_KEY="<token>"\n'
            "to the repo's .envrc, then run `direnv allow`."
        )
        sys.exit(1)
    print("LABEL_STUDIO_API_KEY found")

    try:
        ls = LabelStudio(args.url, api_key)
        for key in PROJECTS if args.project == "all" else [args.project]:
            setup_project(ls, key, args.backend_url)
    except requests.ConnectionError:
        print(
            f"Label Studio is not reachable at {args.url}. Start it with\n"
            "  training_project/scripts/start_label_studio.sh"
        )
        sys.exit(1)
    except (RuntimeError, requests.RequestException) as e:
        print(f"Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
