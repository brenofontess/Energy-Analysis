"""
Utilities module for Energy Price Agents project.
Includes configuration loader, logger helpers, checksums, and gate verifiers.
"""

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


def get_project_root() -> Path:
  """Returns the absolute root directory of the project."""
  return Path(__file__).resolve().parent.parent


def load_config(config_path: Optional[str | Path] = None) -> Dict[str, Any]:
  """Loads the central project configuration yaml."""
  if config_path is None:
    config_path = get_project_root() / "config.yaml"
  with open(config_path, "r", encoding="utf-8") as f:
    return yaml.safe_load(f)


def calculate_file_hash(
    file_path: str | Path, algorithm: str = "sha256"
) -> str:
  """Calculates cryptographic checksum of a file for reproducibility and gate auditing."""
  hasher = hashlib.new(algorithm)
  with open(file_path, "rb") as f:
    for chunk in iter(lambda: f.read(65536), b""):
      hasher.update(chunk)
  return hasher.hexdigest()


def log_correction(
    correction_id: str,
    column: str,
    affected_rows: int,
    method: str,
    justification: str,
    before_summary: str,
    after_summary: str,
    approved_by: str = "Agente 4",
    log_csv_path: Optional[str | Path] = None,
) -> None:
  """Appends an entry into logs/correction_log.csv using proper CSV quoting."""
  if log_csv_path is None:
    log_csv_path = get_project_root() / "logs" / "correction_log.csv"
  log_path = Path(log_csv_path)
  log_path.parent.mkdir(parents=True, exist_ok=True)
  write_header = not log_path.exists() or log_path.stat().st_size == 0

  now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
  with open(log_path, "a", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    if write_header:
      writer.writerow([
          "id",
          "timestamp_utc",
          "column",
          "affected_rows",
          "method",
          "justification",
          "before_summary",
          "after_summary",
          "approved_by",
      ])
    writer.writerow([
        correction_id,
        now_utc,
        column,
        int(affected_rows),
        method,
        justification,
        before_summary,
        after_summary,
        approved_by,
    ])


def update_gate_status(
    gate_key: str,
    status: str,
    checks_dict: Dict[str, Any],
    notes: str = "",
    gates_json_path: Optional[str | Path] = None,
    from_agent: str = "",
    to_agent: str = "",
) -> None:
  """Updates logs/gate_results.json with current checks and gate status."""
  if gates_json_path is None:
    gates_json_path = get_project_root() / "logs" / "gate_results.json"
  gpath = Path(gates_json_path)

  if gpath.exists():
    with open(gpath, "r", encoding="utf-8") as f:
      data = json.load(f)
  else:
    data = {"project": "energy-price-agents-mibel-pt", "gates": {}}

  gates = data.setdefault("gates", {})
  if gate_key not in gates:
    gates[gate_key] = {
        "name": gate_key,
        "from_agent": from_agent,
        "to_agent": to_agent,
        "status": status,
        "checks": {},
        "notes": "",
    }
  gates[gate_key]["status"] = status
  gates[gate_key].setdefault("checks", {}).update(checks_dict)
  if from_agent:
    gates[gate_key]["from_agent"] = from_agent
  if to_agent:
    gates[gate_key]["to_agent"] = to_agent
  if notes:
    gates[gate_key]["notes"] = notes
  data["last_updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

  with open(gpath, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)


def append_to_run_log(
    agent_name: str,
    crisp_phase: str,
    event_description: str,
    status: str,
    notes: str = "",
    log_md_path: Optional[str | Path] = None,
) -> None:
  """Appends an execution event to logs/run_log.md."""
  if log_md_path is None:
    log_md_path = get_project_root() / "logs" / "run_log.md"

  now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
  table_row = (
      f"| {now_utc} | {agent_name} | {crisp_phase} | "
      f"{event_description} | {status} | {notes} |\n"
  )

  with open(log_md_path, "a", encoding="utf-8") as f:
    f.write(table_row)
