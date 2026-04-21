"""
HTTP client for Track A (Wireless) Agent Tool Server.
Wraps server.py endpoints as Python callables for the ReAct agent.
Update BASE_URL and AUTH_TOKEN via environment variables.
"""
import os
import requests

BASE_URL    = os.getenv("TRACK_A_SERVER_URL", "http://localhost:8000")
AUTH_TOKEN  = os.getenv("TRACK_A_AUTH_TOKEN", "")
HEADERS     = {"Authorization": f"Bearer {AUTH_TOKEN}", "Content-Type": "application/json"}
TIMEOUT     = 30  # seconds


def _post(endpoint: str, payload: dict) -> str:
    resp = requests.post(f"{BASE_URL}{endpoint}", json=payload, headers=HEADERS, timeout=TIMEOUT)
    resp.raise_for_status()
    return str(resp.json())


def get_cell_kpis(cell_id: str, metric: str = "all", time_window: str = "1h") -> str:
    """Return KPI measurements (RSRP, SINR, PRB utilisation, throughput) for a cell."""
    return _post("/api/cell/kpis", {"cell_id": cell_id, "metric": metric, "time_window": time_window})


def get_alarms(cell_id: str = None, severity: str = "all", limit: int = 20) -> str:
    """Return active/recent network alarms filtered by cell and severity level."""
    return _post("/api/alarms", {"cell_id": cell_id, "severity": severity, "limit": limit})


def get_neighbor_config(cell_id: str) -> str:
    """Return the neighbour cell list (NCL) and handover parameters for a cell."""
    return _post("/api/cell/neighbors", {"cell_id": cell_id})


def get_interference_report(cell_id: str, direction: str = "DL") -> str:
    """Return inter-cell interference measurements for downlink or uplink direction."""
    return _post("/api/cell/interference", {"cell_id": cell_id, "direction": direction})


def get_cell_config(cell_id: str) -> str:
    """Return radio configuration (band, power, antenna tilt, PCI) for a cell."""
    return _post("/api/cell/config", {"cell_id": cell_id})


def get_ue_sessions(cell_id: str, limit: int = 10) -> str:
    """Return active UE session details currently attached to a cell."""
    return _post("/api/ue/sessions", {"cell_id": cell_id, "limit": limit})


def run_coverage_simulation(cell_id: str, scenario: str) -> str:
    """Simulate coverage under a given fault scenario (e.g. antenna_down, power_reduced)."""
    return _post("/api/simulation/coverage", {"cell_id": cell_id, "scenario": scenario})


# Tool registry exported to the agent
TRACK_A_TOOLS = {
    "get_cell_kpis":            get_cell_kpis,
    "get_alarms":               get_alarms,
    "get_neighbor_config":      get_neighbor_config,
    "get_interference_report":  get_interference_report,
    "get_cell_config":          get_cell_config,
    "get_ue_sessions":          get_ue_sessions,
    "run_coverage_simulation":  run_coverage_simulation,
}
