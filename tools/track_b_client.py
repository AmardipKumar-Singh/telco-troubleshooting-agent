"""
HTTP client for Track B (IP Networks) Agent Tool Server.
Covers topology, routing, fault, CLI and path tools.
Inspect Track B/server.py to add any remaining endpoints.
"""
import os
import requests

BASE_URL   = os.getenv("TRACK_B_SERVER_URL", "http://localhost:8001")
AUTH_TOKEN = os.getenv("TRACK_B_AUTH_TOKEN", "")
HEADERS    = {"Authorization": f"Bearer {AUTH_TOKEN}", "Content-Type": "application/json"}
TIMEOUT    = 30


def _post(endpoint: str, payload: dict) -> str:
    resp = requests.post(f"{BASE_URL}{endpoint}", json=payload, headers=HEADERS, timeout=TIMEOUT)
    resp.raise_for_status()
    return str(resp.json())


def get_topology(node_id: str = None) -> str:
    """Return full or node-specific network topology graph (nodes + links)."""
    return _post("/api/topology", {"node_id": node_id})


def get_link_status(node_a: str, node_b: str) -> str:
    """Return the operational status of a link between two nodes."""
    return _post("/api/link/status", {"node_a": node_a, "node_b": node_b})


def trace_path(src: str, dst: str, protocol: str = "IP") -> str:
    """Trace the active forwarding path between source and destination nodes."""
    return _post("/api/path/trace", {"src": src, "dst": dst, "protocol": protocol})


def get_routing_table(node_id: str, vrf: str = "default") -> str:
    """Return the routing table of a network device, optionally scoped to a VRF."""
    return _post("/api/routing/table", {"node_id": node_id, "vrf": vrf})


def get_fault_log(node_id: str = None, limit: int = 20) -> str:
    """Return fault/event logs for a specific node or the entire network."""
    return _post("/api/faults", {"node_id": node_id, "limit": limit})


def get_interface_stats(node_id: str, interface: str) -> str:
    """Return traffic, error and drop counters for a specific device interface."""
    return _post("/api/interface/stats", {"node_id": node_id, "interface": interface})


def run_cli_command(node_id: str, command: str, vendor: str = "auto") -> str:
    """Execute a CLI command on a device. Vendor: huawei | cisco | h3c | auto."""
    return _post("/api/cli/exec", {"node_id": node_id, "command": command, "vendor": vendor})


def get_bgp_summary(node_id: str) -> str:
    """Return BGP peer summary and session states for a device."""
    return _post("/api/bgp/summary", {"node_id": node_id})


def get_ospf_neighbors(node_id: str) -> str:
    """Return OSPF neighbour adjacency table for a device."""
    return _post("/api/ospf/neighbors", {"node_id": node_id})


def restore_topology(action: str, node_id: str = None, link: dict = None) -> str:
    """Restore network topology by re-enabling a node or link after a fault."""
    return _post("/api/topology/restore", {"action": action, "node_id": node_id, "link": link})


# Tool registry — add remaining tools by inspecting Track B/server.py
TRACK_B_TOOLS = {
    "get_topology":         get_topology,
    "get_link_status":      get_link_status,
    "trace_path":           trace_path,
    "get_routing_table":    get_routing_table,
    "get_fault_log":        get_fault_log,
    "get_interface_stats":  get_interface_stats,
    "run_cli_command":      run_cli_command,
    "get_bgp_summary":      get_bgp_summary,
    "get_ospf_neighbors":   get_ospf_neighbors,
    "restore_topology":     restore_topology,
}
