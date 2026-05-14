# Agent Coordination

## Primary Agent: NOC Engineer

This configuration defines a single-agent setup. One agent handles all CTBench troubleshooting tasks sequentially.

## Behavior Rules

- Always complete the full reasoning cycle before producing a final answer
- Do not delegate to sub-agents; all tool calls are made by this agent
- Maximum concurrency within a single problem: 1 (sequential tool calls only)
- If a command returns a syntax error (HTTP 422), try the correct vendor syntax and retry once
- If a device is not found (HTTP 404), skip it and note it in the answer
- Stop calling tools once enough evidence is gathered to answer the question with confidence

## Session Naming

Sessions are prefixed with `ctbench-q{id}-` for traceability.
