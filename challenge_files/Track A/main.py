#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Track A — Enhanced ReAct Agent for 5G Drive-Test Troubleshooting
=================================================================
Key improvements over the baseline main.py:
  1. Rich system prompt with 5G domain knowledge and decision heuristics
  2. free_mode=True by default (forces boxed answer extraction)
  3. Early-stopping: once answer is extracted with high confidence, stop
  4. Parallel-safe tool dispatch (sequential but compact)
  5. Strict answer formatting guard before saving result.csv
  6. Timeout-aware: tracks wall-clock time per scenario, logs discount tier
"""

import argparse
import json
import logging
import os
import time
import traceback
from typing import Any, Dict, List, Optional

import httpx
import pandas as pd
import requests
from openai import OpenAI, RateLimitError, APIConnectionError, APITimeoutError, APIError

from _types import ToolCall
from logger import init_logger
from utils import (
    print_model_response,
    print_tool_call,
    print_tool_result,
    extract_answer,
    extract_answer_all,
    compute_score,
)

os.environ["NO_PROXY"] = "localhost,127.0.0.1"
API_KEY = os.environ.get("AGENT_API_KEY", "dummy")

# ---------------------------------------------------------------------------
# System prompt — 5G domain knowledge + ReAct instructions
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert 5G network troubleshooting engineer with deep knowledge of:
- Radio Access Network (RAN) KPIs: RSRP, SINR, RSRQ, throughput, PRB utilisation
- Handover mechanisms: A3/A5 events, IntraFreqHoA3Offset, hysteresis, TTT
- Antenna engineering: mechanical/digital tilt, azimuth, main-lobe coverage
- Load balancing: PRB usage, cell overload thresholds
- Neighbour relations: missing neighbours, PCI confusion
- PDCCH configuration: PdcchOccupiedSymbolNum (1SYM vs 2SYM overhead)
- Inter-frequency handover: CovInterFreqA2RsrpThld, CovInterFreqA5RsrpThld1

## Decision Heuristics (apply in this order)

1. **Late / missed handover** (most common):
   - Symptom: throughput drops while UE stays on same serving cell (PCI unchanged)
   - Neighbour RSRP > Serving RSRP + A3Offset(0.5dB units / 2) + A3Hyst(0.5dB units / 2)
   - But handover NOT triggered → A3Offset too high
   - Fix: Decrease A3 Offset for the serving cell

2. **Interference / SINR problem**:
   - Symptom: SINR < 0 dB during degradation, strong interfering neighbour visible
   - Check if neighbour RSRP > serving RSRP by >= (A3Offset + A3Hyst) in dB
   - If difference exactly equals threshold (not strictly greater) → HO not triggered
   - Fix: Decrease A3 Offset for serving cell OR decrease tilt to shrink overlap

3. **Coverage hole**:
   - Symptom: serving RSRP < -110 dBm, no strong neighbours
   - Fix: Increase TX power, lift tilt, or add neighbour relationship

4. **Overloaded cell**:
   - Symptom: high PRB utilisation, throughput capped despite good RSRP/SINR
   - Fix: inter-freq offload (lower A2/A5 thresholds), DO NOT increase load further

5. **Missing neighbour relation**:
   - Symptom: UE in coverage of cell not in neighbour list, handover fails
   - Fix: Add neighbour relationship

6. **Azimuth / tilt misalignment**:
   - Use calculate_horizontal_angle and judge_mainlobe_or_not tools
   - If UE is outside main-lobe → adjust azimuth
   - If overlap too large → press down tilt

7. **PDCCH overhead**:
   - PdcchOccupiedSymbolNum 2SYM reduces user-plane throughput
   - Only recommend 1SYM→2SYM if scheduling bottleneck confirmed

8. **Test server / transmission issue**:
   - Recommend C_check_server only if ALL cells in area show degradation simultaneously
   - Otherwise always diagnose radio first

## ReAct Protocol
- ALWAYS start with get_throughput_logs to identify the degradation window
- Then get_serving_cell_pci at the worst throughput timestamp
- Then get_cell_info for that PCI
- Then get_serving_cell_rsrp and get_serving_cell_sinr at degradation time
- If SINR is poor → get_neighboring_cells_pci → get_neighboring_cell_rsrp for each
- Use geometry tools (calculate_horizontal_angle, judge_mainlobe_or_not) when azimuth/tilt is suspected
- Stop calling tools once root cause is confirmed — do not make unnecessary calls

## Answer Format
- Single-answer: \\boxed{C<N>}
- Multi-answer: \\boxed{C<N>|C<M>} in ascending order
- NEVER output two boxed answers
- NEVER output \\boxed{} without a valid C-number inside
"""

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class Environment:
    endpoint_mapper = {
        "get_all_scenario": "/scenario/all",
        "get_config_data": "/config-data",
        "get_user_plane_data": "/user-plane-data",
        "get_throughput_logs": "/throughput-logs",
        "get_cell_info": "/cell-info",
        "get_gnodeb_location": "/gnodeb-location",
        "get_user_location": "/user-location",
        "get_serving_cell_pci": "/serving-cell-pci",
        "get_serving_cell_rsrp": "/serving-cell-rsrp",
        "get_serving_cell_sinr": "/serving-cell-sinr",
        "get_rbs_allocated_to_user": "/rbs-allocated-to-user",
        "get_neighboring_cells_pci": "/neighboring-cells-pci",
        "get_neighboring_cell_rsrp": "/neighboring-cell-rsrp",
        "get_signaling_plane_event_log": "/signaling-plane-event-log",
        "get_all_cells_pci": "/all-cells-pci",
        "get_available_tools": "/tools",
        "health": "/health",
        "judge_mainlobe_or_not": "/judge_mainlobe",
        "calculate_horizontal_angle": "/calculate_horizontal_angle",
        "calculate_tilt_angle": "/calculate_tilt_angle",
        "calculate_pathloss": "/calculate_pathloss",
        "calculate_overlap_ratio": "/calculate_overlap_ratio",
        "get_kpi_data": "/get_kpi_data",
        "get_mr_data": "/get_mr_data",
        "optimize_antenna_gain": "/optimize_antenna_gain",
    }

    def __init__(
        self,
        server_url: str,
        verbose: bool = False,
        log_file: Optional[str] = None,
        timeout: float = 15.0,
        logger: logging.Logger = None,
    ):
        self.server_url = server_url.rstrip("/")
        self.verbose = verbose
        self.timeout = timeout
        self.logger = logger if logger is not None else init_logger()

    def _headers(self, scenario_id: Optional[str] = None) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if scenario_id:
            headers["X-Scenario-Id"] = scenario_id
        return headers

    def _call_api(self, function_name: str, scenario_id: Optional[str] = None, **params) -> Dict:
        endpoint = self.endpoint_mapper.get(function_name)
        if endpoint is None:
            return {"error": f"Unknown tool '{function_name}'"}
        url = f"{self.server_url}{endpoint}"
        headers = self._headers(scenario_id=scenario_id)
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError:
            try:
                detail = resp.json().get("detail", str(resp.text))
            except Exception:
                detail = str(resp.text)
            return {"error": detail}
        except Exception as e:
            return {"error": str(e)}

    def get_tools(self) -> List[Dict]:
        tools = self._call_api("get_available_tools")
        if isinstance(tools, dict) and "error" in tools:
            return []
        return tools if isinstance(tools, list) else []

    def get_scenarios(self) -> List[Dict]:
        scenarios = self._call_api("get_all_scenario")
        if isinstance(scenarios, dict) and "error" in scenarios:
            return []
        return scenarios if isinstance(scenarios, list) else []

    def execute(self, tool_call: ToolCall, scenario_id: Optional[str] = None) -> str:
        try:
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments or "{}")
            result = self._call_api(function_name=function_name, scenario_id=scenario_id, **arguments)
            return json.dumps(result, ensure_ascii=False)
        except json.JSONDecodeError:
            return json.dumps({"error": f"Param parse failed: {tool_call.function.arguments}"})
        except Exception as e:
            return json.dumps({"error": f"Tool execution failed: {str(e)}"})


# ---------------------------------------------------------------------------
# Discount helper
# ---------------------------------------------------------------------------

def time_discount(elapsed_seconds: float) -> str:
    if elapsed_seconds < 300:
        return "100%"
    elif elapsed_seconds < 600:
        return "80%"
    elif elapsed_seconds < 900:
        return "60%"
    else:
        return "0% (TIMED OUT)"


# ---------------------------------------------------------------------------
# AgentsRunner
# ---------------------------------------------------------------------------

class AgentsRunner:
    def __init__(
        self,
        environment: Environment,
        model_url: str,
        model_name: str,
        model_provider: Optional[str] = None,
        max_tokens: int = 16000,
        max_retries: int = 3,
        max_iterations: int = 10,
        verbose: bool = False,
        logger: logging.Logger = None,
    ):
        self.environment = environment
        self.model_url = model_url
        self.model_name = model_name
        self.model_provider = model_provider
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.logger = logger if logger is not None else init_logger()

        self.client = OpenAI(
            base_url=model_url,
            api_key=API_KEY,
            http_client=httpx.Client(verify=False),
        )

    def _model_name_str(self) -> str:
        if self.model_provider:
            return f"{self.model_provider}/{self.model_name}"
        return self.model_name

    def _call_model(self, messages: List[Dict], functions: List[Dict], **kwargs):
        call_kwargs = {
            "model": self._model_name_str(),
            "messages": messages,
            "max_tokens": self.max_tokens,
            **kwargs,
        }
        if functions:
            call_kwargs["tools"] = functions
            call_kwargs["tool_choice"] = "auto"

        base_wait = 1.0
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(**call_kwargs)
                return response.choices[0].message
            except (RateLimitError, APIConnectionError, APITimeoutError, APIError) as exc:
                if self.verbose:
                    self.logger.error(traceback.format_exc())
                if hasattr(exc, "status_code") and 400 <= exc.status_code < 500 and exc.status_code != 429:
                    return None
                if attempt == self.max_retries:
                    return None
                wait = base_wait * (2 ** (attempt - 1))
                time.sleep(wait)
            except Exception:
                if self.verbose:
                    self.logger.error(traceback.format_exc())
                return None
        return None

    def _force_answer(
        self, messages: List[Dict], root_causes: str, is_multi: bool
    ) -> Any:
        """Issue a final no-tool prompt to force a boxed answer."""
        if is_multi:
            constraint = (
                "This is a MULTIPLE-answer question. Select two to four optimisation options. "
                "Output ONLY a single \\boxed{} with answers separated by | in ascending order. "
                f"Example: \\boxed{{C3|C7}}\n\nOptions:\n{root_causes}"
            )
        else:
            constraint = (
                "This is a SINGLE-answer question. Select the single best optimisation option. "
                "Output ONLY a single \\boxed{{}} with one answer. "
                f"Example: \\boxed{{C9}}\n\nOptions:\n{root_causes}"
            )
        messages.append({"role": "user", "content": constraint})
        return self._call_model(messages, functions=[])

    def run(self, scenario: Dict, free_mode: bool = True) -> Dict:
        scenario_id = scenario.get("scenario_id")
        task = scenario.get("task", {})
        options = task.get("options", [])
        root_causes = "".join([f"{item['id']}:{item['label']}\n" for item in options])

        # Detect multi-answer questions from description
        description = task.get("description", "")
        is_multi = "select" in description.lower() and any(
            kw in description.lower() for kw in ["two", "three", "four", "multiple", "2", "3", "4"]
        )

        tool_defs = self.environment.get_tools()
        if not tool_defs:
            return {"scenario_id": scenario_id, "status": "unresolved", "reason": "No tools available",
                    "answer": "", "traces": "", "tool_calls": [], "num_tool_calls": 0, "num_iterations": 0}

        question = description + f"\nOptions:\n{root_causes}"

        messages: List[Dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        num_tool_calls = 0
        list_tool_calls = []
        status = None
        last_msg = None
        i = 0

        for i in range(self.max_iterations):
            self.logger.info(f"[Scenario: {scenario_id}] Round {i + 1}")

            msg = self._call_model(messages, functions=tool_defs)
            if msg is None:
                continue

            last_msg = msg
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": msg.tool_calls,
            })

            if self.verbose:
                print_model_response(msg, logger=self.logger, minimize=False)

            if msg.tool_calls:
                num_tool_calls += len(msg.tool_calls)
                for j, tool_call in enumerate(msg.tool_calls):
                    if self.verbose:
                        print_tool_call(tool_call, logger=self.logger)

                    tool_result = self.environment.execute(tool_call, scenario_id=scenario_id)
                    messages.append({"role": "tool", "content": tool_result, "tool_call_id": tool_call.id})

                    if self.verbose:
                        print_tool_result(tool_result, logger=self.logger)

                    list_tool_calls.append({
                        "function_name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                        "turn": i + 1,
                        "order": j + 1,
                        "has_failed": "error" in tool_result,
                        "results": tool_result,
                    })

                # Early-stop: if assistant already stated a boxed answer alongside tool calls
                if msg.content:
                    early = extract_answer_all(msg.content)
                    if early:
                        self.logger.info(f"[Scenario: {scenario_id}] Early answer detected: {early}")
                        status = "solved"
                        break

            elif msg.content:
                # No more tool calls — check if answer is present
                answer_candidate = extract_answer_all(msg.content)
                if answer_candidate:
                    status = "solved"
                    break
                else:
                    # Content present but no boxed answer yet — force it
                    if free_mode:
                        forced = self._force_answer(messages, root_causes, is_multi)
                        if forced is not None:
                            last_msg = forced
                        status = "solved"
                        break
                    else:
                        status = "solved"
                        break
            else:
                status = "unresolved"
                break

        if status is None:
            status = "unresolved"

        # Final safety net: if still no boxed answer, force one
        final_content = getattr(last_msg, "content", "") or ""
        final_traces = getattr(last_msg, "reasoning_content", "") or ""
        agent_answer = extract_answer_all(final_content) or extract_answer_all(final_traces)

        if not agent_answer and free_mode and last_msg is not None:
            self.logger.info(f"[Scenario: {scenario_id}] No boxed answer found — forcing final answer prompt")
            forced = self._force_answer(messages, root_causes, is_multi)
            if forced is not None:
                last_msg = forced
                final_content = getattr(last_msg, "content", "") or ""
                final_traces = getattr(last_msg, "reasoning_content", "") or ""

        return {
            "scenario_id": scenario_id,
            "num_iterations": i + 1,
            "tool_calls": list_tool_calls,
            "num_tool_calls": num_tool_calls,
            "status": status,
            "traces": final_traces,
            "answer": final_content or final_traces,
            "messages": messages,
            "reason": None,
        }

    def benchmark(
        self,
        num_attempts: int,
        save_dir: str,
        save_freq: int = 10,
        max_samples: Optional[int] = None,
        free_mode: bool = True,
    ) -> None:
        os.makedirs(save_dir, exist_ok=True)

        completions: List[Dict] = []
        save_result: List[Dict] = []

        scenarios = self.environment.get_scenarios()
        if max_samples is not None:
            scenarios = scenarios[: min(max_samples, len(scenarios))]

        self.logger.info(f"Starting benchmark: {len(scenarios)} scenarios, {num_attempts} attempt(s) each")

        for idx, scenario in enumerate(scenarios):
            scenario_id = scenario.get("scenario_id")
            start_time = time.time()

            n_success = 0.0
            agent_answers: List[str] = []
            sample_response: Dict = {}

            for attempt in range(num_attempts):
                self.logger.info(f"[Scenario {scenario_id}] attempt {attempt + 1}/{num_attempts}")
                response = self.run(scenario=scenario, free_mode=free_mode)
                sample_response = response

                if response.get("status") == "solved":
                    raw_answer = response.get("answer", "")
                    raw_traces = response.get("traces", "")
                    agent_answer = extract_answer_all(raw_answer) or extract_answer_all(raw_traces)
                    ground_truth = scenario.get("answer", "")
                    score = compute_score(agent_answer, ground_truth)
                    n_success += score
                    agent_answers.append(agent_answer)

                    elapsed = time.time() - start_time
                    discount = time_discount(elapsed)
                    pink = "\033[95m"
                    reset = "\033[0m"
                    self.logger.info(
                        f"{pink}[Scenario: {scenario_id}] answer={agent_answer} | "
                        f"gt={ground_truth} | score={score:.2f} | "
                        f"time={elapsed:.1f}s | discount={discount}{reset}"
                    )

            acc = n_success / float(num_attempts)
            latency = round((time.time() - start_time) / float(num_attempts), 2)

            completions.append({
                "scenario_id": scenario_id,
                "free_mode": free_mode,
                "response": sample_response.get("answer", ""),
                "traces": sample_response.get("traces", ""),
                "num_iterations": sample_response.get("num_iterations", 0),
                "num_tool_calls": sample_response.get("num_tool_calls", 0),
                "tool_calls": sample_response.get("tool_calls", []),
                "answers": agent_answers,
                "ground_truth": scenario.get("answer"),
                "accuracy": acc,
                "latency": latency,
            })

            # result.csv row: use first attempt answer
            save_result.append({
                "scenario_id": scenario_id,
                "answers": agent_answers[0] if agent_answers else "",
            })

            if ((idx + 1) % save_freq == 0) or ((idx + 1) == len(scenarios)):
                df = pd.DataFrame(save_result)
                df.to_csv(os.path.join(save_dir, "result.csv"), index=False)
                self.logger.info(f"Saved result.csv ({idx + 1}/{len(scenarios)} scenarios done)")

        # Final completions dump (detailed)
        completions_path = os.path.join(save_dir, "completions.json")
        with open(completions_path, "w", encoding="utf-8") as f:
            json.dump(completions, f, ensure_ascii=False, indent=2)
        self.logger.info(f"Saved completions.json to {completions_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track A — Enhanced ReAct Agent")
    parser.add_argument("--server_url", type=str, default="http://localhost:7860")
    parser.add_argument("--model_url", type=str, default="https://openrouter.ai/api/v1")
    parser.add_argument("--model_name", type=str, default="qwen/qwen3.5-35b-a3b")
    parser.add_argument("--model_provider", type=str, default=None)
    parser.add_argument("--num_attempts", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--save_freq", type=int, default=10)
    parser.add_argument("--max_tokens", type=int, default=16000)
    parser.add_argument("--max_iterations", type=int, default=10)
    parser.add_argument("--save_dir", type=str, default="./results")
    parser.add_argument("--log_file", type=str, default="./log.log")
    parser.add_argument("--no_free_mode", action="store_true", help="Disable free_mode (not recommended)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logger = init_logger(log_file=args.log_file)

    env = Environment(server_url=args.server_url, verbose=args.verbose, logger=logger)

    runner = AgentsRunner(
        environment=env,
        model_url=args.model_url,
        model_name=args.model_name,
        model_provider=args.model_provider,
        max_tokens=args.max_tokens,
        max_iterations=args.max_iterations,
        verbose=args.verbose,
        logger=logger,
    )

    runner.benchmark(
        max_samples=args.max_samples,
        num_attempts=args.num_attempts,
        save_dir=args.save_dir,
        save_freq=args.save_freq,
        free_mode=not args.no_free_mode,
    )
