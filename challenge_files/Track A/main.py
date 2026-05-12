#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Track A — Enhanced ReAct Agent for 5G Drive-Test Troubleshooting
=================================================================
Fixes in this version:
  - Primary model: qwen/qwen3-coder:free (strongest free Qwen, Tools, 262K ctx)
  - Rotation list: 7 models confirmed active with Tools support (May 2026)
  - 200 req/day per-model budget guard — rotates before hitting daily cap
  - max_retries=0 on OpenAI client — prevents duplicate retry logs
  - NotFoundError rotates immediately without looping
  - Rate limiter default: 15 req/min (safe under 20/min free cap)
  - save_freq=5 for more frequent checkpointing
"""

import argparse
import json
import logging
import os
import random
import sys
import threading
import time
import traceback
from typing import Any, Dict, List, Optional

import httpx
import pandas as pd
import requests
from openai import (
    OpenAI, RateLimitError, APIConnectionError,
    APITimeoutError, APIError, NotFoundError
)

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
# System prompt
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
   - But handover NOT triggered -> A3Offset too high
   - Fix: Decrease A3 Offset for the serving cell

2. **Interference / SINR problem**:
   - Symptom: SINR < 0 dB during degradation, strong interfering neighbour visible
   - Check if neighbour RSRP > serving RSRP by >= (A3Offset + A3Hyst) in dB
   - Fix: Decrease A3 Offset for serving cell OR decrease tilt to shrink overlap

3. **Coverage hole**:
   - Symptom: serving RSRP < -110 dBm, no strong neighbours
   - Fix: Increase TX power, lift tilt, or add neighbour relationship

4. **Overloaded cell**:
   - Symptom: high PRB utilisation, throughput capped despite good RSRP/SINR
   - Fix: inter-freq offload (lower A2/A5 thresholds)

5. **Missing neighbour relation**:
   - Symptom: UE in coverage of cell not in neighbour list, handover fails
   - Fix: Add neighbour relationship

6. **Azimuth / tilt misalignment**:
   - Use calculate_horizontal_angle and judge_mainlobe_or_not tools
   - If UE outside main-lobe -> adjust azimuth; if overlap too large -> press down tilt

7. **PDCCH overhead**:
   - Only recommend 1SYM->2SYM if scheduling bottleneck confirmed

8. **Test server / transmission issue**:
   - Recommend C_check_server only if ALL cells show degradation simultaneously

## ReAct Protocol
- ALWAYS start with get_throughput_logs to find the degradation window
- Then get_serving_cell_pci at worst throughput timestamp
- Then get_cell_info for that PCI
- Then get_serving_cell_rsrp and get_serving_cell_sinr at degradation time
- If SINR poor -> get_neighboring_cells_pci -> get_neighboring_cell_rsrp for each
- Use geometry tools when azimuth/tilt suspected
- Stop calling tools once root cause confirmed

## Answer Format
- Single: \\boxed{C<N>}
- Multi: \\boxed{C<N>|C<M>} ascending order
- NEVER two boxed answers; NEVER empty \\boxed{}

## Tool Call Format
Use the function-calling interface ONLY. Do NOT write tool calls as plain text.
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

    def __init__(self, server_url, verbose=False, log_file=None, timeout=15.0, logger=None):
        self.server_url = server_url.rstrip("/")
        self.verbose = verbose
        self.timeout = timeout
        self.logger = logger or init_logger()

    def health_check(self) -> bool:
        url = f"{self.server_url}/health"
        try:
            resp = requests.get(url, timeout=5.0)
            if resp.status_code == 200:
                self.logger.info(f"Tool Server health check PASSED: {url}")
                return True
            self.logger.error(f"Health check FAILED (HTTP {resp.status_code}): {url}")
            return False
        except Exception as e:
            self.logger.error(f"Tool Server NOT reachable at {url}: {e}")
            return False

    def _headers(self, scenario_id=None):
        h = {"Content-Type": "application/json"}
        if scenario_id:
            h["X-Scenario-Id"] = scenario_id
        return h

    def _call_api(self, function_name, scenario_id=None, **params):
        endpoint = self.endpoint_mapper.get(function_name)
        if endpoint is None:
            return {"error": f"Unknown tool '{function_name}'"}
        url = f"{self.server_url}{endpoint}"
        try:
            resp = requests.get(url, params=params, headers=self._headers(scenario_id), timeout=self.timeout)
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
            self.logger.error(f"get_tools() failed: {tools['error']}")
            return []
        return tools if isinstance(tools, list) else []

    def get_scenarios(self) -> List[Dict]:
        scenarios = self._call_api("get_all_scenario")
        if isinstance(scenarios, dict) and "error" in scenarios:
            self.logger.error(f"get_scenarios() failed: {scenarios['error']}")
            return []
        return scenarios if isinstance(scenarios, list) else []

    def execute(self, tool_call: ToolCall, scenario_id=None) -> str:
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
# Helpers
# ---------------------------------------------------------------------------

def time_discount(elapsed_seconds: float) -> str:
    if elapsed_seconds < 300: return "100%"
    elif elapsed_seconds < 600: return "80%"
    elif elapsed_seconds < 900: return "60%"
    else: return "0% (TIMED OUT)"


class RateLimiter:
    """Proactive token-bucket rate limiter."""
    def __init__(self, max_per_minute: int = 15):
        self.min_interval = 60.0 / max_per_minute
        self.last_call_time = 0.0
        self._lock = threading.Lock()

    def wait(self):
        with self._lock:
            elapsed = time.time() - self.last_call_time
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self.last_call_time = time.time()


# ---------------------------------------------------------------------------
# Model Rotator
# All 7 models confirmed active with Tools support on OpenRouter (May 2026)
# Free tier: 20 req/min AND 200 req/day per model
# 7 models x 200 req/day = 1400 req/day — covers 500 scenarios (~2-3 calls each)
# ---------------------------------------------------------------------------
FREE_MODEL_ROTATION = [
    "qwen/qwen3-coder:free",                          # primary: 480B MoE, 262K ctx
    "google/gemma-4-31b-it:free",                     # fallback 1: Gemma4 31B, 262K
    "google/gemma-4-26b-a4b-it:free",                 # fallback 2: Gemma4 26B, 262K
    "nvidia/nemotron-3-super-120b-a12b:free",         # fallback 3: NVIDIA 120B, 262K
    "openai/gpt-oss-120b:free",                       # fallback 4: GPT-OSS 120B, 131K
    "meta-llama/llama-3.3-70b-instruct:free",         # fallback 5: Llama 3.3 70B, 66K
    "qwen/qwen3-next-80b-a3b-instruct:free",          # fallback 6: Qwen3 Next 80B, 262K
]

DAILY_BUDGET_PER_MODEL = 180  # conservative under 200/day hard limit


class ModelRotator:
    def __init__(self, models: List[str], daily_budget: int = 180):
        self.models = models
        self.daily_budget = daily_budget
        self._index = 0
        self._call_counts = {m: 0 for m in models}
        self._lock = threading.Lock()

    def current(self) -> str:
        return self.models[self._index]

    def record_success(self):
        """Count successful calls; auto-rotate if approaching daily cap."""
        with self._lock:
            m = self.models[self._index]
            self._call_counts[m] += 1
            if self._call_counts[m] >= self.daily_budget:
                self._index = (self._index + 1) % len(self.models)
                self._call_counts[self.models[self._index]] = 0  # reset new model count

    def rotate(self) -> str:
        with self._lock:
            self._index = (self._index + 1) % len(self.models)
            return self.models[self._index]


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
        max_retries: int = 6,
        max_iterations: int = 10,
        verbose: bool = False,
        logger: logging.Logger = None,
        rate_limit_per_minute: int = 15,
    ):
        self.environment = environment
        self.model_url = model_url
        self.model_name = model_name
        self.model_provider = model_provider
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.logger = logger or init_logger()
        self._rate_limiter = RateLimiter(max_per_minute=rate_limit_per_minute)
        self._rotator = ModelRotator(FREE_MODEL_ROTATION, daily_budget=DAILY_BUDGET_PER_MODEL)
        self.client = OpenAI(
            base_url=model_url,
            api_key=API_KEY,
            http_client=httpx.Client(verify=False),
            max_retries=0,  # we handle all retries ourselves — prevents duplicate calls
        )

    def _call_model(self, messages: List[Dict], functions: List[Dict], **kwargs):
        call_kwargs = {"messages": messages, "max_tokens": self.max_tokens, **kwargs}
        if functions:
            call_kwargs["tools"] = functions
            call_kwargs["tool_choice"] = "auto"

        base_wait = 8.0
        consecutive_429s = 0
        models_tried = set()

        for attempt in range(1, self.max_retries + 1):
            self._rate_limiter.wait()
            current_model = self._rotator.current()
            call_kwargs["model"] = current_model

            try:
                response = self.client.chat.completions.create(**call_kwargs)
                if response is None or not response.choices:
                    self.logger.warning(f"Empty response from {current_model} — retrying")
                    time.sleep(base_wait)
                    continue

                self._rotator.record_success()
                consecutive_429s = 0
                return response.choices[0].message

            except NotFoundError:
                models_tried.add(current_model)
                next_model = self._rotator.rotate()
                self.logger.warning(f"404 No endpoints for {current_model} — rotating to {next_model}")
                if len(models_tried) >= len(FREE_MODEL_ROTATION):
                    self.logger.error("All models in rotation returned 404. No active free models.")
                    return None
                time.sleep(1.0)
                continue

            except RateLimitError:
                consecutive_429s += 1
                if consecutive_429s >= 2:
                    next_model = self._rotator.rotate()
                    self.logger.warning(f"429 saturated on {current_model} — rotating to {next_model}")
                    consecutive_429s = 0
                    time.sleep(3.0)
                else:
                    wait = min(base_wait * (2 ** (attempt - 1)) + random.uniform(0, 1), 60.0)
                    self.logger.warning(f"Rate limited on {current_model} (attempt {attempt}/{self.max_retries}). Waiting {wait:.1f}s...")
                    time.sleep(wait)
                if attempt == self.max_retries:
                    self.logger.error(f"Exhausted all {self.max_retries} retries")
                    return None

            except (APIConnectionError, APITimeoutError):
                if attempt == self.max_retries:
                    return None
                time.sleep(min(base_wait * (2 ** (attempt - 1)), 60.0))

            except APIError as exc:
                status = getattr(exc, "status_code", None)
                if status and 400 <= status < 500 and status != 429:
                    next_model = self._rotator.rotate()
                    self.logger.warning(f"HTTP {status} on {current_model} — rotating to {next_model}")
                    time.sleep(1.0)
                    continue
                if attempt == self.max_retries:
                    return None
                time.sleep(min(base_wait * (2 ** (attempt - 1)), 60.0))

            except Exception:
                if self.verbose:
                    self.logger.error(traceback.format_exc())
                return None

        return None

    def _force_answer(self, messages: List[Dict], root_causes: str, is_multi: bool):
        if is_multi:
            constraint = (
                "MULTIPLE-answer question. Select 2-4 options. "
                "Output ONLY \\boxed{C<N>|C<M>} ascending. Example: \\boxed{C3|C7}\n"
                f"Options:\n{root_causes}"
            )
        else:
            constraint = (
                "SINGLE-answer question. Output ONLY \\boxed{C<N>}. Example: \\boxed{C5}\n"
                f"Options:\n{root_causes}"
            )
        messages.append({"role": "user", "content": constraint})
        return self._call_model(messages, functions=[])

    def run(self, scenario: Dict, free_mode: bool = True) -> Dict:
        scenario_id = scenario.get("scenario_id")
        task = scenario.get("task", {})
        options = task.get("options", [])
        root_causes = "".join([f"{item['id']}:{item['label']}\n" for item in options])
        description = task.get("description", "")
        is_multi = "select" in description.lower() and any(
            kw in description.lower() for kw in ["two", "three", "four", "multiple", "2", "3", "4"]
        )

        tool_defs = self.environment.get_tools()
        if not tool_defs:
            self.logger.error(f"[Scenario: {scenario_id}] No tools returned. Is the Tool Server running?")
            return {
                "scenario_id": scenario_id, "status": "unresolved",
                "reason": "No tools available", "answer": "", "traces": "",
                "tool_calls": [], "num_tool_calls": 0, "num_iterations": 0,
            }

        messages: List[Dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": description + f"\nOptions:\n{root_causes}"},
        ]

        num_tool_calls = 0
        list_tool_calls = []
        status = None
        last_msg = None
        i = 0

        for i in range(self.max_iterations):
            self.logger.info(f"[Scenario: {scenario_id}] Round {i + 1} (model: {self._rotator.current()})")
            msg = self._call_model(messages, functions=tool_defs)
            if msg is None:
                self.logger.warning(f"[Scenario: {scenario_id}] _call_model returned None at round {i+1}")
                continue

            last_msg = msg
            messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": msg.tool_calls})

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
                        "turn": i + 1, "order": j + 1,
                        "has_failed": "error" in tool_result,
                        "results": tool_result,
                    })
                if msg.content and extract_answer_all(msg.content):
                    status = "solved"
                    break
            elif msg.content:
                if extract_answer_all(msg.content):
                    status = "solved"
                    break
                elif free_mode:
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

        final_content = getattr(last_msg, "content", "") or ""
        final_traces = getattr(last_msg, "reasoning_content", "") or ""

        if not extract_answer_all(final_content) and free_mode and last_msg is not None:
            self.logger.info(f"[Scenario: {scenario_id}] No boxed answer — forcing final prompt")
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
        self, num_attempts: int, save_dir: str,
        save_freq: int = 5, max_samples: Optional[int] = None,
        free_mode: bool = True,
    ) -> None:
        os.makedirs(save_dir, exist_ok=True)
        completions: List[Dict] = []
        save_result: List[Dict] = []

        scenarios = self.environment.get_scenarios()
        if max_samples is not None:
            scenarios = scenarios[:min(max_samples, len(scenarios))]

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
                    self.logger.info(
                        f"\033[95m[Scenario: {scenario_id}] answer={agent_answer} | "
                        f"gt={ground_truth} | score={score:.2f} | "
                        f"time={elapsed:.1f}s | discount={time_discount(elapsed)}\033[0m"
                    )

            latency = round((time.time() - start_time) / float(num_attempts), 2)
            completions.append({
                "scenario_id": scenario_id,
                "response": sample_response.get("answer", ""),
                "traces": sample_response.get("traces", ""),
                "num_iterations": sample_response.get("num_iterations", 0),
                "num_tool_calls": sample_response.get("num_tool_calls", 0),
                "tool_calls": sample_response.get("tool_calls", []),
                "answers": agent_answers,
                "ground_truth": scenario.get("answer"),
                "accuracy": n_success / float(num_attempts),
                "latency": latency,
            })
            save_result.append({
                "scenario_id": scenario_id,
                "answers": agent_answers[0] if agent_answers else "",
            })

            if ((idx + 1) % save_freq == 0) or ((idx + 1) == len(scenarios)):
                pd.DataFrame(save_result).to_csv(os.path.join(save_dir, "result.csv"), index=False)
                self.logger.info(f"Saved result.csv ({idx + 1}/{len(scenarios)} done)")

        with open(os.path.join(save_dir, "completions.json"), "w", encoding="utf-8") as f:
            json.dump(completions, f, ensure_ascii=False, indent=2)
        self.logger.info("Saved completions.json")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track A — ReAct Agent")
    parser.add_argument("--server_url", type=str, default="http://localhost:7860")
    parser.add_argument("--model_url", type=str, default="https://openrouter.ai/api/v1")
    parser.add_argument("--model_name", type=str, default="qwen/qwen3-coder:free")
    parser.add_argument("--model_provider", type=str, default=None)
    parser.add_argument("--num_attempts", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--save_freq", type=int, default=5)
    parser.add_argument("--max_tokens", type=int, default=16000)
    parser.add_argument("--max_iterations", type=int, default=10)
    parser.add_argument("--save_dir", type=str, default="./results")
    parser.add_argument("--log_file", type=str, default="./log.log")
    parser.add_argument("--rate_limit_per_minute", type=int, default=15)
    parser.add_argument("--no_free_mode", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logger = init_logger(log_file=args.log_file)
    env = Environment(server_url=args.server_url, verbose=args.verbose, logger=logger)

    if not env.health_check():
        logger.error("Aborting: Tool Server is not reachable.")
        sys.exit(1)

    runner = AgentsRunner(
        environment=env,
        model_url=args.model_url,
        model_name=args.model_name,
        model_provider=args.model_provider,
        max_tokens=args.max_tokens,
        max_iterations=args.max_iterations,
        verbose=args.verbose,
        logger=logger,
        rate_limit_per_minute=args.rate_limit_per_minute,
    )

    runner.benchmark(
        max_samples=args.max_samples,
        num_attempts=args.num_attempts,
        save_dir=args.save_dir,
        save_freq=args.save_freq,
        free_mode=not args.no_free_mode,
    )
