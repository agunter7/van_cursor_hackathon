#!/usr/bin/env python3
"""
Streamlit demo UI for the FPGA DCP optimizer.

Runs: python dcp_optimizer.py <input.dcp> [--model ...] [options]
Streams combined stdout/stderr live, then shows a summary dashboard from parsed output.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
OPTIMIZER = PROJECT_ROOT / "dcp_optimizer.py"
DEFAULT_MODEL = "x-ai/grok-4.1-fast"


def build_command(
    input_dcp: Path,
    model: str,
    *,
    output_dcp: Optional[Path],
    debug: bool,
    test_mode: bool,
    max_nets: int,
    prompt_file: str,
) -> list[str]:
    cmd = [
        sys.executable,
        str(OPTIMIZER),
        str(input_dcp.resolve()),
        "--model",
        model.strip(),
    ]
    if output_dcp and str(output_dcp).strip():
        cmd += ["--output", str(Path(output_dcp).expanduser().resolve())]
    if debug:
        cmd.append("--debug")
    if test_mode:
        cmd.append("--test")
        if max_nets != 5:
            cmd += ["--max-nets", str(max_nets)]
    if prompt_file.strip():
        cmd += ["--prompt", prompt_file.strip()]
    return cmd


def parse_run_summary(log: str) -> dict:
    """Extract paths, timing, and outcome from optimizer stdout."""
    d: dict = {
        "input": None,
        "output": None,
        "run_dir": None,
        "model": None,
        "initial_wns": None,
        "final_wns": None,
        "wns_change": None,
        "initial_fmax": None,
        "final_fmax": None,
        "fmax_change": None,
        "clock_period_ns": None,
        "target_fmax": None,
        "total_runtime_s": None,
        "success": None,
        "outcome_line": None,
        "test_mode": "TEST MODE" in log or "[TEST]" in log,
    }

    for pattern, key in [
        (r"Input:\s+(\S.+)", "input"),
        (r"Output:\s+(\S.+)", "output"),
        (r"Run dir:\s+(\S.+)", "run_dir"),
        (r"Model:\s+(\S.+)", "model"),
    ]:
        m = re.search(pattern, log)
        if m:
            d[key] = m.group(1).strip()

    def _float(m: Optional[re.Match]) -> Optional[float]:
        if not m:
            return None
        try:
            return float(m.group(1))
        except ValueError:
            return None

    m = re.search(r"\*\*\* WNS Change:\s*([+-]?\d+\.?\d*)\s*ns\s*\*\*\*", log)
    if m:
        d["wns_change"] = _float(m)

    m = re.search(r"\*\*\* Fmax Change:\s*([+-]?\d+\.?\d*)\s*MHz\s*\*\*\*", log)
    if m:
        d["fmax_change"] = _float(m)

    # Timing Results block (test summary and similar)
    m = re.search(r"Initial WNS:\s*([+-]?\d+\.?\d*)\s*ns", log)
    if m:
        d["initial_wns"] = _float(m)
    m = re.search(r"Final WNS:\s+([+-]?\d+\.?\d*)\s*ns", log)
    if m:
        d["final_wns"] = _float(m)

    m = re.search(r"Initial fmax:\s*([+-]?\d+\.?\d*)\s*MHz", log)
    if m:
        d["initial_fmax"] = _float(m)
    m = re.search(r"Final fmax:\s+([+-]?\d+\.?\d*)\s*MHz", log)
    if m:
        d["final_fmax"] = _float(m)

    m = re.search(r"WNS Change:\s+([+-]?\d+\.?\d*)\s*ns", log)
    if m and d["wns_change"] is None:
        d["wns_change"] = _float(m)

    m = re.search(r"Fmax Change:\s*([+-]?\d+\.?\d*)\s*MHz", log)
    if m and d["fmax_change"] is None:
        d["fmax_change"] = _float(m)

    m = re.search(r"Clock period:\s*([+-]?\d+\.?\d*)\s*ns\s*\(target fmax:\s*([+-]?\d+\.?\d*)\s*MHz\)", log)
    if m:
        d["clock_period_ns"] = _float(m)
        try:
            d["target_fmax"] = float(m.group(2))
        except ValueError:
            pass

    m = re.search(r"Total runtime:\s*([+-]?\d+\.?\d*)\s*seconds", log)
    if m:
        d["total_runtime_s"] = _float(m)

    if re.search(r"✓ Optimization completed successfully", log):
        d["success"] = True
        d["outcome_line"] = "Optimization completed successfully"
    elif re.search(r"✗ Optimization did not complete successfully", log):
        d["success"] = False
        d["outcome_line"] = "Optimization did not complete successfully"
    elif re.search(r"\[TEST\]\s*Test completed successfully", log):
        d["success"] = True
        d["outcome_line"] = "Test mode completed successfully"
    elif re.search(r"\[TEST\].*completed successfully", log, re.I):
        d["success"] = True

    if d["success"] is None:
        if "IMPROVEMENT:" in log or "REGRESSION:" in log or "NO CHANGE:" in log:
            if "REGRESSION:" in log:
                d["outcome_line"] = "Finished with timing regression"
            elif "IMPROVEMENT:" in log:
                d["outcome_line"] = "Finished with timing improvement"
            else:
                d["outcome_line"] = "Finished (no WNS change)"

    return d


def run_optimizer_streaming(cmd: list[str], env: dict) -> tuple[str, int]:
    """Run optimizer; return full log text and exit code. Updates Streamlit live."""
    log_lines: list[str] = []
    log_box = st.empty()

    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    assert proc.stdout is not None
    for line in iter(proc.stdout.readline, ""):
        log_lines.append(line)
        full = "".join(log_lines)
        if len(full) > 400_000:
            full = "… (truncated for display)\n" + full[-350_000:]
        log_box.code(full, language="text")
    proc.wait()
    full_log = "".join(log_lines)
    log_box.code(full_log if len(full_log) <= 400_000 else "… (truncated)\n" + full_log[-350_000:], language="text")
    return full_log, proc.returncode or 0


def main():
    st.set_page_config(
        page_title="FPGA DCP Optimizer",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("FPGA DCP optimization demo")
    st.caption("Runs `dcp_optimizer.py` in the project root with a live log and post-run summary.")

    with st.sidebar:
        st.header("Run configuration")
        input_dcp = st.text_input(
            "Input `.dcp` path",
            placeholder="/path/to/design.dcp",
            help="Absolute or relative path; relative paths are resolved from the project root.",
        )
        output_dcp = st.text_input(
            "Output `.dcp` (optional)",
            placeholder="Leave empty for default timestamped name next to input",
        )
        model = st.text_input("Model (`--model`)", value=DEFAULT_MODEL)
        test_mode = st.toggle("Test mode (`--test`, no LLM)", value=False)
        max_nets = st.number_input("Max nets (test mode)", min_value=1, max_value=100, value=5)
        prompt_file = st.text_input("Prompt file in PROMPTS/", value="SYSTEM_PROMPT.TXT")
        debug = st.toggle("Debug (`--debug`)", value=False)

        st.divider()
        st.subheader("API key")
        st.caption("Required for agent mode. Inherited from the environment if unset below.")
        api_key_input = st.text_input(
            "OpenRouter API key",
            type="password",
            placeholder="Leave blank to use OPENROUTER_API_KEY from the environment",
        )

        st.divider()
        with st.expander("Equivalent command"):
            st.code(
                "cd " + str(PROJECT_ROOT) + "\n"
                + " ".join(
                    [
                        "python3",
                        "dcp_optimizer.py",
                        "<input.dcp>",
                        "--model",
                        model.strip() or DEFAULT_MODEL,
                    ]
                )
                + (" --test" if test_mode else "")
                + (" --debug" if debug else "")
                + (f" --prompt {prompt_file}" if prompt_file.strip() else ""),
                language="bash",
            )

    col_run, col_clear = st.columns([1, 4])
    with col_run:
        run_clicked = st.button("Run optimization", type="primary", use_container_width=True)
    with col_clear:
        if st.button("Clear last results"):
            for k in ("last_log", "last_cmd", "last_exit", "last_summary"):
                st.session_state.pop(k, None)
            st.rerun()

    if not run_clicked:
        if "last_summary" in st.session_state:
            st.subheader("Last run — dashboard")
            _render_dashboard(
                st.session_state["last_summary"],
                st.session_state.get("last_exit", -1),
                st.session_state.get("last_log", ""),
                st.session_state.get("last_cmd", []),
            )
        else:
            st.info("Configure the sidebar and click **Run optimization**.")
        return

    if not input_dcp.strip():
        st.error("Set **Input `.dcp` path** in the sidebar.")
        return

    in_path = Path(input_dcp.strip()).expanduser()
    if not in_path.is_absolute():
        in_path = (PROJECT_ROOT / in_path).resolve()
    if not in_path.exists():
        st.error(f"Input file not found: {in_path}")
        return

    out_path: Optional[Path] = None
    if output_dcp.strip():
        out_path = Path(output_dcp.strip()).expanduser()
        if not out_path.is_absolute():
            out_path = (PROJECT_ROOT / out_path).resolve()

    cmd = build_command(
        in_path,
        model,
        output_dcp=out_path,
        debug=debug,
        test_mode=test_mode,
        max_nets=int(max_nets),
        prompt_file=prompt_file,
    )

    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    if api_key_input.strip():
        env["OPENROUTER_API_KEY"] = api_key_input.strip()

    st.subheader("Live output")
    st.code(" ".join(cmd), language="bash")

    with st.status("Running optimizer…", expanded=True) as status:
        log_text, exit_code = run_optimizer_streaming(cmd, env)
        if exit_code == 0:
            status.update(label="Run finished (exit 0)", state="complete")
        else:
            status.update(label=f"Run finished (exit {exit_code})", state="error")

    summary = parse_run_summary(log_text)
    st.session_state["last_log"] = log_text
    st.session_state["last_cmd"] = cmd
    st.session_state["last_exit"] = exit_code
    st.session_state["last_summary"] = summary

    st.divider()
    st.subheader("Run dashboard")
    _render_dashboard(summary, exit_code, log_text, cmd)


def _render_dashboard(summary: dict, exit_code: int, log_text: str, cmd: list[str]) -> None:
    if summary.get("success") is True:
        st.success(summary.get("outcome_line") or "Completed successfully")
    elif summary.get("success") is False:
        st.error(summary.get("outcome_line") or "Did not complete successfully")
    elif exit_code != 0:
        st.error(f"Process exited with code {exit_code}")
    else:
        st.warning(summary.get("outcome_line") or "See log for details; outcome could not be parsed.")

    mcols = st.columns(4)
    with mcols[0]:
        st.metric("Exit code", str(exit_code))
    with mcols[1]:
        if summary.get("total_runtime_s") is not None:
            st.metric("Runtime", f"{summary['total_runtime_s']:.1f} s")
        else:
            st.metric("Runtime", "—")
    with mcols[2]:
        st.metric("Mode", "Test" if summary.get("test_mode") else "Agent")
    with mcols[3]:
        if summary.get("model"):
            st.metric("Model", summary["model"][:24] + ("…" if len(summary["model"]) > 24 else ""))
        else:
            st.metric("Model", "—")

    st.divider()
    pcols = st.columns(2)
    with pcols[0]:
        st.markdown("**Paths**")
        st.text_input("Input DCP", value=summary.get("input") or "—", disabled=True, key="dash_in")
        st.text_input("Output DCP", value=summary.get("output") or "—", disabled=True, key="dash_out")
    with pcols[1]:
        st.markdown("**Run directory**")
        st.text_input("Run dir", value=summary.get("run_dir") or "—", disabled=True, key="dash_run")

    st.divider()
    st.markdown("**Timing**")
    tcols = st.columns(4)
    iw, fw = summary.get("initial_wns"), summary.get("final_wns")
    with tcols[0]:
        st.metric("Initial WNS", f"{iw:.3f} ns" if iw is not None else "—")
    with tcols[1]:
        st.metric("Final WNS", f"{fw:.3f} ns" if fw is not None else "—")
    with tcols[2]:
        wc = summary.get("wns_change")
        if wc is not None:
            st.metric("WNS Δ", f"{wc:+.3f} ns")
        else:
            st.metric("WNS Δ", "—")
    with tcols[3]:
        if summary.get("clock_period_ns") is not None:
            tf = summary.get("target_fmax")
            extra = f" (target {tf:.1f} MHz)" if tf is not None else ""
            st.metric("Clock period", f"{summary['clock_period_ns']:.3f} ns{extra}")
        else:
            st.metric("Clock period", "—")

    fcols = st.columns(3)
    with fcols[0]:
        ifm = summary.get("initial_fmax")
        st.metric("Initial fmax", f"{ifm:.2f} MHz" if ifm is not None else "—")
    with fcols[1]:
        ffm = summary.get("final_fmax")
        st.metric("Final fmax", f"{ffm:.2f} MHz" if ffm is not None else "—")
    with fcols[2]:
        fc = summary.get("fmax_change")
        st.metric("Fmax Δ", f"{fc:+.2f} MHz" if fc is not None else "—")

    if iw is not None and fw is not None:
        try:
            import pandas as pd

            st.bar_chart(
                pd.DataFrame({"WNS (ns)": [iw, fw]}, index=["Initial", "Final"])
            )
        except ImportError:
            st.caption("Install pandas (included with Streamlit) for the WNS chart.")

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Download full log",
            data=log_text,
            file_name="dcp_optimizer_run.log",
            mime="text/plain",
            use_container_width=True,
        )
    with c2:
        st.download_button(
            "Download command (shell)",
            data="#!/bin/bash\nset -euo pipefail\ncd " + str(PROJECT_ROOT) + "\nexec " + " ".join(cmd) + "\n",
            file_name="rerun_demo.sh",
            mime="text/x-shellscript",
            use_container_width=True,
        )

    with st.expander("Full log"):
        st.code(log_text, language="text")


if __name__ == "__main__":
    main()
