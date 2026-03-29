# reasoning/konclude_runner.py
"""
Konclude reasoner runner.

Executes Konclude binary, parses output, and returns a normalized report dict.
"""
from reasoning.axiom_extractor import extract_inconsistency_explanation, extract_all_inconsistencies

import subprocess
import time
import uuid
from pathlib import Path
from typing import Dict, Any

import config
from reasoning.parse_reasoner_stats import parse_stats_file


def _detect_konclude_exec() -> Path:
    """Auto-detect Konclude executable in reasoning/konclude/Binaries/."""
    candidates = [
        config.KONCLUDE_DIR / "Binaries" / "Konclude.exe",
        config.KONCLUDE_DIR / "Konclude.exe",
        config.KONCLUDE_DIR / "Konclude.bat",
    ]
    for p in candidates:
        p = p.resolve()
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"Konclude executable not found in {config.KONCLUDE_DIR}. "
        f"Searched: {[str(c) for c in candidates]}"
    )


def _run_cmd(cmd_list: list) -> tuple[int, str, str]:
    """Execute subprocess and return (returncode, stdout, stderr)."""
    proc = subprocess.run(
        cmd_list,
        capture_output=True,
        text=True,
        shell=False
    )
    return proc.returncode, proc.stdout, proc.stderr


def _choose_command(konclude_exec: Path) -> str:
    """Auto-detect whether Konclude uses 'classification' or 'classify'."""
    rc, so, se = _run_cmd([str(konclude_exec), "--help"])
    low = (so + se).lower()
    if "classification" in low:
        return "classification"
    if "classify" in low:
        return "classify"
    return "classification"  # default fallback


def run(
    input_owl: str,
    out_dir: str = None,
    command: str = None,
    write_stats: bool = True,
) -> Dict[str, Any]:
    """
    Run Konclude reasoner and return a normalized report.

    Args:
        input_owl:   Path to input OWL file.
        out_dir:     Output directory for reasoned OWL + stats.
                     Defaults to config.REASONER_OUT_DIR.
        command:     Konclude command ("classification" or "classify").
                     Auto-detected if None.
        write_stats: Whether to request --write-statistics from Konclude.

    Returns:
        Normalized report dict with schema:
        {
            "is_consistent":       bool,
            "unsat_classes":       list[str],
            "disjoint_violations": list[tuple],
            "raw_logs_path":       str,
            "stats": {
                "reasoner":         "konclude",
                "duration_seconds": float,
                "triple_count":     int,  # not available from Konclude, set to 0
            }
        }

    Raises:
        FileNotFoundError: If input_owl does not exist or Konclude binary not found.
    """
    start_time = time.perf_counter()

    in_path = Path(input_owl).resolve()
    if not in_path.is_file():
        raise FileNotFoundError(f"Input OWL not found: {in_path}")

    konclude_exec = _detect_konclude_exec()

    if out_dir is None:
        out_dir = str(config.REASONER_OUT_DIR)
    out_dir_path = Path(out_dir).resolve()
    out_dir_path.mkdir(parents=True, exist_ok=True)

    if command is None:
        command = _choose_command(konclude_exec)

    # Generate unique output filenames
    uid = uuid.uuid4().hex[:8]
    out_owl = out_dir_path / f"konclude_reasoned_{uid}.owl"
    stats_path = out_dir_path / f"konclude_stats_{uid}.txt"
    raw_log_path = out_dir_path / f"konclude_raw_{uid}.log"

    # Build Konclude command
    cmd = [
        str(konclude_exec),
        command,
        "-i", str(in_path),
        "-o", str(out_owl),
    ]
    if write_stats:
        cmd.extend(["--write-statistics", str(stats_path)])

    print(f"[Konclude] Running: {' '.join(cmd)}")
    rc, stdout, stderr = _run_cmd(cmd)

    # Retry with alternate command if first attempt fails
    if rc != 0 and command in ("classification", "classify"):
        alt = "classify" if command == "classification" else "classification"
        print(f"[Konclude] Retrying with '{alt}' command...")
        cmd[1] = alt
        rc, stdout, stderr = _run_cmd(cmd)

    # Save raw logs for debugging
    with open(raw_log_path, "w", encoding="utf-8") as fh:
        fh.write("=== STDOUT ===\n")
        fh.write(stdout)
        fh.write("\n=== STDERR ===\n")
        fh.write(stderr)

    if rc != 0:
        print(f"[Konclude] ERROR (exit code {rc})")
        if stderr:
            print(stderr)

    # Parse stats
    stats_path_str = str(stats_path) if (
        write_stats and stats_path.is_file()) else None
    fallback_text = stdout + "\n" + stderr if not stats_path_str else None

    parsed = parse_stats_file(stats_path_str, fallback_text)

    elapsed = time.perf_counter() - start_time

    # --- Extract ALL evidence for inconsistencies ---
    evidence = None
    all_issues = None
    if parsed.get("consistency") is False:
        # Comprehensive scan: find every violation
        all_issues = extract_all_inconsistencies(str(in_path))

        if all_issues and all_issues["total_violations"] > 0:
            evidence = all_issues["primary_evidence"]

            # Populate disjoint_violations from all found violations
            dv_list = [v for v in all_issues["all_violations"]
                       if v["error_type"] in ("disjoint_violation", "transitive_disjoint_violation")]
            if dv_list and not parsed.get("disjoint_violations"):
                parsed["disjoint_violations"] = [
                    (v["entity"], v.get("classes", ["", ""])
                     [0], v.get("classes", ["", ""])[1])
                    for v in dv_list
                ]

            # Populate unsat_classes from evidence when available
            for v in all_issues["all_violations"]:
                if v["error_type"] == "unsat_class":
                    iri = v.get("entity_iri") or v.get("entity")
                    if iri and iri not in parsed["unsat_classes"]:
                        parsed["unsat_classes"].append(iri)

            print(f"\n[Konclude] === INCONSISTENCY REPORT ===")
            print(all_issues["summary_text"])
            print(f"[Konclude] === END REPORT ===\n")
        else:
            # Fall back to single-evidence extraction
            expl = extract_inconsistency_explanation(str(in_path))
            if expl:
                evidence = expl
                print(f"[Konclude] Evidence extracted: type={expl.get('error_type', '?')}, "
                      f"entity={expl.get('entity_label', expl.get('entity', '?'))}")

    # Build normalized report
    report = {
        "is_consistent":       parsed["consistency"],
        "unsat_classes":       parsed["unsat_classes"],
        "disjoint_violations": parsed["disjoint_violations"],
        "raw_logs_path":       str(raw_log_path),
        "stats": {
            "reasoner":         "konclude",
            "duration_seconds": elapsed,
            "triple_count":     0,  # Konclude doesn't report this
        },
    }

    # Include full evidence dict so callers can see the actual cause
    if evidence:
        report["evidence"] = evidence
    # Include comprehensive report
    if all_issues:
        report["all_issues"] = all_issues

    return report
