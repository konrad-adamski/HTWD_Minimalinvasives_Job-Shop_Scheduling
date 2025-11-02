# all.py
from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path

# Scripts live in the same folder as this file (00_Problem_Generation/)
SCRIPTS = [
    "00_data_preprocessing.py",
    "01_insert_data_source.py",
    "02_insert_jobs_based_on_max_utilization.py",
    "03_update_jobs_due_dates_and_insert_corresponding_machines.py",
]

def main() -> int:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent              # one level up from 00_Problem_Generation/
    python_exe = sys.executable

    # Ensure the project root is on PYTHONPATH for `import src...`
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        f"{project_root}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)
    )

    print(f"Project root: {project_root}")
    print(f"Script dir:   {script_dir}")

    for i, script in enumerate(SCRIPTS, start=1):
        script_path = script_dir / script
        print(f"\n[{i}/{len(SCRIPTS)}] Running: {script_path.name}")
        # Run with project root as CWD, but execute the script from 00_Problem_Generation/
        result = subprocess.run(
            [python_exe, str(script_path)],
            cwd=str(project_root),
            env=env,
        )
        if result.returncode != 0:
            print(f"Stopped at {script_path.name} (exit code {result.returncode})")
            return result.returncode
        print(f"Finished: {script_path.name}")

    print("\nAll scripts finished successfully.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

