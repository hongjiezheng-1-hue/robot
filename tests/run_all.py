"""Run every test file in this folder and summarise. Usage: python tests/run_all.py

Each file is run as its own process (the files are plain-assert scripts), so a failure in one does not hide the others.
"""

import pathlib
import subprocess
import sys

folder = pathlib.Path(__file__).resolve().parent
failed = []
for path in sorted(folder.glob("test_*.py")):
    result = subprocess.run([sys.executable, str(path)], capture_output=True, text=True)
    passed = sum(1 for line in result.stdout.splitlines() if line.startswith("ok "))
    status = "ok" if result.returncode == 0 else "FAILED"
    print(f"{path.name:36s} {passed:2d} passed  {status}")
    if result.returncode != 0:
        failed.append(path.name)
        print(result.stdout + result.stderr)
print("all tests passed" if not failed else f"failures in: {', '.join(failed)}")
sys.exit(1 if failed else 0)
