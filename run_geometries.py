import subprocess
import sys

for script in ["RS2.py", "RH2.py", "UH2.py", "nil.py", "solv.py"]:
    subprocess.run([sys.executable, script], check=True)
