import subprocess
import sys

for script in ["teb_rs2.py", "teb_rh2.py", "teb_uh2.py", "teb_nil.py", "teb_solv.py"]:
    subprocess.run([sys.executable, script], check=True)
