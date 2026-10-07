import subprocess
import sys

for script in ["1_R3.py", "2_S3.py", "3_H3.py", "4_RS2.py", "5_RH2.py", "6_UH2.py", "7_nil.py", "8_solv.py"]:
    subprocess.run([sys.executable, script], check=True)
