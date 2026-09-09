import subprocess
import sys

for script in ["run_geometries.py", "maps_summary.py", "unified_cosmic_anomaly_analysis.py", "skymap_comparison_vs_planck.py", "teb.py"]:
    subprocess.run([sys.executable, script], check=True)
