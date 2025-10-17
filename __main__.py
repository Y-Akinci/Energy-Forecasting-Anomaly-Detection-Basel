import os, sys
import subprocess

# --- Stelle sicher, dass Projekt-Root korrekt im Pfad ist ---
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# --- Starte das eigentliche Script ---
subprocess.run([sys.executable, "Business_und_Data_Understanding/data_analysis.py"])