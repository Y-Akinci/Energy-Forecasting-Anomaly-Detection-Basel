import pandas as pd
import sklearn
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DATA_DIR = os.path.join(ROOT, "data")
f1 = os.path.join(DATA_DIR, "processed_merged_features.csv")


ds = pd.read_csv(f1, sep=";", decimal=",", encoding="latin1", low_memory=False)
ds = ds.set_index("Start der Messung (UTC)")

