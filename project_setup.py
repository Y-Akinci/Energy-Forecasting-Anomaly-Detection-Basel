import os, sys

def add_project_root():
    """Fügt den Projekt-Hauptordner zum Python-Pfad hinzu."""
    root_dir = os.path.dirname(os.path.abspath(__file__))
    if root_dir not in sys.path:
        sys.path.append(root_dir)
