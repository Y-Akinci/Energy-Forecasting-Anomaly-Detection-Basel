import os, sys

def add_project_root():
    """Fügt den Projekt-Hauptordner zum Python-Pfad hinzu."""
    # Egal, von wo du startest → gehe IMMER bis zum Projekt-Root hoch
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
