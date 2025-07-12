from pathlib import Path
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns


def set_project_root():
    """
    Sets the project root directory as the current working directory.

    If the current working directory is "notebooks", changes the working directory to its parent (assumed to be the project root).
    Otherwise, keeps the current directory as the project root.
    Also appends the "src" directory within the project root to the Python module search path (sys.path).

    Prints the new current working directory.

    Returns:
        None
    """
    current_path = Path.cwd()
    if current_path.name == "notebooks":
        project_root = current_path.parent
    else:
        project_root = current_path

    os.chdir(project_root)
    print(f"Current working directory: {Path.cwd()}")
    sys.path.append(str(project_root / "src"))


plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")