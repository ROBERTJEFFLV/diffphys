#!/usr/bin/env python3
"""Direct entry for the event-constrained task; the shared CLI also routes here."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from diffphys.waypoint_training import main
if __name__ == '__main__':
    main()
