#!/usr/bin/env python3
"""Single CLI: hover compatibility or fixed-point event-constrained agile training."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
# The former CLI is retained verbatim for the unchanged hover path. Binding this
# blob hash here also binds that implementation into the old source manifest.
HOVER_CLI_BLOB = '45db380b0bc868437c836da1875a24366ea452c5'


def _waypoints(argv):
    parser = argparse.ArgumentParser(add_help=False,fromfile_prefix_chars='@')
    parser.convert_arg_line_to_args = lambda line: line.split() if not line.lstrip().startswith('#') else []
    parser.add_argument('--task')
    parser.add_argument('--checkpoint')
    parser.add_argument('--resume')
    args,_ = parser.parse_known_args(argv)
    if args.task is not None:
        return args.task == 'agile'
    path = args.checkpoint or args.resume
    if path:
        import torch
        value = torch.load(path,map_location='cpu',weights_only=True)
        return isinstance(value,dict) and value.get('schema') == 'event-waypoint-checkpoint-v1'
    return False


def _legacy():
    path = ROOT/'tools/_hover_training_cli.py'
    data = path.read_bytes()
    blob = hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
    if blob != HOVER_CLI_BLOB:
        raise RuntimeError('hover CLI changed; review its source binding before executing')
    spec = importlib.util.spec_from_file_location('_diffphys_hover_cli',path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args(argv=None):
    if _waypoints(argv):
        from diffphys.waypoint_training import parse_args as parse
        return parse(argv)
    return _legacy().parse_args(argv)


def main(argv=None):
    if _waypoints(argv):
        from diffphys.waypoint_training import main as run
        return run(argv)
    return _legacy().main(argv)


if __name__ == '__main__':
    main()
