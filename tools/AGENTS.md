# Current training entry

`train_response_control.py` dispatches agile requests to the fixed-point event-constrained trainer. `_hover_training_cli.py` is the byte-identical former entry (Git blob 45db380b0bc868437c836da1875a24366ea452c5) used for legacy hover; do not edit it casually. `train_waypoint_control.py` is a direct alias, not a second algorithm.

No trajectory clock is sent to the event-waypoint Actor. Use configs/agile_flight.args and configs/waypoint_flight.json. The older root guidance on S9 references is historical for this task. See docs/waypoint_training.md for current constraints and verification limitations. Do not launch long training or delegate work without explicit user authorization.
