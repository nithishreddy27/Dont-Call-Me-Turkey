# Task
Predict whether each precomputed audio-feature clip contains a turkey sound.

# Metric
Submissions are scored with Area Under Receiver Operating Characteristic Curve (ROC AUC), where higher is better. Submit one finite numeric probability for `is_turkey` for each clip; values must be in the inclusive range `[0, 1]`. Probabilities are read at standard floating-point precision and are not rounded before scoring.

# Submission Format
Submit a CSV with exactly these columns:

```csv
vid_id,is_turkey
test_clip_000000,0.5
test_clip_000001,0.5
test_clip_000002,0.5
test_clip_000003,0.5
```

# Dataset
`train.json` contains the labeled training clips as a JSON list of records.

- `vid_id`: opaque clip identifier string.
- `start_time_seconds_youtube_clip`: integer start time for the represented clip segment.
- `end_time_seconds_youtube_clip`: integer end time for the represented clip segment.
- `audio_embedding`: variable-length list of audio frames; each frame is a list of 128 integer embedding values.
- `is_turkey`: binary target, `1` for clips containing a turkey sound and `0` otherwise.

`test.json` contains the unlabeled clips to predict, with the same columns as `train.json` except `is_turkey` is omitted.

`sample_submission.csv` contains the required submission columns and the test `vid_id` values in the expected order.
