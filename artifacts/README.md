# Artifacts directory

Place trained checkpoints, experiment outputs, and profiling logs here. The inference scripts default to loading
`artifacts/checkpoints/model.pt`, so drop your HDMapNet weights (e.g., the `model.pt` produced by `train.py`) in that
path:

```
artifacts/
└── checkpoints/
    └── model.pt  # <-- put your file here
```

Feel free to keep multiple checkpoints (e.g., `model_cam.pt`, `model_fusion.pt`). Use `--modelf <path>` when running
`scripts/run_inference.py` to point to a specific file.
