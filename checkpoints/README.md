# Checkpoints

Place a local Orbax directory here. Git ignores the weights; only this README is tracked.

A valid directory must contain:

- `_CHECKPOINT_METADATA`
- `params/`
- `assets/<asset_id>/norm_stats.json` (this repo’s default `asset_id` is `training_dataset`)

Layouts that `./scripts/serve.sh` can auto-discover:

```
checkpoints/                          # flat
checkpoints/pi0_aloha/<exp>/<step>/   # nested; the largest numeric step wins
```

Do not commit `params/`, `*.tar`, or other weight archives.
