train.yaml settings guide

run
  run_name: Run identifier used in logs and saved snapshots.
  save_dir: Directory to save checkpoints and config snapshots.
  device: Training device string (e.g. cuda, cpu).
  seed: Random seed for reproducibility.

data
  T: Number of samples per signal (sequence length).
  train_n: Number of training samples (dummy dataset in train.py).
  val_n: Number of validation samples (dummy dataset in train.py).
  batch_size: Batch size for data loaders.
  num_workers: DataLoader worker processes.
  zarr/events_per_sample: Zarr only. Number of events per sample; can be 2, 3, or a list like [2, 3] to mix.
  zarr/source_fractions: Zarr only. Optional per-source sampling fraction map, e.g. {shift: 0.5}.
  persistent_workers: DataLoader only. Keep worker processes alive between epochs.
  prefetch_factor: DataLoader only. Batches prefetched per worker (PyTorch default is 2).

train
  epochs: Total training epochs.
  lr: Learning rate.
  wd: Weight decay for optimizer.
  amp: Enable automatic mixed precision.
  grad_clip: Gradient clipping max norm.
  log_every: Log interval in epochs (metrics CSV only).
  log_csv: Write metrics to a CSV file at log_every epochs.
  csv_path: Optional CSV file path. If null, uses save_dir/run_name_metrics.csv.
  sched: Scheduler type (plateau or none).
  sched_patience: ReduceLROnPlateau patience in epochs (used when sched=plateau).
  early_stop_patience: Early stopping patience in epochs. Use 0 to disable.

loss
  type: Loss type. Use bce (default) or focal.
  wP: Loss weight for P channel.
  wS: Loss weight for S channel.
  wE: Loss weight for event channel.
  focal/alpha: Focal alpha for [P, S, event], each in [0, 1].
  focal/gamma: Focusing factor gamma (>= 0), larger emphasizes hard samples.
  focal/eps: Small epsilon added inside log for numerical stability.

metrics
  thr_p: Peak threshold for P predictions.
  thr_s: Peak threshold for S predictions.
  thr_event: Threshold for event interval predictions.
  tol_p: Tolerance (samples) for P peak matching.
  tol_s: Tolerance (samples) for S peak matching.
  iou_thr: IoU threshold for event interval matching.
  merge_gap_event: Merge predicted event intervals if gap <= this value.

model
  d_model: Model hidden dimension.
  d_state: Mamba state dimension.
  core_mamba_nums: Number of core Mamba blocks.
  decode_mamba_nums: Number of decode Mamba blocks.
  performer_heads: Performer attention heads.

model.cnn
  c0/c1/c2: CNN channel sizes for each stage.
  k: CNN kernel size.
  norm: CNN normalization (gn, bn, none).
  act: CNN activation (silu, relu, gelu).
