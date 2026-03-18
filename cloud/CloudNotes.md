# Cloud Experiment Infrastructure

Cloud infrastructure for running Python (TF1) experiments from Hasani et al. 2021 (Table 3).
Each experiment runs on its own self-contained GCP VM that clones the repo, downloads data,
trains the model, uploads results to GCS, and self-deletes.

## Quick Reference

```bash
# Launch a full run (all experiments, all models, 5 seeds, 200 epochs)
./cloud/launch_all_fast.sh full200

# Launch just SRNN with 1 seed
./cloud/launch_all_fast.sh test --models "srnn" --seeds 1 --epochs 20

# Monitor progress
./cloud/monitor.sh full200

# Collect and tabulate results
python3 cloud/collect_results.py full200

# Relaunch any failed/preempted experiments (forces on-demand VMs)
./cloud/relaunch_missing.sh full200

# Dry run (preview without launching)
./cloud/launch_all_fast.sh full200 --dry-run
```

## Architecture

```
laptop / master-launcher VM
         │
         │  gcloud compute instances create
         │  (via launch_all_fast.sh)
         ▼
   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
   │  worker VM   │  │  worker VM   │  │  worker VM   │  ...up to 32
   │  har/lstm/s1 │  │  har/srnn/s1 │  │  gesture/... │
   └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
          │                │                │
          │  startup.sh:   │                │
          │  1. git clone  │                │
          │  2. download   │                │
          │     dataset    │                │
          │  3. train      │                │
          │  4. upload     │                │
          │     results    │                │
          │  5. self-delete │                │
          ▼                ▼                ▼
   ┌─────────────────────────────────────────────┐
   │  GCS: gs://liquidneuralnets-experiments/    │
   │    results-py/<run>/<model>/<exp>/seed<N>/  │
   │      - training_log.txt  (success marker)   │
   │      - <model>_32.csv    (best-epoch stats) │
   │      - run_metadata.json                    │
   └─────────────────────────────────────────────┘
```

## Scripts

### `config.env`
Shared configuration sourced by all scripts. Key settings:
- **GCP_PROJECT** — `liquidneuralnets`
- **GCS_BUCKET** — `gs://liquidneuralnets-experiments`
- **GCP_MACHINE_TYPE** — default `n2-standard-2` (overridden per-experiment in `experiments/*.env`)
- **GCP_USE_SPOT** — `true`/`false` for spot vs on-demand VMs
- **MAX_CONCURRENT_VMS** — concurrency cap (default 32)
- **MODELS** — list of model types to run: `lstm ltc ctrnn ctgru node srnn`

### `experiments/*.env`
Per-experiment config files with training args, machine type overrides, and seed count.
Example (`har.env`):
```
EXPERIMENT_NAME=har
ARGS="--epochs 200 --size 32 --batch_size 128 --ltc_lr_for_srnn --log 1"
MACHINE_TYPE=n2-standard-2
N_SEEDS=5
```

Experiments using `n2-highcpu-4`: power, person, smnist.
All others use `n2-standard-2`.

### `launch_run.sh`
Launches a single experiment VM. Handles:
- Concurrency check (won't exceed MAX_CONCURRENT_VMS)
- Duplicate VM detection (won't relaunch if VM already exists)
- Existing results detection (warns if results already in GCS)
- Spot vs on-demand (reads GCP_USE_SPOT from config.env)

```bash
./cloud/launch_run.sh <run_name> <experiment> <model> <seed> [--epochs N]
```

### `launch_all_fast.sh`
Main launcher — generates the full job matrix and launches in two phases:
1. **Phase 1 (Burst):** Fires VMs in parallel batches of 8 with 15s pauses, up to MAX_CONCURRENT_VMS.
2. **Phase 2 (Sequential):** Polls every 30s for free slots, launches one VM at a time as slots open.

```bash
./cloud/launch_all_fast.sh <run_name> [--seeds N] [--models "m1 m2"] [--epochs N] [--dry-run]
```

### `launch_all.sh`
Original sequential launcher. Superseded by `launch_all_fast.sh` but kept for reference.

### `relaunch_missing.sh`
Scans GCS for experiments missing results and relaunches only those. Key features:
- **Forces on-demand** (`GCP_USE_SPOT=false`) so retries complete reliably
- **Auto-cleans terminated VMs** (from spot preemption) before relaunching
- **Skips running VMs** (won't relaunch in-progress work)
- Same Phase 1/Phase 2 batching as `launch_all_fast.sh`

```bash
./cloud/relaunch_missing.sh <run_name> [--seeds N] [--models "m1 m2"] [--dry-run]
```

### `startup.sh`
Runs on each worker VM at boot (as root). Steps:
1. Read experiment config from VM metadata
2. `git clone` the repo (3 retries)
3. Download dataset from GCS
4. Set up Python venv (or use pre-built image venv)
5. Run training (`python3 <experiment>.py --model <model> --seed <seed> ...`)
6. Upload results (CSVs, training log, metadata JSON) to GCS
7. Self-delete the VM

On failure: uploads error log + error metadata, then self-deletes.

### `monitor.sh`
Dashboard showing vCPU quota, running VMs, and per-experiment completion status.

```bash
./cloud/monitor.sh <run_name>              # full status for a run
./cloud/monitor.sh <run_name> har srnn     # filter to specific experiment/model
./cloud/monitor.sh                         # just show running VMs + quota
```

Legend: ✅ = done, ⏳ = running, · = not started

### `collect_results.py`
Downloads best-epoch results from GCS and produces:
- Terminal table (plain text)
- Markdown table (pandoc-ready, with YAML frontmatter for PDF generation)
- CSV data file

```bash
python3 cloud/collect_results.py <run_name>
python3 cloud/collect_results.py <run_name> --seeds 5 --csv results.csv
```

### `build_image.sh`
Creates the `srnn-python` VM image with pre-installed Python dependencies (TF 2.15, tf_keras, etc.).
Run once when setting up the project or updating dependencies.

## Master Launcher VM

A persistent `e2-medium` VM (`master-launcher`) for running launch scripts, so you don't need
to keep your laptop awake during long launches.

**Specs:** e2-medium (1 vCPU, 4 GB RAM), us-central1-a, ~$24/month always-on.

**Setup (already done):**
```bash
gcloud compute instances create master-launcher \
  --project=liquidneuralnets --zone=us-central1-a \
  --machine-type=e2-medium \
  --image-family=debian-12 --image-project=debian-cloud \
  --boot-disk-size=10GB \
  --scopes=compute-rw,storage-full \
  --metadata=enable-oslogin=true
```

**Usage:**
```bash
# SSH into master
gcloud compute ssh master-launcher --zone=us-central1-a --tunnel-through-iap

# Pull latest code, start tmux, launch
cd liquid_time_constant_networks && git pull
tmux new -s launch
./cloud/launch_all_fast.sh full200
# Ctrl+B, D to detach — disconnect safely

# Reattach later
tmux attach -t launch
```

**Important:** Always `git pull` before launching to get latest experiment scripts.

**Note:** e2-micro (0.25 vCPU, 1 GB) is too small — the `gcloud` CLI is a heavy Python app
and hangs on micro instances. Use e2-medium or larger.

## GCP Quotas

Key quotas in `us-central1` (as of March 2026):

| Quota | Limit | Notes |
|-------|-------|-------|
| N2_CPUS | 200 | Main capacity — 100 × n2-standard-2 VMs |
| E2_CPUS | 24 | 12 × e2-standard-2 VMs |
| INSTANCES | 50 | Max simultaneous VMs (all types) |
| CPUS_ALL_REGIONS | 150 | Global cap — 75 × 2-vCPU VMs |
| PREEMPTIBLE_CPUS | 0 | **No spot quota** — request increase to use spot |
| HDD quota | 500 GB | 32 × 15GB = 480GB (why MAX_CONCURRENT_VMS=32) |

## GCS Results Layout

```
gs://liquidneuralnets-experiments/
├── datasets/               # Shared training data
│   ├── har/
│   ├── gesture/
│   └── ...
└── results-py/
    └── <run_name>/         # e.g. full200
        └── <model>/        # e.g. srnn
            └── <experiment>/  # e.g. har
                └── seed<N>/
                    ├── training_log.txt      # Full console output (success marker)
                    ├── <model>_32.csv        # Best-epoch metrics
                    └── run_metadata.json     # Run config, timing, git commit
```

## Models

| Model | Key | Description |
|-------|-----|-------------|
| LSTM | `lstm` | Baseline LSTM |
| LTC | `ltc` | Liquid Time-Constant (semi-implicit solver) |
| CTRNN | `ctrnn` | Continuous-Time RNN |
| CT-GRU | `ctgru` | Continuous-Time GRU |
| Neural ODE | `node` | Neural ODE |
| SRNN | `srnn` | Structured RNN (Dale's law, 50% E/I) |
| SRNN Per-Neuron | `srnn-per-neuron` | SRNN with per-neuron dynamics params |

Previously available (removed): `ltc_rk` (Runge-Kutta LTC), `ltc_ex` (Explicit LTC).

## Experiments

| Experiment | Type | Dataset | Config |
|-----------|------|---------|--------|
| har | Classification (6 classes) | UCI HAR | `har.env` |
| gesture | Classification (5 classes) | EMG Gesture | `gesture.env` |
| occupancy | Classification (2 classes) | Room Occupancy | `occupancy.env` |
| smnist | Classification (10 classes) | Sequential MNIST | `smnist.env` |
| traffic | Regression | Metro Traffic | `traffic.env` |
| power | Regression | Household Power | `power.env` |
| ozone-fixed | Classification (2 classes) | Ozone Level | `ozone-fixed.env` |
| person | Classification (5 classes) | Person Activity | `person.env` |
| cheetah | Regression (17-dim) | Half-Cheetah | `cheetah.env` |
