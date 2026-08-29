# Status

**Last updated: 2026-08-28.** This file is a snapshot, not a source of truth —
if it disagrees with `wandb`, the remote server, or `git log`, trust those and
update this file.

## Current: Stage 1 continued pretraining

Running on the remote GPU server (`kabidi@41.226.183.241:2224`, single NVIDIA
GB10 unified-memory dev board — `nvidia-smi` reports memory as "Not
Supported", GPU+CPU share the 119GB system RAM pool). Training in a tmux
session named `kothar-pretrain`, logging to
`~/work/rnd/eshmun/logs/kothar-pretrain-409m.log`, tracked on Weights & Biases
as `flursky/Kothar/ltiumvoj` ("kothar-pretrain-409m").

As of this writing: **step ~3060/10500** (~29%), effective batch size 192
(`per_device_batch_size=24 × grad_accum=8 × 1 GPU`), ~74s/step, loss ≈3.5 and
slowly declining, checkpointing every 250 steps to
`~/work/rnd/eshmun/checkpoints/kothar-pretrain-409m/checkpoint-N`
(`save_total_limit=3`, so only the 3 most recent survive on disk).
**1,050 steps/epoch** (confirmed from the live log: step 3050 was logged at
`epoch=2.905`, and 3050/2.905 = 1050.0 exactly) — not the 6,296 an earlier
version of this doc mistakenly carried over from the botched first resume
attempt (wrong batch size).

## Planned early stop: epoch 5 / step 5250 (decided 2026-08-28)

Training will be stopped at **step 5250** (= epoch 5.0 exactly, at 1,050
steps/epoch), not the full planned 10 epochs / 10,500 steps. Implemented as
an automated watcher (a persistent background poll on this machine, checking
every 5 minutes for `checkpoints/kothar-pretrain-409m/checkpoint-5250/trainer_state.json`
on the remote server, then `tmux kill-session -t kothar-pretrain`) rather than
restarting the run with `--max-steps 5250` — see
[`kothar-pipeline.md`](kothar-pipeline.md#stage-1--continued-pretraining-pretrainpy-in-progress)
for why: `--max-steps` on resume rebuilds the cosine LR schedule around the
new, smaller total, silently distorting the LR trajectory. This way the
original 10,500-step schedule stays intact and training just stops partway
through it once checkpoint-5250 is confirmed written to disk.

**Caveat**: the watcher only survives as long as the Claude Code session/process
that armed it stays alive — a session restart before step 5250 (this has
already happened once earlier in the same conversation, dropping an
unrelated log-tailing monitor that had to be manually re-armed) would silently
kill the watcher without stopping training. Treat "training progress" checks
between now and step 5250 as a safety net, not just a status query — if the
watcher died, someone needs to notice and either re-arm it or stop training
manually once past step 5250.

## Incident (2026-08-27→28): server reboot, crash, and resume

The server rebooted (cause not investigated — outside this project's control),
killing the training tmux session mid-run. Diagnosed via wandb: run `ltiumvoj`
last heartbeat 2026-08-27 22:30 UTC, last logged `global_step=2750`
(loss=3.573), but the last checkpoint actually saved to disk was
`checkpoint-2500` (steps 2500–2750 were lost — hadn't hit a save interval yet
at `save_steps=500`).

Two mistakes were caught and corrected while resuming (worth keeping in mind
for any future resume, since neither is checked automatically):

1. **`save_steps` reduced 500→250** going forward (per explicit decision, to
   reduce future crash exposure) — requires passing `--save-steps 250` on
   resume *and* the `SyncStateIntervalsCallback` in `pretrain.py` (it exists
   precisely because `Trainer` otherwise silently restores the old value from
   `trainer_state.json` on resume, ignoring the CLI override). The script's
   own `--save-steps` default was also changed 500→250 so a bare resume gets
   the new cadence even without remembering the flag.
2. **`per_device_batch_size` must be passed explicitly on resume.** The first
   resume attempt used the script's default (4, effective batch 32) instead of
   the value the run was actually trained with (24, effective batch 192) —
   `Trainer` logs a mismatch warning against `trainer_state.json` but doesn't
   enforce or auto-correct it the way `SyncStateIntervalsCallback` does for
   the step intervals. Caught by reading that warning in the resume log
   before it ran for more than a few steps; corrected by killing and
   relaunching with `--per-device-batch-size 24` explicit. **Any future
   resume of this run needs `--per-device-batch-size 24` passed explicitly.**
3. **wandb run continuity**: `pretrain.py`'s `WandbCallback` (stock HF
   integration) calls `wandb.init()` with no run id, so a bare resume creates
   a *new* wandb run (same display name, different run id) rather than
   continuing `ltiumvoj`. Fixed by exporting `WANDB_RUN_ID=ltiumvoj
   WANDB_RESUME=must` before relaunching. Confirmed working: resumed log shows
   `wandb: Resuming run kothar-pretrain-409m`. **Any future resume needs these
   two env vars set**, or the run history in `experiments.md` will fragment
   across multiple wandb run ids.

Throughput sanity-checked against pre-crash history (wandb `_timestamp` deltas
for steps 2510→2610, the last ~100 steps before the crash): ~73.3s/step then,
~74s/step now — the post-resume pace matches, so the reboot didn't leave the
GPU in a degraded state.

## Open items

- Stage 1 has not plateaued yet through step 3000 (see
  [`experiments.md`](experiments.md#stage-1-validation-perplexity-trend)) —
  the run will still be stopped early at step 5250 per the deliberate
  decision above, not because of a plateau. Re-run `checkpoint_trend.py` and
  the generation/pLDDT pilot on checkpoint-5250 once it lands, and decide
  from there whether Stage 1 output is good enough to move to Stage 2 or
  whether more pretraining is warranted first.
- Stage 2 (thinking-aware SFT, conditions a/b) has no training code yet —
  `ROADMAP.md` Phase 3 tracks this.
- `ROADMAP.md` was rewritten 2026-08-28 to match the actual Kothar phases
  and script paths (previously described a pre-Kothar plan and pointed at
  `trainer/sft/`/`trainer/grpo/`, which don't exist in the current tree).
