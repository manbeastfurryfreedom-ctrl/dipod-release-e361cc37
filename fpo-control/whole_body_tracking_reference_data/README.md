# Whole-Body Tracking Reference Data

This directory contains processed motion capture NPZ files used for G1 humanoid
whole-body motion tracking training.

Generate these files explicitly after the control environment is set up:

```bash
python whole_body_tracking_reference_data/download_lafan_data.py --headless
```

The script downloads LAFAN1 CSV files from HuggingFace, runs
them through IsaacSim forward kinematics to compute body poses, and saves the
results here.

## Available motions

| File | Description |
|------|-------------|
| `walk1_subject1.npz` | Walking motion |
| `run1_subject2.npz` | Running motion |
| `dance1_subject1.npz` | Dance motion (subject 1) |
| `dance1_subject2.npz` | Dance motion (subject 2) |
| `fight1_subject2.npz` | Fighting motion |
| `jumps1_subject1.npz` | Jumping motion |
| `fallAndGetUp1_subject1.npz` | Fall and get up motion |

## NPZ file format

Each file contains:
- `fps` — frame rate (50 FPS)
- `joint_pos` — joint positions `(T, 29)`
- `joint_vel` — joint velocities `(T, 29)`
- `body_pos_w` — body positions in world frame `(T, 30, 3)`
- `body_quat_w` — body quaternions in world frame `(T, 30, 4)`
- `body_lin_vel_w` — body linear velocities `(T, 30, 3)`
- `body_ang_vel_w` — body angular velocities `(T, 30, 3)`

## Source

CSV data from [LAFAN1 Retargeting Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset) (G1 robot).
