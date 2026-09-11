#!/bin/bash
cd "$(dirname "$0")"
source .venv/bin/activate

# Make sure there are no trailing / for the _BASE_DIR. This crashes the evaluate.py script
KITTI_BASE_DIR=/media/aaron/dataset-ssd/datasets/KITTI-odometry
MULRAN_BASE_DIR=/media/aaron/dataset-ssd/datasets/MulRan
ODYSSEY_BASE_DIR=/media/aaron/dataset-ssd/datasets/odyssey_rev1
OUT_DIR=./estimates

echo "Evaluating KITTI dataset"
python3 compare.py \
    "$KITTI_BASE_DIR/poses/*.txt" kitti_gt GT \
    "$OUT_DIR/ours/kitti/*.txt" kitti Ours \
    "$OUT_DIR/kiss-icp/kitti/*/*_poses_kitti.txt" kiss_on_kitti KISS-ICP \
    "$OUT_DIR/mad-icp/kitti/*/estimate.txt" mad_on_kitti MAD-ICP \
    "$OUT_DIR/genz-icp/kitti/*/*_poses_kitti.txt" kiss_on_kitti GenZ-ICP \
    "$OUT_DIR/kiss-slam/kitti/*/*_poses_kitti.txt" kiss_on_kitti KISS-SLAM \
    --metrics KITTI RPE:100 \
    --fps ours_fps kiss_fps mad_fps kiss_fps kiss_slam_fps

echo "Evaluating MulRan dataset"
python3 compare.py \
    "$MULRAN_BASE_DIR/*/global_pose.csv" mulran_gt GT \
    "$OUT_DIR/ours/mulran/*.txt" kitti Ours \
    "$OUT_DIR/kiss-icp/mulran/*/*_poses_kitti.txt" kitti KISS-ICP \
    "$OUT_DIR/mad-icp/mulran/*/estimate.txt" mad_on_mulran MAD-ICP \
    "$OUT_DIR/genz-icp/mulran/*/*_poses_kitti.txt" kitti GenZ-ICP \
    "$OUT_DIR/kiss-slam/mulran/*/*_poses_kitti.txt" kitti KISS-SLAM \
    --metrics KITTI RPE:100 \
    --fps ours_fps kiss_fps mad_fps kiss_fps kiss_slam_fps \
    --clip # MulRan contains corrupt point clouds on some trajectories which causes MAD-ICP to crash near the end of the trajectory.

echo "Evaluating Odyssey dataset"
python3 compare.py \
    "$ODYSSEY_BASE_DIR/*/refsys/lidar_poses_ref.txt" kitti GT \
    "$OUT_DIR/ours/odyssey/*.txt" kitti Ours \
    "$OUT_DIR/kiss-icp/odyssey/*/*_poses_kitti.txt" kitti KISS-ICP \
    "$OUT_DIR/mad-icp/odyssey/*/estimate.txt" kitti MAD-ICP \
    "$OUT_DIR/genz-icp/odyssey/*/*_poses_kitti.txt" kitti GenZ-ICP \
    "$OUT_DIR/kiss-slam/odyssey/*/*_poses_kitti.txt" kitti KISS-SLAM \
    --metrics KITTI RPE:100 \
    --fps ours_fps kiss_fps mad_fps kiss_fps kiss_slam_fps \
    --exclude Tunnel[1-3] # We exclude Tunnel from our evaluation because all trajectories diverge on it