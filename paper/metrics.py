# Copyright (c) 2026 Aaron Kurda
# SPDX-License-Identifier: MIT
import numpy as np
import bisect

def RMSE(errors):
    return np.sqrt(np.mean(np.square(errors)))

def calc_error(E,relation="translation"):
    if relation=="translation":
        error = np.linalg.norm(E[:3,-1])
    if relation=="full":
        error = np.linalg.norm(E - np.eye(4))
    return error


def APE(gt_poses, es_poses,relation="translation"):
    assert gt_poses.shape == es_poses.shape
    errors = np.array([np.linalg.inv(gt_poses[i]) @ es_poses[i] for i in range(len(gt_poses))])
    errors = np.array([calc_error(e,relation) for e in errors])
    return np.array(errors)

def RPE(gt_poses,es_poses,length=100,step=10,relation="translation"):
    def compute_arc_length(positions):
        d = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        return np.insert(np.cumsum(d), 0, 0.0)
    
    def find_rpe_pairs(arc_length, L,step):
        pairs = []
        N = len(arc_length)
        for i in range(0,N,step):
            target = arc_length[i] + L
            j = bisect.bisect_left(arc_length, target)
            if j < N:
                pairs.append((i, j))

        return pairs
    
    arc = compute_arc_length(gt_poses[:,:3,-1])
    pairs = find_rpe_pairs(arc, length,step)

    errors = []
    try:
        for i, j in pairs:
            T_ij_est = np.linalg.inv(es_poses[i]) @ es_poses[j]
            T_ij_gt  = np.linalg.inv(gt_poses[i]) @ gt_poses[j]
            E = np.linalg.inv(T_ij_gt) @ T_ij_est
            error = calc_error(E,relation)
            errors.append(error)
    except:
        return np.array([float("nan")])
    return np.array(errors)