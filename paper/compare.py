
import numpy as np
from kitti_metric import KITTI_metric
from metrics import RPE
from pathlib import Path
from scipy.spatial.transform import Rotation
import re
import glob


"""
DATALOADING FUNCTIONS
"""
def kitti(poses_file):
    """For Loading poses in KITTI format"""
    _poses = np.loadtxt(poses_file, delimiter=" ").reshape((-1,3,4))
    poses = np.zeros((len(_poses),4,4))
    poses[:,:3] = _poses
    poses[:,-1,-1] = 1
    return None, poses

def kitti_gt(file_path) -> np.ndarray:
    """For loading the GT KITTI poses. Similar to kitti, but also applies calibration"""
    seq = str(Path(file_path).stem)
    import pykitti
    import getpass
    drive = pykitti.odometry(f"/media/{getpass.getuser()}/dataset-ssd/datasets/KITTI-odometry", seq)
    #drive = pykitti.odometry("/mnt/ceph-hdd/projects/scc_umin_baum/kurda/KITTI-odometry", seq)
    T_cam0_velo = drive.calib.T_cam0_velo
    poses = np.array(drive.poses)

    poses = np.linalg.inv(T_cam0_velo) @ poses @ T_cam0_velo
    return None, poses

def kiss_on_kitti(file_path):
    seq = str(Path(file_path).stem).replace("_poses_kitti","")
    import pykitti
    drive = pykitti.odometry(f"/media/{getpass.getuser()}/dataset-ssd/datasets/KITTI-odometry", seq)
    T_cam0_velo = drive.calib.T_cam0_velo
    _, poses = kitti(file_path)

    poses = np.linalg.inv(T_cam0_velo) @ poses @ T_cam0_velo
    return None, poses

def mad_on_kitti(file_path):
    seq = str(Path(file_path).parent.stem)
    import pykitti
    drive = pykitti.odometry(f"/media/{getpass.getuser()}/dataset-ssd/datasets/KITTI-odometry", seq)
    T_cam0_velo = drive.calib.T_cam0_velo
    _, poses = kitti(file_path)

    poses = np.linalg.inv(T_cam0_velo) @ poses @ T_cam0_velo
    return None, poses


def mad_on_kitti(file_path):
    seq = str(Path(file_path).parent.stem)
    import pykitti
    drive = pykitti.odometry(f"/media/{getpass.getuser()}/dataset-ssd/datasets/KITTI-odometry", seq)
    T_cam0_velo = drive.calib.T_cam0_velo
    _, poses = kitti(file_path)
    poses = np.linalg.inv(T_cam0_velo) @ poses @ T_cam0_velo
    return None, poses

def mulran_gt(poses_file: str):
    #Taken from KISS-ICP
    # https://github.com/prbonn/kiss-icp
    """MuRan has more poses than scans, therefore we need to match 1-1 timestamp with pose"""
    def read_csv(poses_file: str):
        poses = np.loadtxt(poses_file, delimiter=",")
        timestamps = poses[:, 0]
        poses = poses[:, 1:]
        n = poses.shape[0]
        poses = np.concatenate(
            (poses, np.zeros((n, 3), dtype=np.float32), np.ones((n, 1), dtype=np.float32)),
            axis=1,
        )
        poses = poses.reshape((n, 4, 4))  # [N, 4, 4]
        return poses, timestamps
    
    def closest_searchsorted(a, v):
        ii_right = np.searchsorted(a,v)
        ii_right = np.clip(ii_right,0,len(a)-1)
        ii_left = ii_right - 1
        ii_left = np.clip(ii_left,0,len(a)-1)
        dt_left = np.abs(v - a[ii_left])
        dt_right = np.abs(v - a[ii_right])
        ii = np.full(ii_left.shape,-1,int)
        ii[dt_left <= dt_right] = ii_left[dt_left <= dt_right]
        ii[dt_right < dt_left] = ii_right[dt_right < dt_left]
        return ii

    # Read the csv file
    poses, timestamps = read_csv(poses_file)
    # Extract only the poses that has a matching Ouster scan
    lidar_files = sorted(glob.glob(str(Path(poses_file).parent.joinpath("Ouster").joinpath("*.bin"))))
    scan_timestamps = np.array([int(Path(e).stem) for e in lidar_files])
    poses = poses[closest_searchsorted(timestamps,scan_timestamps)]
    # Convert from global coordinate poses to local poses
    first_pose = poses[0, :, :]
    poses = np.linalg.inv(first_pose) @ poses
    T_lidar_to_base, T_base_to_lidar = mulran_calibration()
    return None, T_lidar_to_base @ poses @ T_base_to_lidar

def mad_on_mulran(poses_file):
    _, poses = kitti(poses_file)
    T_lidar_to_base, T_base_to_lidar = mulran_calibration()
    return None, T_lidar_to_base @ poses @ T_base_to_lidar

def mulran_calibration():
    T_lidar_to_base = np.array(
        [
            [-9.9998295e-01, -5.8398386e-03, -5.2257060e-06, 1.7042000e00],
            [5.8398386e-03, -9.9998295e-01, 1.7758769e-06, -2.1000000e-02],
            [-5.2359878e-06, 1.7453292e-06, 1.0000000e00, 1.8047000e00],
            [0.0000000e00, 0.0000000e00, 0.0000000e00, 1.0000000e00],
        ]
    )
    T_base_to_lidar = np.linalg.inv(T_lidar_to_base)
    return T_lidar_to_base, T_base_to_lidar

"""
FPS Functions

"""
def kiss_fps(path):
    try:
        path = Path(path).parent.joinpath("result_metrics.log")
        lines = tuple(open(path, 'r'))
        fps = float(lines[-3].split("|")[2])    
        return fps
    except:
        return float("nan")
    
def kiss_slam_fps(path):
    try:
        path = Path(path).parent.joinpath("result_metrics.log")
        lines = tuple(open(path, 'r'))
        fps = float(lines[-4].split("|")[2])    
        return fps
    except:
        return float("nan")

def ours_fps(file_path):
    import yaml
    yaml_path = Path(file_path).with_suffix(".yaml")
    with open(yaml_path, "r") as f:
        summary = yaml.safe_load(f)

    return float(summary["report"]["fps"])

def mad_fps(path):
    try:
        path = Path(path).parent.joinpath("timings.txt")
        lines = tuple(open(path, 'r'))
        return len(lines) / np.sum([float(e) for e in lines])
    except:
        return float("nan")

"""
METRICS
RPE also available (imported at the top)
"""

def KITTI(gt_poses,es_poses):
    res = KITTI_metric(gt_poses,es_poses)[:,4]
    return res

"""
ARGPARSE AND TABLE CREATION
"""

def main(paths, dataloading_functions, metric,clip=False):
    _, gt_poses = dataloading_functions[0](paths[0])
    out = []
    for i, path in enumerate(paths[1:]):
        _, poses = dataloading_functions[i+1](path)
        poses = np.array([np.linalg.inv(poses[0]) @ e for e in poses])
        all_metrics = []
        for metric in metrics:
            if not clip:
                all_metrics.append(np.mean(metric(gt_poses,poses)))
            else:
                all_metrics.append(np.mean(metric(gt_poses[:len(poses)],poses)))
        out.append(all_metrics)
    return out

def glob_with_captures(pattern):
    regex = "^" + re.escape(pattern).replace(r"\*", r"([^/]*)") + "$"
    regex = regex.replace(r"\?", ".")
    regex = re.compile(regex)

    paths = sorted(glob.glob(pattern))
    matches = []
    for path in paths:
        m = regex.match(path)
        matches.append(m.groups()[0])
    
    return paths, matches

def parse_metric_string(metric_string):
    metric_string_s = metric_string.split(":")
    metric_function = globals()[metric_string_s[0]]
    if len(metric_string_s) > 1:
        assert len(metric_string_s) == 2
        return partial(metric_function,length=float(metric_string_s[1]))
    
    return metric_function


"""
TABLE FORMATTING
"""

def format_value(v):
    """Format float to 2 decimals, otherwise convert to string"""
    return f"{v:.2f}" if isinstance(v, (float, np.floating)) else str(v)

import operator
def format_rows(data,num_metrics,mode="green", comp=operator.lt):
    # Process each row
    processed_data = []
    
    for row in data:
        row_copy = row.copy()
        # Going in num_methods steps through table
        for m in range(num_metrics):
            min_val = float('nan')
            min_indices = []
            num_methods = (len(row)-1) // num_metrics
            for i, val in enumerate(row[1+m*num_methods:1+(m+1)*num_methods],1+m*num_methods):
                try:
                    num_val = float(val)
                    if np.isnan(min_val) or comp(num_val,min_val):
                        min_val = num_val
                        min_indices = [i]
                    elif num_val == min_val:
                        min_indices.append(i)
                except (ValueError, TypeError):
                    continue
        
            # Apply bold formatting
            for i in min_indices:
                if mode == "green":
                    row_copy[i] = f"\033[32m{row_copy[i]}\033[0m"
                elif mode == "bold":
                    row_copy[i] = f"\033[1m{row_copy[i]}\033[0m"
                elif mode == "latex":
                    row_copy[i] = "\\textbf{" + f"{row_copy[i]}" + "}"
            
        processed_data.append(row_copy)
    
    return processed_data

if __name__ == "__main__":
    import glob
    from functools import partial
    import argparse
    import re
    import tabulate
    import getpass

    parser = argparse.ArgumentParser(
        prog='ProgramName',
        description='What the program does',
        epilog='Text at the bottom of help')
    
    parser.add_argument('path_dataloader_name',nargs='+',default=[])
    parser.add_argument("-m", '--metrics',nargs='+',default=["KITTI","RPE:100"])
    parser.add_argument("--formatting",default="color")
    parser.add_argument('--fps',nargs='+',default=[])
    parser.add_argument('--clip',action="store_true")
    parser.add_argument('--exclude',nargs='+',default=[])
    args = parser.parse_args()

    assert len(args.path_dataloader_name) % 3 == 0

    paths = [args.path_dataloader_name[i] for i in range(0,len(args.path_dataloader_name),3)]
    dataloader = [args.path_dataloader_name[i] for i in range(1,len(args.path_dataloader_name),3)]
    dataloader = [globals()[e] for e in dataloader]
    names = [args.path_dataloader_name[i] for i in range(2,len(args.path_dataloader_name),3)]

    metrics = [parse_metric_string(e) for e in args.metrics]

    if "*" in paths[0]:
        # This is a glob
        glob_paths_s = []
        captures_s = []
        for path in paths:
            glob_paths, captures = glob_with_captures(path)
            glob_paths_s.append(glob_paths)
            captures_s.append(captures)
        paths_s = list(zip(*glob_paths_s))
        captures = captures_s[0]
    else:
        paths_s = [paths]
        captures = list(range(len(paths)))

    _captures = []
    results_s = []
    for seq, _paths in zip(captures,paths_s):
        # Excluding certain trajectories
        skip = False
        for e in args.exclude:
            pattern = re.compile(e)
            if pattern.match(seq):
                skip = True
                continue

        if not skip:
            results = main(_paths,dataloader,metrics,args.clip) # Return: num_sequences,num_methods,num_metrics ndarray
            results_s.append(results)
            _captures.append(seq)
        
    captures = _captures
    results_s = np.array(results_s)
    results_s = np.swapaxes(results_s,1,2) # Swapping to num_sequences,num_metrics,num_methods
    results_s = np.reshape(results_s, (results_s.shape[0],-1)) # Reshaping num_sequences, (metric 1 for all methods, metric 2 fro all methods...)

    table_names_header = [""] + list(map(str,names[1:] * len(metrics)))
    table_metric_header = [""] + list(map(str,np.repeat(args.metrics,(len(names)-1))))
    header = [f"{e1}\n{e2}" for e1,e2 in zip(table_names_header,table_metric_header)] # Formatting for multiple header rows in tabulate with grid, header are off for latex
    
    # Table with row results
    table = []
    for seq, results in zip(captures, results_s):
        table.append([seq,*(results.tolist())])
    mean_error = np.nanmean(results_s,axis=0)
    table.append(["mean",*(mean_error.tolist())])

    #Formatting table, rounding etc
    table_formatted = []
    for row in table:
        row_formatted = []
        for e in row:
            row_formatted.append(format_value(e))
        table_formatted.append(row_formatted)

    # Making the min pretty
    if args.formatting == "color":
        table_formatted = format_rows(table_formatted,len(args.metrics),mode="green")
    elif args.formatting == "latex":
        table_formatted = format_rows(table_formatted,len(args.metrics),mode="latex")
    elif args.formatting == "none":
        pass

    # Patching in last row fps into table_formatted
    if len(args.fps) > 0:
        fps_functions = [globals()[e] for e in args.fps]
        mean_fps = []
        _, captures = glob_with_captures(paths[0])
        for glob_path, func in zip(paths[1:],fps_functions):
            glob_paths, _ = glob_with_captures(glob_path)

            # Removing the excluded sequences
            sequence_paths = []
            seqs = []
            for p, cap in zip(glob_paths,captures):
                skip = False
                for e in args.exclude:
                    import re
                    pattern = re.compile(e)
                    if pattern.match(cap):
                        skip = True
                        continue

                if not skip:
                    seqs.append(cap)
                    sequence_paths.append(p)

            if len(sequence_paths) != len(results_s):
                print("Found only", len(sequence_paths) ," fps files for ", len(results_s), " sequences.")

            mean_fps.append(format_value(np.nanmean([func(e) for e in sequence_paths])))

        mean_fps_entries = np.full(len(results_s),np.nan)
        mean_fps_entries[:len(mean_fps)] = np.array(mean_fps)
        #fps_row = ["FPS", *(np.tile(mean_fps,len(args.metrics)))]
        fps_row = ["FPS", *mean_fps_entries]
        fps_row_formatted = fps_row

        # Making the max pretty
        if args.formatting == "color":
            fps_row_formatted = format_rows([fps_row],len(args.metrics),mode="green",comp=operator.gt)
        elif args.formatting == "latex":
            fps_row_formatted = format_rows([fps_row],len(args.metrics),mode="latex",comp=operator.gt)
        elif args.formatting == "none":
            pass

        # Tagging max to the end of the table

        table_formatted.append(fps_row_formatted[0])


    if args.formatting == "color":
        print(tabulate.tabulate(table_formatted,headers=header,tablefmt="grid"))
    elif args.formatting == "latex":
        print(tabulate.tabulate(table_formatted,headers=header,tablefmt="latex_raw"))
    elif args.formatting == "none":
        print(tabulate.tabulate(table_formatted,headers=header,tablefmt="grid"))

    
