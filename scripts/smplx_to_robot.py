import argparse
import pathlib
import os
import time

import numpy as np
import torch

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import KinematicsModel
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting import torch_utils
from general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast

from rich import print

if __name__ == "__main__":
    
    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smplx_file",
        help="SMPLX motion file to load.",
        type=str,
        # required=True,
        default="/home/robros/igris-c_retarget/GMR/motions/ACCAD/Female1Walking_c3d/B1_-_stand_to_walk_stageii.npz",

    )
    
    parser.add_argument(
        "--robot",
        choices=["unitree_g1", "unitree_g1_with_hands", "unitree_h1", "unitree_h1_2",
                 "booster_t1", "booster_t1_29dof","stanford_toddy", "fourier_n1", 
                "engineai_pm01", "kuavo_s45", "hightorque_hi", "galaxea_r1pro", "berkeley_humanoid_lite", "booster_k1",
                "pnd_adam_lite", "openloong", "tienkung", "igris_c"],
        default="unitree_g1",
    )
    
    parser.add_argument(
        "--save_path",
        default=None,
        help="Path to save the robot motion.",
    )
    
    parser.add_argument(
        "--loop",
        default=False,
        action="store_true",
        help="Loop the motion.",
    )

    parser.add_argument(
        "--record_video",
        default=False,
        action="store_true",
        help="Record the video.",
    )
    
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="Path to save the recorded video (mp4). If not set, a default path in videos/ will be used.",
    )

    parser.add_argument(
        "--rate_limit",
        default=False,
        action="store_true",
        help="Limit the rate of the retargeted robot motion to keep the same as the human motion.",
    )

    args = parser.parse_args()


    SMPLX_FOLDER = HERE / ".." / "assets" / "body_models"
    
    
    # Load SMPLX trajectory
    smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(
        args.smplx_file, SMPLX_FOLDER
    )
    
    # align fps
    tgt_fps = 30
    smplx_data_frames, aligned_fps, _, _, _, _ = get_smplx_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=tgt_fps
    )
    
   
    # Initialize the retargeting system
    retarget = GMR(
        actual_human_height=actual_human_height,
        src_human="smplx",
        tgt_robot=args.robot,
    )
    
    default_video_path = f"videos/{args.robot}_{args.smplx_file.split('/')[-1].split('.')[0]}.mp4"
    video_path = args.video_path if args.video_path is not None else default_video_path
    if args.record_video:
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
    robot_motion_viewer = RobotMotionViewer(robot_type=args.robot,
                                            motion_fps=aligned_fps,
                                            transparent_robot=0,
                                            record_video=args.record_video,
                                            video_path=video_path,)
    

    curr_frame = 0
    # FPS measurement variables
    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0  # Display FPS every 2 seconds
    
    if args.save_path is not None:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:  # Only create directory if it's not empty
            os.makedirs(save_dir, exist_ok=True)
        qpos_list = []
    
    # Start the viewer
    i = 0

    while True:
        if args.loop:
            i = (i + 1) % len(smplx_data_frames)
        else:
            i += 1
            if i >= len(smplx_data_frames):
                break
        
        # FPS measurement
        fps_counter += 1
        current_time = time.time()
        if current_time - fps_start_time >= fps_display_interval:
            actual_fps = fps_counter / (current_time - fps_start_time)
            print(f"Actual rendering FPS: {actual_fps:.2f}")
            fps_counter = 0
            fps_start_time = current_time
        
        # Update task targets.
        smplx_data = smplx_data_frames[i]

        # retarget
        qpos = retarget.retarget(smplx_data)

        # visualize
        robot_motion_viewer.step(
            root_pos=qpos[:3],
            root_rot=qpos[3:7],
            dof_pos=qpos[7:],
            human_motion_data=retarget.scaled_human_data,
            # human_motion_data=smplx_data,
            human_pos_offset=np.array([0.0, 0.0, 0.0]),
            show_human_body_name=False,
            rate_limit=args.rate_limit,
        )
        if args.save_path is not None:
            qpos_list.append(qpos)
            
    if args.save_path is not None:
        import pickle
        root_pos = np.array([qpos[:3] for qpos in qpos_list])
        # save from wxyz to xyzw
        root_rot = np.array([qpos[3:7][[1,2,3,0]] for qpos in qpos_list])
        dof_pos = np.array([qpos[7:] for qpos in qpos_list])
        local_body_pos = None
        body_names = None
        pose_aa = None
        if qpos_list:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            kinematics_model = KinematicsModel(retarget.xml_file, device=device)
            body_names = kinematics_model.body_names
            root_rot_tensor = torch.from_numpy(root_rot).to(device=device, dtype=torch.float)
            pose_aa = np.zeros(
                (dof_pos.shape[0], kinematics_model.num_joint, 3), dtype=np.float32
            )
            pose_aa[:, 0, :] = torch_utils.quat_to_exp_map(root_rot_tensor).cpu().numpy()
            for joint_idx in range(1, kinematics_model.num_joint):
                joint = kinematics_model._joints[joint_idx]
                if joint.dof_dim == 0:
                    continue
                dof_start = joint.dof_idx
                if joint.dof_dim == 1:
                    axis = joint._axis.detach().cpu().numpy().astype(np.float32)
                    pose_aa[:, joint_idx, :] = dof_pos[:, dof_start:dof_start + 1] * axis[None, :]
                elif joint.dof_dim == 3:
                    pose_aa[:, joint_idx, :] = dof_pos[:, dof_start:dof_start + 3]
                else:
                    raise ValueError(f"Unsupported dof_dim: {joint.dof_dim}")
        
        motion_data = {
            "fps": aligned_fps,
            "root_pos": root_pos,
            "root_rot": root_rot,
            "dof_pos": dof_pos,
            "local_body_pos": local_body_pos,
            "link_body_list": body_names,
            "pose_aa": pose_aa,
        }
        with open(args.save_path, "wb") as f:
            pickle.dump(motion_data, f)
        print(f"Saved to {args.save_path}")
            
      
    
    robot_motion_viewer.close()
