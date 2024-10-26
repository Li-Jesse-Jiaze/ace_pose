import numpy as np
import glob
import os
from scipy.spatial.transform import Rotation as R

def add_noise_to_pose(pose_matrix, rot_noise_std_deg=1.0, trans_noise_std=0.01):
    """
    Adds random noise to a pose matrix.
    
    Parameters:
        pose_matrix (numpy.ndarray): 4x4 pose matrix.
        rot_noise_std_deg (float): Standard deviation of rotation noise in degrees.
        trans_noise_std (float): Standard deviation of translation noise (same unit as translation).
        
    Returns:
        numpy.ndarray: Noisy 4x4 pose matrix.
    """
    # Decompose rotation and translation
    rotation = R.from_matrix(pose_matrix[:3, :3])
    translation = pose_matrix[:3, 3]
    
    # Generate random rotation noise
    rot_noise = R.from_euler('xyz', np.random.normal(0, rot_noise_std_deg, 3), degrees=True)
    noisy_rotation = rot_noise * rotation  # Apply rotation noise first, then original rotation
    
    # Generate random translation noise
    trans_noise = np.random.normal(0, trans_noise_std, 3)
    noisy_translation = translation + trans_noise
    
    # Combine into new pose matrix
    noisy_pose = np.eye(4)
    noisy_pose[:3, :3] = noisy_rotation.as_matrix()
    noisy_pose[:3, 3] = noisy_translation
    
    return noisy_pose

def process_pose_files(directory, rot_noise_std_deg=1.0, trans_noise_std=0.01, overwrite=True, save_suffix='_noisy'):
    """
    Processes all .pose.txt files in a directory by adding noise and writing back the files.
    
    Parameters:
        directory (str): Directory containing .pose.txt files.
        rot_noise_std_deg (float): Standard deviation of rotation noise in degrees.
        trans_noise_std (float): Standard deviation of translation noise.
        overwrite (bool): Whether to overwrite the original files. If False, save as new files.
        save_suffix (str): Suffix to add to filenames if not overwriting.
    """
    pose_files = glob.glob(os.path.join(directory, '*.pose.txt'))
    print(f"Found {len(pose_files)} .pose.txt files.")
    
    for file_path in pose_files:
        # Read pose matrix
        with open(file_path, 'r') as f:
            lines = f.readlines()
            pose = []
            for line in lines:
                pose.append([float(num) for num in line.strip().split()])
            pose_matrix = np.array(pose)
        
        # Add noise
        noisy_pose = add_noise_to_pose(pose_matrix, rot_noise_std_deg, trans_noise_std)
        
        # Prepare save path
        if overwrite:
            save_path = file_path
        else:
            base, ext = os.path.splitext(file_path)
            save_path = f"{base}{save_suffix}{ext}"
        
        # Write back to file
        with open(save_path, 'w') as f:
            for row in noisy_pose:
                f.write(' '.join(f"{num:.15f}" for num in row) + '\n')
        
        print(f"Processed and saved: {save_path}")

if __name__ == "__main__":
    # Set parameters
    poses_directory = 'data/7scenes_source/redkitchen/seq-14/'
    rotation_noise_std_deg = 5.0  # Rotation noise standard deviation in degrees
    translation_noise_std = 0.1   # Translation noise standard deviation (same unit as translation)
    overwrite_files = False          # Whether to overwrite original files. If False, save as new files
    save_suffix = '_noisy'          # Suffix to add if not overwriting
    
    # Check if directory exists
    if not os.path.isdir(poses_directory):
        print(f"Directory does not exist: {poses_directory}")
        exit(1)
    
    # Process pose files
    process_pose_files(
        directory=poses_directory,
        rot_noise_std_deg=rotation_noise_std_deg,
        trans_noise_std=translation_noise_std,
        overwrite=overwrite_files,
        save_suffix=save_suffix
    )
