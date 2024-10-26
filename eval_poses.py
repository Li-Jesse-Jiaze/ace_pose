#!/usr/bin/env python3
# Copyright © Niantic, Inc. 2024.

import argparse
import logging
import math
import random
from collections import namedtuple
from distutils.util import strtobool
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

import eval_poses_util as tutil
from ace_pose import dataset_io

_logger = logging.getLogger(__name__)


def _strtobool(x):
    return bool(strtobool(x))

TestEstimate = namedtuple("TestEstimate", [
    "pose_est",
    "pose_gt",
    "focal_length",
    "confidence",
    "image_file"
])


def kabsch(pts1, pts2, estimate_scale=False):
    c_pts1 = pts1 - pts1.mean(axis=0)
    c_pts2 = pts2 - pts2.mean(axis=0)

    covariance = np.matmul(c_pts1.T, c_pts2) / c_pts1.shape[0]

    U, S, VT = np.linalg.svd(covariance)

    d = np.sign(np.linalg.det(np.matmul(VT.T, U.T)))
    correction = np.eye(3)
    correction[2, 2] = d

    if estimate_scale:
        pts_var = np.mean(np.linalg.norm(c_pts2, axis=1) ** 2)
        scale_factor = pts_var / np.trace(S * correction)
    else:
        scale_factor = 1.

    R = scale_factor * np.matmul(np.matmul(VT.T, correction), U.T)
    t = pts2.mean(axis=0) - np.matmul(R, pts1.mean(axis=0))

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return T, scale_factor


def print_hyp(hypothesis, hyp_name):
    h_translation = np.linalg.norm(hypothesis['transformation'][:3, 3])
    h_angle = np.linalg.norm(Rotation.from_matrix(hypothesis['transformation'][:3, :3]).as_rotvec()) * 180 / math.pi
    _logger.debug(f"{hyp_name}: score={hypothesis['score']}, translation={h_translation:.2f}m, "
                 f"rotation={h_angle:.1f}deg.")


def get_inliers(h_T, poses_gt, poses_est, inlier_threshold_t, inlier_threshold_r):

    # h_T aligns ground truth poses with estimates poses
    poses_gt_transformed = h_T @ poses_gt

    # calculate differences in position and rotations
    translations_delta = poses_gt_transformed[:, :3, 3] - poses_est[:, :3, 3]
    rotations_delta = poses_gt_transformed[:, :3, :3] @ poses_est[:, :3, :3].transpose([0, 2, 1])

    # translation inliers
    inliers_t = np.linalg.norm(translations_delta, axis=1) < inlier_threshold_t
    # rotation inliers
    inliers_r = Rotation.from_matrix(rotations_delta).magnitude() < (inlier_threshold_r / 180 * math.pi)
    # intersection of both
    return np.logical_and(inliers_r, inliers_t)

def estimate_alignment(estimates,
                       confidence_threshold,
                       min_cofident_estimates=10,
                       inlier_threshold_t=0.05,
                       inlier_threshold_r=5,
                       ransac_iterations=10000,
                       refinement_max_hyp=12,
                       refinement_max_it=8,
                       estimate_scale=False
                       ):
    _logger.debug("Estimate transformation between pose estimates and ground truth.")

    # Filter estimates using confidence threshold
    valid_estimates = [estimate for estimate in estimates if ((not np.any(np.isinf(estimate.pose_gt))) and (not np.any(np.isnan(estimate.pose_gt))))]
    confident_estimates = [estimate for estimate in valid_estimates if estimate.confidence > confidence_threshold]
    num_confident_estimates = len(confident_estimates)

    _logger.debug(f"{num_confident_estimates} estimates considered confident.")

    if num_confident_estimates < min_cofident_estimates:
        _logger.debug("Too few confident estimates. Aborting alignment.")
        return None, 1

    # gather estimated and ground truth poses
    poses_est = np.ndarray((num_confident_estimates, 4, 4))
    poses_gt = np.ndarray((num_confident_estimates, 4, 4))
    for i, estimate in enumerate(confident_estimates):
        poses_est[i] = estimate.pose_est
        poses_gt[i] = estimate.pose_gt

    # start robust RANSAC loop
    ransac_hypotheses = []

    for hyp_idx in range(ransac_iterations):

        # sample hypothesis
        min_sample_size = 3
        samples = random.sample(range(num_confident_estimates), min_sample_size)
        h_pts1 = poses_gt[samples, :3, 3]
        h_pts2 = poses_est[samples, :3, 3]

        h_T, h_scale = kabsch(h_pts1, h_pts2, estimate_scale)

        # calculate inliers
        inliers = get_inliers(h_T, poses_gt, poses_est, inlier_threshold_t, inlier_threshold_r)

        if inliers[samples].sum() >= 3:
            # only keep hypotheses if minimal sample is all inliers
            ransac_hypotheses.append({
                "transformation": h_T,
                "inliers": inliers,
                "score": inliers.sum(),
                "scale": h_scale
            })

    if len(ransac_hypotheses) == 0:
        _logger.debug("Did not fine a single valid RANSAC hypothesis, abort alignment estimation.")
        return None, 1

    # sort according to score
    ransac_hypotheses = sorted(ransac_hypotheses, key=lambda x: x['score'], reverse=True)

    for hyp_idx, hyp in enumerate(ransac_hypotheses):
        print_hyp(hyp, f"Hypothesis {hyp_idx}")

    # create shortlist of best hypotheses for refinement
    _logger.debug(f"Starting refinement of {refinement_max_hyp} best hypotheses.")
    ransac_hypotheses = ransac_hypotheses[:refinement_max_hyp]

    # refine all hypotheses in the short list
    for ref_hyp in ransac_hypotheses:

        print_hyp(ref_hyp, "Pre-Refinement")

        # refinement loop
        for ref_it in range(refinement_max_it):

            # re-solve alignment on all inliers
            h_pts1 = poses_gt[ref_hyp['inliers'], :3, 3]
            h_pts2 = poses_est[ref_hyp['inliers'], :3, 3]

            h_T, h_scale = kabsch(h_pts1, h_pts2, estimate_scale)

            # calculate new inliers
            inliers = get_inliers(h_T, poses_gt, poses_est, inlier_threshold_t, inlier_threshold_r)

            # check whether hypothesis score improved
            refined_score = inliers.sum()

            if refined_score > ref_hyp['score']:

                ref_hyp['transformation'] = h_T
                ref_hyp['inliers'] = inliers
                ref_hyp['score'] = refined_score
                ref_hyp['scale'] = h_scale

                print_hyp(ref_hyp, f"Refinement interation {ref_it}")

            else:
                _logger.debug(f"Stopping refinement. Score did not improve: New score={refined_score}, "
                             f"Old score={ref_hyp['score']}")
                break

    # re-sort refined hyotheses
    ransac_hypotheses = sorted(ransac_hypotheses, key=lambda x: x['score'], reverse=True)

    for hyp_idx, hyp in enumerate(ransac_hypotheses):
        print_hyp(hyp, f"Hypothesis {hyp_idx}")

    return ransac_hypotheses[0]['transformation'], ransac_hypotheses[0]['scale']



if __name__ == '__main__':
    # Setup logging.
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description='Compute pose error metrics for an ACE pose file using (pseudo) ground truth pose files.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('ace_pose_file', type=Path, help='Path to an ACE pose file with one line per image.')

    parser.add_argument('gt_pose_files', type=str,
                        help="Glob pattern for pose files, e.g. 'datasets/scene/*.txt', each file is assumed to "
                             "contain a 4x4 pose matrix, cam2world, correspondence with rgb files in the ACE pose "
                             "file is assumed by alphabetical order")

    parser.add_argument('--estimate_alignment', type=_strtobool, default=True,
                        help='Estimate rigid body transformation between estimates and ground truth.')

    parser.add_argument('--estimate_alignment_scale', type=_strtobool, default=True,
                        help='Estimate similarity transformation when estimating alignment')

    parser.add_argument('--estimate_alignment_conf_threshold', type=float, default=500,
                        help='Only consider pose estimates with higher confidence when estimates the alignment.')

    parser.add_argument('--pose_error_thresh_t', type=float, default=0.05,
                        help='Pose threshold (translation) for evaluation and alignment')

    parser.add_argument('--pose_error_thresh_r', type=float, default=5,
                        help='Pose threshold (rotation) for evaluation and alignment')

    opt = parser.parse_args()

    _logger.info("Reading ACE pose file.")

    with open(opt.ace_pose_file, 'r') as f:
        ace_pose_data = f.readlines()

    # Dict mapping file name to ACE estimate
    ace_estimates = {}

    # parse pose file data
    for pose_line in ace_pose_data:
        # image info as: file_name, q_w, q_x, q_y, q_z, t_x, t_y, t_z, focal_length, confidence
        pose_tokens = pose_line.split()

        # read file name and confidence
        file_name = pose_tokens[0]
        confidence = float(pose_tokens[-1])

        # read pose
        q_wxyz = [float(t) for t in pose_tokens[1:5]]
        t_xyz = [float(t) for t in pose_tokens[5:8]]

        # quaternion to rotation matrix
        R = Rotation.from_quat(q_wxyz[1:] + [q_wxyz[0]]).as_matrix()

        # construct full pose matrix
        T_world2cam = np.eye(4)
        T_world2cam[:3, :3] = R
        T_world2cam[:3, 3] = t_xyz

        # pose files contain world-to-cam but we need cam-to-world
        T_cam2world = np.linalg.inv(T_world2cam)

        # store ACE estimate
        ace_estimates[file_name] = (T_cam2world, confidence)

    _logger.info(f"Read {len(ace_estimates)} poses from: {opt.ace_pose_file}")

    # sort ACE estimates by file names
    sorted_ace_poses = [ace_estimates[key] for key in sorted(ace_estimates.keys())]

    # load ground truth poses, sorted by file name
    sorted_gt_poses = dataset_io.load_pose_files(opt.gt_pose_files)
    # convert torch to numpy
    sorted_gt_poses = [pose.numpy() for pose in sorted_gt_poses]

    _logger.info(f"Loaded {len(sorted_gt_poses)} ground truth poses.")

    # Keep track of rotation and translation errors for calculation of the median error.
    rErrs = []
    tErrs = []

    # Percentage of frames predicted within certain threshold from their GT pose.
    accuracy = 0

    if opt.estimate_alignment:
        # alignment needs a list of pose correspondences with confidences
        pose_correspondences = []

        # walk through ACE estimates and GT poses in parallel
        for (ace_pose, ace_confidence), gt_pose in zip(sorted_ace_poses, sorted_gt_poses):
            pose_correspondences.append((tutil.TestEstimate(
                pose_est=ace_pose,
                pose_gt=gt_pose,
                confidence=ace_confidence,
                image_file=None,
                focal_length=None
            )))

        alignment_transformation, alignment_scale = tutil.estimate_alignment(
            estimates=pose_correspondences,
            confidence_threshold=opt.estimate_alignment_conf_threshold,
            estimate_scale=opt.estimate_alignment_scale,
            inlier_threshold_r=opt.pose_error_thresh_r,
            inlier_threshold_t=opt.pose_error_thresh_t,
        )

        if alignment_transformation is None:
            _logger.info(f"Alignment requested but failed. Setting all pose errors to {math.inf}.")
    else:
        alignment_transformation = np.eye(4)
        alignment_scale = 1.

    # Evaluation Loop
    for (ace_pose, ace_confidence), gt_pose in zip(sorted_ace_poses, sorted_gt_poses):

        if alignment_transformation is not None:
            # Apply alignment transformation to GT pose
            gt_pose = alignment_transformation @ gt_pose

            # Calculate translation error.
            t_err = float(np.linalg.norm(gt_pose[0:3, 3] - ace_pose[0:3, 3]))

            # Correct translation scale with the inverse alignment scale (since we align GT with estimates)
            t_err = t_err / alignment_scale

            # Rotation error.
            gt_R = gt_pose[0:3, 0:3]
            out_R = ace_pose[0:3, 0:3]

            r_err = np.matmul(out_R, np.transpose(gt_R))
            # Compute angle-axis representation.
            r_err = cv2.Rodrigues(r_err)[0]
            # Extract the angle.
            r_err = np.linalg.norm(r_err) * 180 / math.pi
        else:
            pose_gt = None
            t_err, r_err = math.inf, math.inf

        _logger.info(f"Rotation Error: {r_err:.2f}deg, Translation Error: {t_err * 100:.1f}cm")

        # Save the errors.
        rErrs.append(r_err)
        tErrs.append(t_err * 100) # in cm

        # Check various thresholds.
        if r_err < opt.pose_error_thresh_r and t_err < opt.pose_error_thresh_t:
            accuracy += 1

    total_frames = len(rErrs)
    assert total_frames == len(ace_estimates)

    # Compute median errors.
    tErrs.sort()
    rErrs.sort()
    median_idx = total_frames // 2
    median_rErr = rErrs[median_idx]
    median_tErr = tErrs[median_idx]

    # Compute final precision.
    accuracy = accuracy / total_frames * 100

    _logger.info("===================================================")
    _logger.info("Test complete.")

    _logger.info(f'Accuracy: {accuracy:.1f}%')
    _logger.info(f"Median Error: {median_rErr:.1f}deg, {median_tErr:.1f}cm")
