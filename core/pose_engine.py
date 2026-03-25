"""
PoseEngine — MediaPipe Pose Landmarker wrapper for tennis swing analysis.
Uses the new Tasks API (mediapipe 0.10.14+).
Extracts 33 body keypoints, calculates joint angles, and tracks
wrist/elbow velocities for swing phase detection.
"""

import cv2
import numpy as np
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# MediaPipe Pose landmark indices (same as legacy API)
LANDMARKS = {
    'nose': 0,
    'left_shoulder': 11, 'right_shoulder': 12,
    'left_elbow': 13, 'right_elbow': 14,
    'left_wrist': 15, 'right_wrist': 16,
    'left_hip': 23, 'right_hip': 24,
    'left_knee': 25, 'right_knee': 26,
    'left_ankle': 27, 'right_ankle': 28,
    'left_pinky': 17, 'right_pinky': 18,
    'left_index': 19, 'right_index': 20,
}

# Skeleton connections for drawing
SKELETON_CONNECTIONS = [
    ('left_shoulder', 'right_shoulder'),
    ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'),
    ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'),
    ('left_shoulder', 'left_hip'), ('right_shoulder', 'right_hip'),
    ('left_hip', 'right_hip'),
    ('left_hip', 'left_knee'), ('left_knee', 'left_ankle'),
    ('right_hip', 'right_knee'), ('right_knee', 'right_ankle'),
]

# Default model path
DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 'data', 'models', 'pose_landmarker_lite.task'
)


class PoseEngine:
    def __init__(self, model_path=None, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        if model_path is None:
            model_path = DEFAULT_MODEL_PATH

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)

    def extract_keypoints(self, frame):
        """Extract 33 pose keypoints from a single frame.
        Returns dict of {landmark_name: (x_px, y_px, z, visibility)} or None if no pose detected.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_image)

        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return None

        h, w = frame.shape[:2]
        landmarks = result.pose_landmarks[0]  # first person
        keypoints = {}
        for name, idx in LANDMARKS.items():
            lm = landmarks[idx]
            keypoints[name] = (lm.x * w, lm.y * h, lm.z, lm.visibility)
        return keypoints

    def extract_keypoints_batch(self, frames):
        """Extract keypoints for a list of frames."""
        return [self.extract_keypoints(f) for f in frames]

    @staticmethod
    def calculate_angle(p1, p2, p3):
        """Calculate angle at p2 formed by p1-p2-p3 (in degrees).
        Each point is (x, y, ...).
        """
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))

    def get_joint_angles(self, keypoints, side='right'):
        """Calculate key joint angles for swing analysis.
        Returns dict with elbow_angle, shoulder_angle, hip_angle, knee_angle.
        """
        if keypoints is None:
            return None

        s = side

        angles = {}

        # Elbow angle: shoulder-elbow-wrist
        angles['elbow_angle'] = self.calculate_angle(
            keypoints[f'{s}_shoulder'], keypoints[f'{s}_elbow'], keypoints[f'{s}_wrist']
        )

        # Shoulder angle: elbow-shoulder-hip
        angles['shoulder_angle'] = self.calculate_angle(
            keypoints[f'{s}_elbow'], keypoints[f'{s}_shoulder'], keypoints[f'{s}_hip']
        )

        # Hip rotation: use Z-depth difference between shoulders as rotation proxy.
        # MediaPipe Z is depth relative to hip midpoint. Larger Z diff between
        # left/right shoulder means more rotation (one shoulder closer to camera).
        # We also use the 2D shoulder-width-to-hip-width ratio as a secondary cue:
        # when rotated, the shoulder line appears shorter in 2D than the hip line.
        shoulder_width_2d = np.sqrt(
            (keypoints['right_shoulder'][0] - keypoints['left_shoulder'][0])**2 +
            (keypoints['right_shoulder'][1] - keypoints['left_shoulder'][1])**2
        )
        hip_width_2d = np.sqrt(
            (keypoints['right_hip'][0] - keypoints['left_hip'][0])**2 +
            (keypoints['right_hip'][1] - keypoints['left_hip'][1])**2
        )
        # Width ratio: 1.0 = no rotation, <1 = shoulders rotated relative to hips
        width_ratio = shoulder_width_2d / (hip_width_2d + 1e-8)

        # Z-based rotation: difference in Z between left and right shoulder
        z_diff = abs(keypoints['right_shoulder'][2] - keypoints['left_shoulder'][2])

        # Combined rotation estimate: map to 0-180° range
        # width_ratio of 0.3 ≈ 90° rotation, 1.0 ≈ 0° rotation
        rotation_from_width = max(0, min(180, (1.0 - width_ratio) * 150))
        rotation_from_z = min(180, z_diff * 300)  # Z is normalized, scale up
        angles['hip_rotation'] = max(rotation_from_width, rotation_from_z)

        # Knee bend: hip-knee-ankle
        angles['knee_angle'] = self.calculate_angle(
            keypoints[f'{s}_hip'], keypoints[f'{s}_knee'], keypoints[f'{s}_ankle']
        )

        # Racket lag: angle between forearm (elbow→wrist) and the vertical axis.
        # Measures how far the racket head is "lagging" behind during backswing.
        # 0° = arm pointing straight down, 90° = arm horizontal, >90° = behind body
        elbow = np.array(keypoints[f'{s}_elbow'][:2])
        wrist = np.array(keypoints[f'{s}_wrist'][:2])
        forearm_vec = wrist - elbow
        vertical = np.array([0, 1])  # down in image coords
        cos_lag = np.dot(forearm_vec, vertical) / (np.linalg.norm(forearm_vec) + 1e-8)
        cos_lag = np.clip(cos_lag, -1.0, 1.0)
        angles['racket_lag'] = np.degrees(np.arccos(cos_lag))

        # Contact point height relative to hip (higher = serving/overhead)
        hip_y = keypoints[f'{s}_hip'][1]
        wrist_y = keypoints[f'{s}_wrist'][1]
        angles['contact_height_ratio'] = hip_y / (wrist_y + 1e-8)

        return angles

    def get_wrist_velocity(self, keypoints_sequence, fps=30):
        """Calculate wrist velocity (px/frame) across a keypoint sequence.
        Returns list of velocities (same length as input, first element is 0).
        """
        velocities = [0.0]
        for i in range(1, len(keypoints_sequence)):
            if keypoints_sequence[i] is None or keypoints_sequence[i - 1] is None:
                velocities.append(0.0)
                continue
            curr = np.array(keypoints_sequence[i]['right_wrist'][:2])
            prev = np.array(keypoints_sequence[i - 1]['right_wrist'][:2])
            vel = np.linalg.norm(curr - prev) * fps  # px/s
            velocities.append(vel)
        return velocities

    def get_wrist_acceleration(self, velocities, fps=30):
        """Calculate wrist acceleration from velocity sequence."""
        accels = [0.0]
        for i in range(1, len(velocities)):
            accel = (velocities[i] - velocities[i - 1]) * fps  # px/s²
            accels.append(accel)
        return accels

    def draw_skeleton(self, frame, keypoints, color=(0, 255, 127), thickness=2, alpha=1.0):
        """Draw pose skeleton on frame."""
        if keypoints is None:
            return frame

        overlay = frame.copy() if alpha < 1.0 else frame

        for start, end in SKELETON_CONNECTIONS:
            if start in keypoints and end in keypoints:
                p1 = (int(keypoints[start][0]), int(keypoints[start][1]))
                p2 = (int(keypoints[end][0]), int(keypoints[end][1]))
                vis1 = keypoints[start][3] if len(keypoints[start]) > 3 else 1.0
                vis2 = keypoints[end][3] if len(keypoints[end]) > 3 else 1.0
                if vis1 > 0.5 and vis2 > 0.5:
                    cv2.line(overlay, p1, p2, color, thickness)

        for name, kp in keypoints.items():
            vis = kp[3] if len(kp) > 3 else 1.0
            if vis > 0.5:
                cv2.circle(overlay, (int(kp[0]), int(kp[1])), 4, color, -1)

        if alpha < 1.0:
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        return frame

    def draw_divergence_lines(self, frame, user_kp, pro_kp, threshold_deg=15):
        """Draw red lines between user and pro joints where angle divergence exceeds threshold."""
        if user_kp is None or pro_kp is None:
            return frame

        user_angles = self.get_joint_angles(user_kp)
        pro_angles = self.get_joint_angles(pro_kp)
        if user_angles is None or pro_angles is None:
            return frame

        joint_map = {
            'elbow_angle': 'right_elbow',
            'shoulder_angle': 'right_shoulder',
            'knee_angle': 'right_knee',
        }

        for angle_name, joint_name in joint_map.items():
            diff = abs(user_angles[angle_name] - pro_angles[angle_name])
            if diff > threshold_deg:
                u_pt = (int(user_kp[joint_name][0]), int(user_kp[joint_name][1]))
                p_pt = (int(pro_kp[joint_name][0]), int(pro_kp[joint_name][1]))
                cv2.line(frame, u_pt, p_pt, (0, 0, 255), 2)
                cv2.putText(frame, f"{diff:.0f}deg", u_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        return frame

    def release(self):
        self.landmarker.close()
