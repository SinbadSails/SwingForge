"""
PoseEngine — MediaPipe Pose wrapper for tennis swing analysis.
Extracts 33 body keypoints, calculates joint angles, and tracks
wrist/elbow velocities for swing phase detection.
"""

import cv2
import numpy as np
import mediapipe as mp


# MediaPipe Pose landmark indices
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


class PoseEngine:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def extract_keypoints(self, frame):
        """Extract 33 pose keypoints from a single frame.
        Returns dict of {landmark_name: (x_px, y_px, z, visibility)} or None if no pose detected.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        if not results.pose_landmarks:
            return None

        h, w = frame.shape[:2]
        keypoints = {}
        for name, idx in LANDMARKS.items():
            lm = results.pose_landmarks.landmark[idx]
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
        o = 'left' if side == 'right' else 'right'  # opposite side

        angles = {}

        # Elbow angle: shoulder-elbow-wrist
        angles['elbow_angle'] = self.calculate_angle(
            keypoints[f'{s}_shoulder'], keypoints[f'{s}_elbow'], keypoints[f'{s}_wrist']
        )

        # Shoulder angle: elbow-shoulder-hip
        angles['shoulder_angle'] = self.calculate_angle(
            keypoints[f'{s}_elbow'], keypoints[f'{s}_shoulder'], keypoints[f'{s}_hip']
        )

        # Hip rotation: angle between shoulder line and hip line (projected to 2D)
        shoulder_mid = np.array([(keypoints['left_shoulder'][0] + keypoints['right_shoulder'][0]) / 2,
                                  (keypoints['left_shoulder'][1] + keypoints['right_shoulder'][1]) / 2])
        hip_mid = np.array([(keypoints['left_hip'][0] + keypoints['right_hip'][0]) / 2,
                             (keypoints['left_hip'][1] + keypoints['right_hip'][1]) / 2])

        shoulder_vec = np.array([keypoints['right_shoulder'][0] - keypoints['left_shoulder'][0],
                                  keypoints['right_shoulder'][1] - keypoints['left_shoulder'][1]])
        hip_vec = np.array([keypoints['right_hip'][0] - keypoints['left_hip'][0],
                             keypoints['right_hip'][1] - keypoints['left_hip'][1]])

        cos_rot = np.dot(shoulder_vec, hip_vec) / (np.linalg.norm(shoulder_vec) * np.linalg.norm(hip_vec) + 1e-8)
        cos_rot = np.clip(cos_rot, -1.0, 1.0)
        angles['hip_rotation'] = np.degrees(np.arccos(cos_rot))

        # Knee bend: hip-knee-ankle
        angles['knee_angle'] = self.calculate_angle(
            keypoints[f'{s}_hip'], keypoints[f'{s}_knee'], keypoints[f'{s}_ankle']
        )

        # Racket lag: wrist-elbow-shoulder angle at backswing
        angles['racket_lag'] = self.calculate_angle(
            keypoints[f'{s}_wrist'], keypoints[f'{s}_elbow'], keypoints[f'{s}_shoulder']
        )

        # Contact point height relative to hip
        hip_y = keypoints[f'{s}_hip'][1]
        wrist_y = keypoints[f'{s}_wrist'][1]
        # In image coords, y increases downward, so lower y = higher position
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
        """Draw pose skeleton on frame.
        Args:
            color: BGR tuple, default neon green
            alpha: opacity (1.0 = fully opaque)
        """
        if keypoints is None:
            return frame

        overlay = frame.copy() if alpha < 1.0 else frame

        # Draw connections
        for start, end in SKELETON_CONNECTIONS:
            if start in keypoints and end in keypoints:
                p1 = (int(keypoints[start][0]), int(keypoints[start][1]))
                p2 = (int(keypoints[end][0]), int(keypoints[end][1]))
                vis1 = keypoints[start][3] if len(keypoints[start]) > 3 else 1.0
                vis2 = keypoints[end][3] if len(keypoints[end]) > 3 else 1.0
                if vis1 > 0.5 and vis2 > 0.5:
                    cv2.line(overlay, p1, p2, color, thickness)

        # Draw keypoints
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
        self.pose.close()
