"""
SwingClassifier — Detects swing phases and classifies stroke type.
Uses wrist velocity/acceleration curves from PoseEngine to segment:
  Preparation → Loading → Contact → Follow-through
"""

import numpy as np


# Swing phases
PHASES = ['idle', 'preparation', 'loading', 'contact', 'follow_through']

# Acceleration threshold for phase transitions (px/s²)
ACCEL_THRESHOLD = 800

# Stroke type detection thresholds
SERVE_WRIST_HEIGHT_RATIO = 0.6  # wrist above 60% of body height = likely serve


class SwingClassifier:
    def __init__(self, fps=30):
        self.fps = fps

    def detect_phases(self, keypoints_sequence, wrist_velocities, wrist_accelerations):
        """Segment a swing into phases based on wrist dynamics.
        Returns list of phase labels (same length as input).
        """
        n = len(keypoints_sequence)
        phases = ['idle'] * n

        if n < 5:
            return phases

        # Find the contact frame: peak wrist velocity
        peak_idx = int(np.argmax(wrist_velocities))
        if wrist_velocities[peak_idx] < 50:  # no significant motion
            return phases

        # Walk backward from peak to find loading start (acceleration crosses threshold)
        loading_start = peak_idx
        for i in range(peak_idx - 1, -1, -1):
            if abs(wrist_accelerations[i]) < ACCEL_THRESHOLD * 0.3:
                loading_start = i
                break

        # Preparation starts even earlier (low velocity phase before loading)
        prep_start = loading_start
        for i in range(loading_start - 1, -1, -1):
            if wrist_velocities[i] < wrist_velocities[loading_start] * 0.1:
                prep_start = i
                break

        # Follow-through: after peak velocity until velocity drops below 20% of peak
        follow_end = peak_idx
        peak_vel = wrist_velocities[peak_idx]
        for i in range(peak_idx + 1, n):
            if wrist_velocities[i] < peak_vel * 0.2:
                follow_end = i
                break
        else:
            follow_end = n - 1

        # Assign phases
        for i in range(n):
            if i < prep_start:
                phases[i] = 'idle'
            elif i < loading_start:
                phases[i] = 'preparation'
            elif i < peak_idx:
                phases[i] = 'loading'
            elif i == peak_idx:
                phases[i] = 'contact'
            elif i <= follow_end:
                phases[i] = 'follow_through'
            else:
                phases[i] = 'idle'

        return phases

    def classify_stroke(self, keypoints_at_contact):
        """Classify stroke type from keypoints at contact frame.
        Returns one of: 'forehand', 'backhand', 'serve', 'volley', 'unknown'
        """
        if keypoints_at_contact is None:
            return 'unknown'

        kp = keypoints_at_contact

        # Check for serve: wrist significantly above shoulder
        r_wrist_y = kp['right_wrist'][1]
        r_shoulder_y = kp['right_shoulder'][1]
        nose_y = kp['nose'][1]
        r_hip_y = kp['right_hip'][1]

        # In image coords, lower y = higher position
        body_height = r_hip_y - nose_y
        wrist_above_shoulder = r_shoulder_y - r_wrist_y

        if wrist_above_shoulder > body_height * 0.3:
            return 'serve'

        # Forehand vs backhand: check which side of body the wrist is on
        r_shoulder_x = kp['right_shoulder'][0]
        l_shoulder_x = kp['left_shoulder'][0]
        r_wrist_x = kp['right_wrist'][0]

        # For right-handed: if wrist is on the right side of body center = forehand
        body_center_x = (r_shoulder_x + l_shoulder_x) / 2

        # Determine facing direction from hip-shoulder relationship
        r_hip_x = kp['right_hip'][0]
        l_hip_x = kp['left_hip'][0]
        facing_right = r_shoulder_x > l_shoulder_x

        if facing_right:
            if r_wrist_x > body_center_x:
                return 'forehand'
            else:
                return 'backhand'
        else:
            if r_wrist_x < body_center_x:
                return 'forehand'
            else:
                return 'backhand'

    def get_contact_frame_index(self, phases):
        """Return the frame index of the contact phase."""
        for i, p in enumerate(phases):
            if p == 'contact':
                return i
        return None

    def detect_follow_through_completion(self, keypoints_sequence, phases):
        """Check if the arm crosses the body during follow-through.
        Returns True if follow-through is complete.
        """
        follow_frames = [i for i, p in enumerate(phases) if p == 'follow_through']
        if not follow_frames or keypoints_sequence[follow_frames[-1]] is None:
            return False

        last_kp = keypoints_sequence[follow_frames[-1]]
        r_wrist_x = last_kp['right_wrist'][0]
        l_shoulder_x = last_kp['left_shoulder'][0]
        r_shoulder_x = last_kp['right_shoulder'][0]
        body_center_x = (l_shoulder_x + r_shoulder_x) / 2

        # Follow-through is complete if wrist crosses past body center to opposite side
        facing_right = r_shoulder_x > l_shoulder_x
        if facing_right:
            return r_wrist_x < body_center_x
        else:
            return r_wrist_x > body_center_x
