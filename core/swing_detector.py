"""
SwingDetector — Detects discrete swings from a stream of wrist positions.
Captures the full swing window, scores it as ONE unit, and stores a replay buffer.

A swing is:
  1. Wrist starts moving fast (velocity > threshold)
  2. Velocity peaks (contact)
  3. Velocity drops back down (follow-through complete)
  4. Score the entire captured window as one swing

States: IDLE → WINDING_UP → SWINGING → SCORING → IDLE
"""

import numpy as np
from collections import deque


class SwingDetector:
    def __init__(self, fps=30):
        self.fps = fps
        self.state = 'IDLE'

        # Wrist position history (y-coordinates for velocity calculation)
        self.wrist_history = deque(maxlen=90)  # 3 seconds of history

        # Swing capture
        self.swing_frame_count = 0  # counter for frames in current swing
        self.swing_frames = []  # frames captured during current swing
        self.swing_keypoints = []  # keypoints captured during current swing
        self.swing_angles = []  # angles captured during current swing

        # Velocity tracking
        self.velocity_history = deque(maxlen=30)

        # Thresholds — expressed as fraction of torso length per frame.
        # This auto-adapts to user distance from camera.
        # At 6ft with 150px torso: start = 0.08 * 150 = 12px/frame
        # At 10ft with 80px torso: start = 0.08 * 80 = 6.4px/frame (still triggers)
        # At 3ft with 300px torso: start = 0.08 * 300 = 24px/frame (harder to false-trigger)
        self.start_ratio = 0.08     # fraction of torso length for swing start
        self.peak_ratio = 0.13      # fraction of torso length for real swing confirmation
        self.end_ratio = 0.03       # fraction of torso length for swing end
        self.torso_length = 150     # default, updated each frame from keypoints
        self.min_swing_frames = 8   # minimum frames for a valid swing (~0.25s)
        self.max_swing_frames = 60  # maximum frames (~2s) — cap it

        # Cooldown
        self.cooldown_frames = 30  # frames to wait after a swing before detecting next
        self.cooldown_counter = 0

        # Results
        self.completed_swings = []  # list of swing dicts
        self.peak_velocity = 0

    def update(self, wrist_pos, frame=None, keypoints=None, angles=None,
               torso_length=None, hip_pos=None):
        """Call every frame with current wrist (x, y) position.
        torso_length: shoulder-to-hip distance in pixels (for distance-adaptive thresholds)
        hip_pos: (x, y) of hip midpoint — used to filter out walking (whole body moving)
        Returns: 'IDLE', 'SWINGING', or a swing_result dict when a swing completes.
        """
        if wrist_pos is None:
            return self.state

        # Update torso length for adaptive thresholds
        if torso_length is not None and torso_length > 20:
            self.torso_length = torso_length

        self.wrist_history.append(wrist_pos)

        # Walking filter: distinguish walking (whole body moves together)
        # from swinging (arm moves WAY faster than hips).
        # During a real swing, hips DO move (rotation, weight transfer) —
        # but the wrist moves 3-5x faster than the hips.
        # During walking, wrist and hips move at similar speeds.
        if hip_pos is not None:
            if not hasattr(self, '_hip_history'):
                self._hip_history = deque(maxlen=15)
            self._hip_history.append(hip_pos)
            if len(self._hip_history) >= 2 and len(self.wrist_history) >= 2:
                hip_vel = np.linalg.norm(
                    np.array(self._hip_history[-1]) - np.array(self._hip_history[-2])
                )
                wrist_vel = np.linalg.norm(
                    np.array(self.wrist_history[-1]) - np.array(self.wrist_history[-2])
                )
                # Walking = wrist and hips move at similar speed (ratio < 2)
                # Swinging = wrist moves much faster than hips (ratio > 3)
                # Only filter if hips are actually moving AND wrist isn't much faster
                if hip_vel > self.torso_length * 0.02 and self.state == 'IDLE':
                    ratio = wrist_vel / (hip_vel + 1e-8)
                    if ratio < 2.5:
                        # Wrist not much faster than hips — likely walking
                        return 'IDLE'

        # Calculate velocity
        if len(self.wrist_history) >= 2:
            prev = np.array(self.wrist_history[-2])
            curr = np.array(self.wrist_history[-1])
            velocity = np.linalg.norm(curr - prev)
        else:
            velocity = 0
        self.velocity_history.append(velocity)

        # Adaptive thresholds based on torso length (distance from camera)
        start_threshold = self.start_ratio * self.torso_length
        peak_threshold = self.peak_ratio * self.torso_length
        end_threshold = self.end_ratio * self.torso_length

        # Cooldown after a swing
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return 'IDLE'

        # State machine
        if self.state == 'IDLE':
            if velocity > start_threshold:
                self.state = 'SWINGING'
                self.swing_frame_count = 0
                self.swing_frames = []
                self.swing_keypoints = []
                self.swing_angles = []
                self.peak_velocity = velocity
                # Capture this frame
                if frame is not None:
                    self.swing_frames.append(frame.copy())
                if keypoints is not None:
                    self.swing_keypoints.append(keypoints)
                if angles is not None:
                    self.swing_angles.append(angles)
            return 'IDLE'

        elif self.state == 'SWINGING':
            # Capture frame
            self.swing_frame_count += 1
            if frame is not None:
                self.swing_frames.append(frame.copy())
            if keypoints is not None:
                self.swing_keypoints.append(keypoints)
            if angles is not None:
                self.swing_angles.append(angles)

            self.peak_velocity = max(self.peak_velocity, velocity)

            # Check if swing is over (velocity dropped)
            swing_len = self.swing_frame_count
            recent_vels = list(self.velocity_history)[-5:] if len(self.velocity_history) >= 5 else list(self.velocity_history)
            avg_recent_vel = np.mean(recent_vels)

            swing_ended = (
                swing_len >= self.min_swing_frames and
                avg_recent_vel < end_threshold
            )
            swing_too_long = swing_len >= self.max_swing_frames

            if swing_ended or swing_too_long:
                # Check if it was a real swing (reached peak threshold)
                if self.peak_velocity >= peak_threshold:
                    result = self._score_swing()
                    self.state = 'IDLE'
                    self.cooldown_counter = self.cooldown_frames
                    return result
                else:
                    # Just noise, not a real swing
                    self.state = 'IDLE'
                    self.swing_frames = []
                    self.swing_keypoints = []
                    self.swing_angles = []
                    return 'IDLE'

            return 'SWINGING'

        return self.state

    def _score_swing(self):
        """Score a completed swing and return the result."""
        if not self.swing_angles:
            return 'IDLE'

        # Find the contact frame (peak velocity within the swing)
        if len(self.swing_keypoints) >= 2:
            vels = []
            for i in range(1, len(self.swing_keypoints)):
                if self.swing_keypoints[i] and self.swing_keypoints[i-1]:
                    curr = np.array(self.swing_keypoints[i].get('right_wrist', (0, 0, 0, 0))[:2])
                    prev = np.array(self.swing_keypoints[i-1].get('right_wrist', (0, 0, 0, 0))[:2])
                    vels.append(np.linalg.norm(curr - prev))
                else:
                    vels.append(0)
            contact_idx = int(np.argmax(vels)) + 1 if vels else len(self.swing_keypoints) // 2
        else:
            contact_idx = 0

        # Get contact angles and best loading angles
        contact_angles = self.swing_angles[min(contact_idx, len(self.swing_angles) - 1)]

        # Find best loading angles (before contact)
        best_loading = {}
        for i in range(min(contact_idx, len(self.swing_angles))):
            a = self.swing_angles[i]
            if a:
                if a.get('shoulder_angle', 0) > best_loading.get('shoulder_angle', 0):
                    best_loading['shoulder_angle'] = a['shoulder_angle']
                if a.get('knee_angle', 180) < best_loading.get('knee_angle', 180):
                    best_loading['knee_angle'] = a['knee_angle']
                if a.get('racket_lag', 0) > best_loading.get('racket_lag', 0):
                    best_loading['racket_lag'] = a['racket_lag']

        # Get contact keypoints for stroke classification
        contact_keypoints = None
        if contact_idx < len(self.swing_keypoints) and self.swing_keypoints[contact_idx]:
            contact_keypoints = self.swing_keypoints[contact_idx]

        result = {
            'type': 'swing_complete',
            'frames': self.swing_frame_count,
            'duration_ms': int(self.swing_frame_count / self.fps * 1000),
            'peak_velocity': round(self.peak_velocity, 1),
            'contact_frame_idx': contact_idx,
            'contact_angles': contact_angles,
            'contact_keypoints': contact_keypoints,
            'loading_angles': best_loading if best_loading else None,
            'replay_frames': self.swing_frames[-30:] if self.swing_frames else [],
            'swing_number': len(self.completed_swings) + 1,
        }

        self.completed_swings.append(result)
        return result

    @property
    def swing_count(self):
        return len(self.completed_swings)

    @property
    def average_score(self):
        """Average score across all completed swings (if scored externally)."""
        scores = [s.get('score', 0) for s in self.completed_swings if 'score' in s]
        return np.mean(scores) if scores else 0

    @property
    def is_swinging(self):
        return self.state == 'SWINGING'
