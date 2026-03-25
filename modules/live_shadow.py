"""
Live Shadow Swing Mode — OpenCV webcam window with ghost skeleton overlay.
User mirrors a pro player's swing in real time with per-joint sync scoring.

Controls:
    [SPACE] — start/stop ghost playback
    [P]     — cycle pros
    [S]     — slow ghost to 25% speed
    [F]     — fast (100%)
    [R]     — record session to .mp4
    [Q]     — quit
"""

import cv2
import numpy as np
import json
import os
import time

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.pose_engine import PoseEngine, LANDMARKS

PRO_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'pros')

PRO_FILES = [
    ('Djokovic FH', 'djokovic_forehand.json'),
    ('Alcaraz FH', 'alcaraz_forehand.json'),
    ('Federer Serve', 'federer_serve.json'),
    ('Nadal FH', 'nadal_forehand.json'),
    ('Medvedev BH', 'medvedev_backhand.json'),
]

# Joint groups to track for sync scoring
SYNC_JOINTS = ['right_wrist', 'right_elbow', 'right_shoulder', 'right_hip']


def load_pro_data(pro_file):
    """Load pro keypoint sequence or contact angles."""
    path = os.path.join(PRO_DATA_DIR, pro_file)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def draw_hud(frame, pro_name, speed_pct, phase, sync_scores, overall_sync, tip):
    """Draw the real-time HUD overlay."""
    h, w = frame.shape[:2]
    hud_w, hud_h = 320, 240
    hud_x, hud_y = w - hud_w - 10, 10

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h),
                  (13, 13, 13), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # Border
    cv2.rectangle(frame, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h),
                  (204, 255, 0), 1)

    x0 = hud_x + 12
    y0 = hud_y + 25
    line_h = 24

    cv2.putText(frame, f"SHADOW MODE: {pro_name}",
                (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (204, 255, 0), 1)
    y0 += line_h
    cv2.putText(frame, f"Speed: {speed_pct}%  Phase: {phase.upper()}",
                (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    y0 += line_h + 5

    # Per-joint sync bars
    for joint_name, score in sync_scores.items():
        nice = joint_name.replace('right_', 'R. ').replace('_', ' ').title()
        # Score bar
        bar_w = 120
        filled = int(bar_w * score / 100)
        color = (0, 255, 0) if score > 80 else (0, 255, 255) if score > 50 else (0, 0, 255)
        warn = " !" if score < 50 else ""

        cv2.putText(frame, f"{nice:<14}", (x0, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        bar_x = x0 + 110
        cv2.rectangle(frame, (bar_x, y0 - 10), (bar_x + bar_w, y0), (50, 50, 50), -1)
        cv2.rectangle(frame, (bar_x, y0 - 10), (bar_x + filled, y0), color, -1)
        cv2.putText(frame, f"{score:.0f}%{warn}", (bar_x + bar_w + 5, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        y0 += line_h

    y0 += 5
    cv2.putText(frame, f"SYNC SCORE: {overall_sync:.0f}/100",
                (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (204, 255, 0), 2)
    y0 += line_h
    cv2.putText(frame, f"FOCUS: {tip}",
                (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    return frame


def calculate_sync_score(user_kp, pro_angles, pose_engine, side='right'):
    """Calculate per-joint sync scores comparing user's current pose to pro's contact angles."""
    scores = {}
    if user_kp is None or not pro_angles:
        for j in SYNC_JOINTS:
            scores[j] = 0.0
        return scores

    user_angles = pose_engine.get_joint_angles(user_kp, side=side)
    if user_angles is None:
        for j in SYNC_JOINTS:
            scores[j] = 0.0
        return scores

    angle_map = {
        'right_wrist': 'racket_lag',
        'right_elbow': 'elbow_angle',
        'right_shoulder': 'shoulder_angle',
        'right_hip': 'hip_rotation',
    }

    for joint, angle_key in angle_map.items():
        if angle_key in user_angles and angle_key in pro_angles:
            diff = abs(user_angles[angle_key] - pro_angles[angle_key])
            # Score: 100 at 0° diff, 0 at 60°+ diff
            score = max(0, 100 - (diff / 60) * 100)
            scores[joint] = score
        else:
            scores[joint] = 50.0

    return scores


def run_shadow_mode(playing_hand='right'):
    """Main loop for live shadow swing mode."""
    pose_engine = PoseEngine()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # State
    pro_idx = 0
    speed_pct = 50
    playing = False
    recording = False
    recorder = None
    session_scores = []

    pro_name, pro_file = PRO_FILES[pro_idx]
    pro_data = load_pro_data(pro_file)
    pro_angles = pro_data.get('contact_angles', {}) if pro_data else {}

    print("SwingForge Shadow Mode")
    print("[SPACE] play/pause | [P] cycle pros | [S] slow | [F] fast | [R] record | [Q] quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # mirror

        # Extract user pose
        user_kp = pose_engine.extract_keypoints(frame)

        # Draw user skeleton
        pose_engine.draw_skeleton(frame, user_kp, color=(0, 255, 127), thickness=2)

        # Calculate sync
        sync_scores = calculate_sync_score(user_kp, pro_angles, pose_engine, playing_hand)
        overall = np.mean(list(sync_scores.values()))

        # Get coaching tip
        worst_joint = min(sync_scores, key=sync_scores.get) if sync_scores else ''
        tips = {
            'right_hip': 'Drive your hips!',
            'right_shoulder': 'Turn those shoulders!',
            'right_elbow': 'Extend the elbow!',
            'right_wrist': 'Check your wrist position!',
        }
        tip = tips.get(worst_joint, 'Keep swinging!')

        phase = 'ready'
        if user_kp:
            # Simple phase detection from wrist position
            r_wrist_y = user_kp['right_wrist'][1]
            r_hip_y = user_kp['right_hip'][1]
            r_shoulder_y = user_kp['right_shoulder'][1]
            if r_wrist_y < r_shoulder_y:
                phase = 'load'
            elif r_wrist_y < r_hip_y:
                phase = 'contact'
            else:
                phase = 'follow'

        # Draw HUD
        frame = draw_hud(frame, pro_name, speed_pct, phase,
                         sync_scores, overall, tip)

        # Recording indicator
        if recording:
            cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (45, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            if recorder:
                recorder.write(frame)

        # Track session scores
        if user_kp:
            session_scores.append(overall)

        cv2.imshow('SwingForge Shadow Mode', frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            playing = not playing
        elif key == ord('p'):
            pro_idx = (pro_idx + 1) % len(PRO_FILES)
            pro_name, pro_file = PRO_FILES[pro_idx]
            pro_data = load_pro_data(pro_file)
            pro_angles = pro_data.get('contact_angles', {}) if pro_data else {}
            print(f"Switched to: {pro_name}")
        elif key == ord('s'):
            speed_pct = max(25, speed_pct - 25)
            print(f"Speed: {speed_pct}%")
        elif key == ord('f'):
            speed_pct = min(100, speed_pct + 25)
            print(f"Speed: {speed_pct}%")
        elif key == ord('r'):
            if not recording:
                h, w = frame.shape[:2]
                out_path = f"shadow_session_{int(time.time())}.mp4"
                recorder = cv2.VideoWriter(out_path,
                                            cv2.VideoWriter_fourcc(*'mp4v'),
                                            30, (w, h))
                recording = True
                print(f"Recording to {out_path}")
            else:
                recording = False
                if recorder:
                    recorder.release()
                    recorder = None
                print("Recording stopped.")

    # Session summary
    if session_scores:
        avg = np.mean(session_scores)
        print(f"\nSession Summary:")
        print(f"  Swings analyzed: {len(session_scores)}")
        print(f"  Average sync score: {avg:.1f}/100")

    cap.release()
    cv2.destroyAllWindows()
    pose_engine.release()
