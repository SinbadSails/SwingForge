"""
Live Shadow Swing Mode — OpenCV webcam window, zero latency.
Direct GPU-accelerated pose estimation, no web layer.

Controls:
    [P]     — cycle pros (Djokovic → Alcaraz → Federer → Nadal → Medvedev)
    [R]     — record session to .mp4
    [H]     — toggle HUD on/off
    [Q]     — quit
    [ESC]   — quit
"""

import cv2
import numpy as np
import json
import os
import time

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.pose_engine import PoseEngine, SKELETON_CONNECTIONS

PRO_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'pros')

PRO_FILES = [
    ('Djokovic FH', 'djokovic_forehand.json'),
    ('Alcaraz FH', 'alcaraz_forehand.json'),
    ('Federer Serve', 'federer_serve.json'),
    ('Nadal FH', 'nadal_forehand.json'),
    ('Medvedev BH', 'medvedev_backhand.json'),
]

SYNC_JOINTS = ['right_wrist', 'right_elbow', 'right_shoulder', 'right_hip']


def load_pro_data(pro_file):
    path = os.path.join(PRO_DATA_DIR, pro_file)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def draw_hud(frame, pro_name, phase, sync_scores, overall_sync, tip, fps,
             angles, pro_angles):
    """Draw the real-time HUD overlay with angles and sync bars."""
    h, w = frame.shape[:2]

    # ── LEFT PANEL: Joint Angles ──
    panel_w, panel_h = 290, 180
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (8 + panel_w, 8 + panel_h), (13, 13, 13), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    cv2.rectangle(frame, (8, 8), (8 + panel_w, 8 + panel_h), (0, 204, 255), 1)

    x0, y0 = 16, 30
    cv2.putText(frame, f"YOUR ANGLES  |  {fps:.0f} FPS",
                (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 204, 255), 1)
    y0 += 24

    if angles:
        for key, val in angles.items():
            if key == 'contact_height_ratio':
                continue
            nice = key.replace('_', ' ').title()
            # Color based on how close to pro
            pro_val = pro_angles.get(key)
            if pro_val is not None:
                diff = abs(val - pro_val)
                color = (0, 255, 0) if diff < 15 else (0, 255, 255) if diff < 30 else (0, 0, 255)
            else:
                color = (200, 200, 200)
            cv2.putText(frame, f"{nice}: {val:.0f} deg", (x0, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)
            y0 += 22
    else:
        cv2.putText(frame, "No pose - step into frame", (x0, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # ── RIGHT PANEL: Sync Score HUD ──
    hud_w, hud_h = 310, 230
    hud_x = w - hud_w - 8
    hud_y = 8
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h),
                  (13, 13, 13), -1)
    cv2.addWeighted(overlay2, 0.8, frame, 0.2, 0, frame)
    cv2.rectangle(frame, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h),
                  (204, 255, 0), 1)

    px, py = hud_x + 12, hud_y + 25
    line_h = 24

    cv2.putText(frame, f"COMPARE: {pro_name}",
                (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (204, 255, 0), 1)
    py += line_h
    cv2.putText(frame, f"Phase: {phase.upper()}",
                (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)
    py += line_h + 4

    # Per-joint sync bars
    for joint_name, score in sync_scores.items():
        nice = joint_name.replace('right_', 'R.').replace('_', ' ').title()
        bar_w = 110
        filled = int(bar_w * score / 100)
        color = (0, 255, 0) if score > 80 else (0, 255, 255) if score > 50 else (0, 0, 255)
        warn = " !" if score < 50 else ""

        cv2.putText(frame, f"{nice}", (px, py),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
        bx = px + 105
        cv2.rectangle(frame, (bx, py - 10), (bx + bar_w, py), (50, 50, 50), -1)
        cv2.rectangle(frame, (bx, py - 10), (bx + filled, py), color, -1)
        cv2.putText(frame, f"{score:.0f}%{warn}", (bx + bar_w + 5, py),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        py += line_h

    py += 4
    # Big sync score
    sc_color = (0, 255, 0) if overall_sync > 75 else (0, 255, 255) if overall_sync > 50 else (0, 0, 255)
    cv2.putText(frame, f"SYNC: {overall_sync:.0f}/100",
                (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.65, sc_color, 2)
    py += line_h + 2
    cv2.putText(frame, f"TIP: {tip}",
                (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

    return frame


def calculate_sync_score(user_kp, pro_angles, pose_engine, side='right'):
    scores = {}
    if user_kp is None or not pro_angles:
        return {j: 0.0 for j in SYNC_JOINTS}

    user_angles = pose_engine.get_joint_angles(user_kp, side=side)
    if user_angles is None:
        return {j: 0.0 for j in SYNC_JOINTS}

    angle_map = {
        'right_wrist': 'racket_lag',
        'right_elbow': 'elbow_angle',
        'right_shoulder': 'shoulder_angle',
        'right_hip': 'hip_rotation',
    }

    for joint, angle_key in angle_map.items():
        if angle_key in user_angles and angle_key in pro_angles:
            diff = abs(user_angles[angle_key] - pro_angles[angle_key])
            scores[joint] = max(0, 100 - (diff / 60) * 100)
        else:
            scores[joint] = 50.0
    return scores


def run_shadow_mode(playing_hand='right'):
    """Main loop — direct OpenCV, zero latency."""
    print("\nSwingForge Shadow Mode — Loading...")
    pose_engine = PoseEngine()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Request high FPS from camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # State
    pro_idx = 0
    recording = False
    recorder = None
    show_hud = True
    session_scores = []
    frame_count = 0
    fps = 0
    fps_timer = time.time()

    pro_name, pro_file = PRO_FILES[pro_idx]
    pro_data = load_pro_data(pro_file)
    pro_angles = pro_data.get('contact_angles', {}) if pro_data else {}

    print("\n  SwingForge Shadow Mode — LIVE")
    print("  ────────────────────────────")
    print("  [P] cycle pros  [R] record  [H] toggle HUD  [Q] quit")
    print(f"  Comparing against: {pro_name}\n")

    cv2.namedWindow('SwingForge', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('SwingForge', 1280, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # mirror
        frame_count += 1

        # FPS counter
        now = time.time()
        if now - fps_timer >= 1.0:
            fps = frame_count / (now - fps_timer)
            frame_count = 0
            fps_timer = now

        # Extract pose
        user_kp = pose_engine.extract_keypoints(frame)

        # Draw skeleton
        if user_kp:
            for start, end in SKELETON_CONNECTIONS:
                if start in user_kp and end in user_kp:
                    p1 = (int(user_kp[start][0]), int(user_kp[start][1]))
                    p2 = (int(user_kp[end][0]), int(user_kp[end][1]))
                    vis1 = user_kp[start][3] if len(user_kp[start]) > 3 else 1.0
                    vis2 = user_kp[end][3] if len(user_kp[end]) > 3 else 1.0
                    if vis1 > 0.5 and vis2 > 0.5:
                        cv2.line(frame, p1, p2, (0, 255, 127), 3)

            for name, kp in user_kp.items():
                vis = kp[3] if len(kp) > 3 else 1.0
                if vis > 0.5:
                    cv2.circle(frame, (int(kp[0]), int(kp[1])), 5, (0, 255, 255), -1)

        # Calculate sync + angles
        angles = pose_engine.get_joint_angles(user_kp, side=playing_hand) if user_kp else None
        sync_scores = calculate_sync_score(user_kp, pro_angles, pose_engine, playing_hand)
        overall = np.mean(list(sync_scores.values()))

        # Phase detection
        phase = 'ready'
        if user_kp:
            r_wrist_y = user_kp['right_wrist'][1]
            r_hip_y = user_kp['right_hip'][1]
            r_shoulder_y = user_kp['right_shoulder'][1]
            if r_wrist_y < r_shoulder_y:
                phase = 'load'
            elif r_wrist_y < r_hip_y:
                phase = 'contact'
            else:
                phase = 'follow'

        # Coaching tip
        worst = min(sync_scores, key=sync_scores.get) if sync_scores else ''
        tips = {
            'right_hip': 'Drive your hips forward!',
            'right_shoulder': 'Turn those shoulders!',
            'right_elbow': 'Extend through contact!',
            'right_wrist': 'Check wrist lag!',
        }
        tip = tips.get(worst, 'Keep swinging!')

        # Draw HUD
        if show_hud:
            frame = draw_hud(frame, pro_name, phase, sync_scores, overall,
                             tip, fps, angles, pro_angles)

        # Recording indicator
        if recording:
            cv2.circle(frame, (30, 30), 12, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (48, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            if recorder:
                recorder.write(frame)

        if user_kp:
            session_scores.append(overall)

        cv2.imshow('SwingForge', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # Q or ESC
            break
        elif key == ord('p'):
            pro_idx = (pro_idx + 1) % len(PRO_FILES)
            pro_name, pro_file = PRO_FILES[pro_idx]
            pro_data = load_pro_data(pro_file)
            pro_angles = pro_data.get('contact_angles', {}) if pro_data else {}
            print(f"  Switched to: {pro_name}")
        elif key == ord('h'):
            show_hud = not show_hud
        elif key == ord('r'):
            if not recording:
                h, w = frame.shape[:2]
                out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                         f"session_{int(time.time())}.mp4")
                recorder = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                            30, (w, h))
                recording = True
                print(f"  Recording → {out_path}")
            else:
                recording = False
                if recorder:
                    recorder.release()
                    recorder = None
                print("  Recording stopped.")

    # Session summary
    if session_scores:
        avg = np.mean(session_scores)
        print(f"\n  ── Session Summary ──")
        print(f"  Frames analyzed: {len(session_scores)}")
        print(f"  Average sync score: {avg:.1f}/100")
        if avg > 80:
            print("  Great session! Your form is locked in.")
        elif avg > 60:
            print("  Solid work — focus on your weakest joint next time.")
        else:
            print("  Keep grinding — consistency comes with reps.")

    cap.release()
    cv2.destroyAllWindows()
    pose_engine.release()


if __name__ == '__main__':
    run_shadow_mode()
