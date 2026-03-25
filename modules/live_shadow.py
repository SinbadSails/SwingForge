"""
Live Shadow Swing Mode v2 — Ghost skeleton + voice coaching + swing detection.

Features:
    - Animated pro ghost skeleton overlay (blue) that loops through the swing
    - Your live skeleton (green) tracked in real-time
    - Phase-matched sync scoring (compares your phase to ghost's phase)
    - Auto swing detection: scores your swing when it detects contact
    - Voice coaching cues via local TTS
    - Session tracking with improvement trend

Controls:
    [P]     — cycle pros
    [V]     — toggle voice coaching
    [G]     — toggle ghost skeleton
    [H]     — toggle HUD
    [R]     — record session
    [+/-]   — ghost speed up/down
    [Q/ESC] — quit
"""

import cv2
import numpy as np
import json
import os
import time

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.pose_engine import PoseEngine, SKELETON_CONNECTIONS
from core.voice_coach import VoiceCoach

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


def draw_ghost_skeleton(frame, ghost_kp, hip_center, scale, alpha=0.5):
    """Draw the pro's ghost skeleton (blue) scaled and positioned on frame."""
    if ghost_kp is None:
        return frame

    overlay = frame.copy()
    color = (255, 191, 0)  # electric blue in BGR

    # Convert normalized keypoints to screen coordinates
    screen_kp = {}
    for name, data in ghost_kp.items():
        if data is None:
            continue
        x = data['x'] * scale + hip_center[0]
        y = data['y'] * scale + hip_center[1]
        screen_kp[name] = (int(x), int(y))

    # Draw connections
    for start, end in SKELETON_CONNECTIONS:
        if start in screen_kp and end in screen_kp:
            cv2.line(overlay, screen_kp[start], screen_kp[end], color, 2)

    # Draw joints
    for name, pt in screen_kp.items():
        cv2.circle(overlay, pt, 4, color, -1)

    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame


def draw_hud(frame, pro_name, phase, sync_scores, overall_sync, tip, fps,
             angles, pro_angles, ghost_frame, total_ghost_frames, ghost_speed,
             swing_count, swing_scores, voice_on, ghost_on):
    """Draw the full HUD with all info."""
    h, w = frame.shape[:2]

    # ── LEFT: Your Angles ──
    pw, ph = 280, 185
    ov = frame.copy()
    cv2.rectangle(ov, (8, 8), (8 + pw, 8 + ph), (13, 13, 13), -1)
    cv2.addWeighted(ov, 0.8, frame, 0.2, 0, frame)
    cv2.rectangle(frame, (8, 8), (8 + pw, 8 + ph), (0, 204, 255), 1)

    x0, y0 = 14, 28
    cv2.putText(frame, f"YOUR ANGLES  |  {fps:.0f} FPS",
                (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 204, 255), 1)
    y0 += 22

    if angles:
        for key, val in angles.items():
            if key == 'contact_height_ratio':
                continue
            nice = key.replace('_', ' ').title()
            pro_val = pro_angles.get(key)
            if pro_val:
                diff = abs(val - pro_val)
                color = (0, 255, 0) if diff < 15 else (0, 255, 255) if diff < 30 else (0, 0, 255)
            else:
                color = (200, 200, 200)
            cv2.putText(frame, f"{nice}: {val:.0f}", (x0, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1)
            y0 += 20
    else:
        cv2.putText(frame, "No pose - step back", (x0, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

    # ── RIGHT: Sync + Controls ──
    rw, rh = 300, 260
    rx = w - rw - 8
    ov2 = frame.copy()
    cv2.rectangle(ov2, (rx, 8), (rx + rw, 8 + rh), (13, 13, 13), -1)
    cv2.addWeighted(ov2, 0.8, frame, 0.2, 0, frame)
    cv2.rectangle(frame, (rx, 8), (rx + rw, 8 + rh), (204, 255, 0), 1)

    px, py = rx + 10, 28
    lh = 22

    cv2.putText(frame, f"COMPARE: {pro_name}",
                (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (204, 255, 0), 1)
    py += lh

    # Ghost progress bar
    if total_ghost_frames > 0:
        bar_w = rw - 20
        progress = ghost_frame / max(total_ghost_frames, 1)
        cv2.rectangle(frame, (px, py - 8), (px + bar_w, py), (50, 50, 50), -1)
        cv2.rectangle(frame, (px, py - 8), (px + int(bar_w * progress), py), (204, 255, 0), -1)
        cv2.putText(frame, f"Ghost: {ghost_frame}/{total_ghost_frames} ({ghost_speed:.0f}%)",
                    (px, py + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1)
    py += lh + 8

    # Sync bars
    for joint_name, score in sync_scores.items():
        nice = joint_name.replace('right_', 'R.').replace('_', ' ').title()
        bar_w = 100
        filled = int(bar_w * score / 100)
        c = (0, 255, 0) if score > 80 else (0, 255, 255) if score > 50 else (0, 0, 255)

        cv2.putText(frame, nice, (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
        bx = px + 95
        cv2.rectangle(frame, (bx, py - 9), (bx + bar_w, py + 1), (50, 50, 50), -1)
        cv2.rectangle(frame, (bx, py - 9), (bx + filled, py + 1), c, -1)
        cv2.putText(frame, f"{score:.0f}%", (bx + bar_w + 4, py),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, c, 1)
        py += lh

    py += 4
    sc = (0, 255, 0) if overall_sync > 75 else (0, 255, 255) if overall_sync > 50 else (0, 0, 255)
    cv2.putText(frame, f"SYNC: {overall_sync:.0f}/100",
                (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.6, sc, 2)
    py += lh + 2
    cv2.putText(frame, f"TIP: {tip}", (px, py),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    py += lh

    # Swing count and history
    if swing_scores:
        avg = np.mean(swing_scores[-10:])
        cv2.putText(frame, f"Swings: {swing_count}  Avg: {avg:.0f}/100",
                    (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (204, 255, 0), 1)

    # ── BOTTOM: Controls reminder ──
    controls = f"[P] pro  [G] ghost:{'ON' if ghost_on else 'OFF'}  [V] voice:{'ON' if voice_on else 'OFF'}  [+/-] speed  [Q] quit"
    cv2.putText(frame, controls, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1)

    return frame


def calculate_sync_score(user_kp, pro_kp_frame, pose_engine, side='right'):
    """Compare user keypoints to pro's keypoints at the SAME phase frame."""
    scores = {}
    if user_kp is None:
        return {j: 0.0 for j in SYNC_JOINTS}

    user_angles = pose_engine.get_joint_angles(user_kp, side=side)
    if user_angles is None:
        return {j: 0.0 for j in SYNC_JOINTS}

    # Get pro angles from the current ghost frame
    if pro_kp_frame is None:
        return {j: 50.0 for j in SYNC_JOINTS}

    # Convert pro keypoints to tuple format
    pro_tuples = {}
    for name, data in pro_kp_frame.items():
        if data is not None:
            pro_tuples[name] = (data['x'], data['y'], data['z'], data['visibility'])

    pro_angles = pose_engine.get_joint_angles(pro_tuples, side=side)
    if pro_angles is None:
        return {j: 50.0 for j in SYNC_JOINTS}

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


def detect_swing(wrist_history, threshold=25):
    """Detect if a swing just happened based on wrist velocity spike."""
    if len(wrist_history) < 10:
        return False

    recent = wrist_history[-5:]
    older = wrist_history[-10:-5]

    recent_speed = np.mean([abs(v) for v in recent])
    older_speed = np.mean([abs(v) for v in older])

    return recent_speed > threshold and recent_speed > older_speed * 2


def run_shadow_mode(playing_hand='right'):
    """Main loop — ghost skeleton + voice coaching + swing detection."""
    print("\n  SwingForge Shadow Mode v2 — Loading...")
    pose_engine = PoseEngine()
    voice_coach = VoiceCoach(enabled=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # State
    pro_idx = 0
    ghost_on = True
    voice_on = True
    show_hud = True
    recording = False
    recorder = None
    ghost_speed_pct = 50
    ghost_frame_idx = 0
    ghost_timer = time.time()

    session_scores = []
    swing_scores = []
    swing_count = 0
    wrist_history = []
    frame_count = 0
    fps = 0
    fps_timer = time.time()
    last_swing_time = 0

    # Load pro data
    pro_name, pro_file = PRO_FILES[pro_idx]
    pro_data = load_pro_data(pro_file)
    pro_angles = pro_data.get('contact_angles', {}) if pro_data else {}
    pro_sequence = pro_data.get('normalized_sequence', []) if pro_data else []
    pro_raw_sequence = pro_data.get('keypoint_sequence', []) if pro_data else []
    total_ghost_frames = len(pro_sequence)

    print(f"\n  SwingForge Shadow Mode v2")
    print(f"  ─────────────────────────")
    print(f"  [P] cycle pros  [G] ghost  [V] voice  [+/-] speed  [Q] quit")
    print(f"  Comparing: {pro_name} ({total_ghost_frames} frames)")
    if total_ghost_frames == 0:
        print(f"  (No keypoint sequence — using static angles only)")
    print()

    voice_coach.say(f"Shadow mode. Comparing to {pro_name.replace(' FH', ' forehand').replace(' BH', ' backhand')}.")

    cv2.namedWindow('SwingForge', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('SwingForge', 1280, 720)

    dropped_frames = 0
    total_loop_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            dropped_frames += 1
            if dropped_frames > 30:
                print(f"  Camera lost — {dropped_frames} consecutive failed reads. Exiting.")
                break
            continue
        dropped_frames = 0  # reset on successful read
        total_loop_frames += 1

        try:
            frame = cv2.flip(frame, 1)
        except Exception:
            continue
        h, w = frame.shape[:2]
        frame_count += 1

        # FPS
        now = time.time()
        if now - fps_timer >= 1.0:
            fps = frame_count / (now - fps_timer)
            frame_count = 0
            fps_timer = now

        # ── All rendering wrapped in try/except ──
        try:
            # Extract user pose
            user_kp = pose_engine.extract_keypoints(frame)
        except Exception:
            user_kp = None

        # ── Rendering (wrapped in try/except so one bad frame doesn't crash) ──
        current_ghost_kp = None
        sync_scores = {j: 0.0 for j in SYNC_JOINTS}
        overall = 0
        angles = None
        phase = 'ready'
        tip = 'Step into frame!'

        try:
            # Track wrist velocity for swing detection
            if user_kp and 'right_wrist' in user_kp:
                wrist_history.append(user_kp['right_wrist'][1])
                if len(wrist_history) > 30:
                    wrist_history = wrist_history[-30:]

            # Advance ghost skeleton — skip null frames automatically
            if ghost_on and total_ghost_frames > 0:
                ghost_interval = (1.0 / 30) / (ghost_speed_pct / 100.0)
                if now - ghost_timer > ghost_interval:
                    # Advance, but skip any null frames (up to 10 skips)
                    for _skip in range(10):
                        ghost_frame_idx = (ghost_frame_idx + 1) % total_ghost_frames
                        if pro_sequence[ghost_frame_idx] is not None:
                            break
                    ghost_timer = now

            # Get current ghost keypoints
            current_ghost_raw_kp = None
            if total_ghost_frames > 0 and ghost_frame_idx < total_ghost_frames:
                current_ghost_kp = pro_sequence[ghost_frame_idx]
                if ghost_frame_idx < len(pro_raw_sequence):
                    current_ghost_raw_kp = pro_raw_sequence[ghost_frame_idx]

            # Draw ghost skeleton
            if ghost_on and current_ghost_kp and user_kp:
                user_hip_x = (user_kp['left_hip'][0] + user_kp['right_hip'][0]) / 2
                user_hip_y = (user_kp['left_hip'][1] + user_kp['right_hip'][1]) / 2
                user_shoulder_y = (user_kp['left_shoulder'][1] + user_kp['right_shoulder'][1]) / 2
                user_torso = abs(user_hip_y - user_shoulder_y)
                if user_torso > 10:
                    frame = draw_ghost_skeleton(
                        frame, current_ghost_kp,
                        hip_center=(int(user_hip_x), int(user_hip_y)),
                        scale=user_torso, alpha=0.5)

            # Draw user skeleton (green)
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

            # Calculate sync score
            if current_ghost_raw_kp:
                sync_scores = calculate_sync_score(user_kp, current_ghost_raw_kp, pose_engine, playing_hand)
            elif user_kp:
                user_angles = pose_engine.get_joint_angles(user_kp, side=playing_hand)
                if user_angles:
                    for joint, akey in {'right_wrist': 'racket_lag', 'right_elbow': 'elbow_angle',
                                         'right_shoulder': 'shoulder_angle', 'right_hip': 'hip_rotation'}.items():
                        if akey in user_angles and akey in pro_angles:
                            diff = abs(user_angles[akey] - pro_angles[akey])
                            sync_scores[joint] = max(0, 100 - (diff / 60) * 100)

            overall = np.mean(list(sync_scores.values())) if sync_scores else 0
            angles = pose_engine.get_joint_angles(user_kp, side=playing_hand) if user_kp else None

            # Phase detection
            if user_kp:
                wy = user_kp['right_wrist'][1]
                hy = user_kp['right_hip'][1]
                sy = user_kp['right_shoulder'][1]
                if wy < sy:
                    phase = 'load'
                elif wy < hy:
                    phase = 'contact'
                else:
                    phase = 'follow'

            # Swing detection
            if detect_swing(wrist_history) and (now - last_swing_time) > 2.0:
                swing_count += 1
                swing_scores.append(overall)
                last_swing_time = now
                if voice_on:
                    voice_coach.announce_score(overall)

            # Voice coaching (periodic)
            if voice_on and user_kp and total_loop_frames % 120 == 0:
                voice_coach.coach_on_angles(sync_scores, overall)

            # Coaching tip
            worst = min(sync_scores, key=sync_scores.get) if sync_scores else ''
            tips = {'right_hip': 'Drive hips forward!', 'right_shoulder': 'Turn shoulders!',
                    'right_elbow': 'Extend elbow!', 'right_wrist': 'Check wrist lag!'}
            tip = tips.get(worst, 'Keep swinging!')

            # Draw HUD
            if show_hud:
                frame = draw_hud(frame, pro_name, phase, sync_scores, overall,
                                 tip, fps, angles, pro_angles,
                                 ghost_frame_idx, total_ghost_frames, ghost_speed_pct,
                                 swing_count, swing_scores, voice_on, ghost_on)

            # Recording
            if recording:
                cv2.circle(frame, (30, 30), 12, (0, 0, 255), -1)
                cv2.putText(frame, "REC", (48, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                if recorder:
                    recorder.write(frame)

            if user_kp:
                session_scores.append(overall)

        except Exception as e:
            if total_loop_frames < 5:
                print(f"  Render error (frame {total_loop_frames}): {type(e).__name__}: {e}")
                import traceback; traceback.print_exc()

        cv2.imshow('SwingForge', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('p'):
            pro_idx = (pro_idx + 1) % len(PRO_FILES)
            pro_name, pro_file = PRO_FILES[pro_idx]
            pro_data = load_pro_data(pro_file)
            pro_angles = pro_data.get('contact_angles', {}) if pro_data else {}
            pro_sequence = pro_data.get('normalized_sequence', []) if pro_data else []
            pro_raw_sequence = pro_data.get('keypoint_sequence', []) if pro_data else []
            total_ghost_frames = len(pro_sequence)
            ghost_frame_idx = 0
            print(f"  Switched to: {pro_name} ({total_ghost_frames} frames)")
            if voice_on:
                voice_coach.say(f"Switched to {pro_name}")
        elif key == ord('g'):
            ghost_on = not ghost_on
            print(f"  Ghost: {'ON' if ghost_on else 'OFF'}")
        elif key == ord('v'):
            voice_on = not voice_on
            voice_coach.enabled = voice_on
            print(f"  Voice: {'ON' if voice_on else 'OFF'}")
        elif key == ord('h'):
            show_hud = not show_hud
        elif key == ord('+') or key == ord('='):
            ghost_speed_pct = min(200, ghost_speed_pct + 25)
            print(f"  Ghost speed: {ghost_speed_pct}%")
        elif key == ord('-'):
            ghost_speed_pct = max(10, ghost_speed_pct - 25)
            print(f"  Ghost speed: {ghost_speed_pct}%")
        elif key == ord('r'):
            if not recording:
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
    voice_coach.stop()
    if session_scores:
        avg = np.mean(session_scores)
        print(f"\n  ── Session Summary ──")
        print(f"  Total frames: {len(session_scores)}")
        print(f"  Swings detected: {swing_count}")
        if swing_scores:
            print(f"  Swing scores: {', '.join(f'{s:.0f}' for s in swing_scores)}")
            print(f"  Average swing score: {np.mean(swing_scores):.1f}/100")
            print(f"  Best swing: {max(swing_scores):.0f}/100")
        print(f"  Overall avg sync: {avg:.1f}/100")

    cap.release()
    cv2.destroyAllWindows()
    pose_engine.release()


if __name__ == '__main__':
    run_shadow_mode()
