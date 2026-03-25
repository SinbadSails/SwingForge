#!/usr/bin/env python3
"""
Extract pro player keypoint sequences from YouTube videos.
Downloads a clip, runs MediaPipe on every frame, saves as JSON.

Usage:
    python tools/extract_pro_keypoints.py --url "YOUTUBE_URL" --name "djokovic_forehand" --start 5 --end 8

    --url     YouTube video URL (slow-mo side view of a stroke)
    --name    Output filename (saved to data/pros/<name>.json)
    --start   Start time in seconds
    --end     End time in seconds
"""

import argparse
import cv2
import json
import os
import sys
import numpy as np
import subprocess
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from core.pose_engine import PoseEngine, LANDMARKS, SKELETON_CONNECTIONS


def download_clip(url, start, end, output_path):
    """Download a segment of a YouTube video using yt-dlp + ffmpeg."""
    # First download the full video to temp
    temp_full = tempfile.mktemp(suffix='.mp4')

    cmd = [
        sys.executable, '-m', 'yt_dlp',
        '--format', 'best[height<=720]',
        '--output', temp_full,
        '--quiet',
        url
    ]
    print(f"  Downloading video...")
    subprocess.run(cmd, check=True)

    # Trim with ffmpeg
    duration = end - start
    cmd_trim = [
        'ffmpeg', '-y', '-ss', str(start), '-i', temp_full,
        '-t', str(duration), '-c', 'copy', output_path
    ]
    print(f"  Trimming {start}s to {end}s...")
    subprocess.run(cmd_trim, check=True, capture_output=True)

    os.remove(temp_full)
    return output_path


def extract_keypoints_from_video(video_path, pose_engine):
    """Run MediaPipe on every frame, return list of keypoint dicts."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"  Processing {total_frames} frames at {fps:.0f} FPS...")

    all_keypoints = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        kp = pose_engine.extract_keypoints(frame)

        if kp is not None:
            # Convert to serializable format
            kp_dict = {}
            for name, (x, y, z, vis) in kp.items():
                kp_dict[name] = {
                    'x': round(float(x), 2),
                    'y': round(float(y), 2),
                    'z': round(float(z), 4),
                    'visibility': round(float(vis), 3)
                }
            all_keypoints.append({
                'frame': frame_idx,
                'keypoints': kp_dict
            })
        else:
            all_keypoints.append({
                'frame': frame_idx,
                'keypoints': None
            })

        frame_idx += 1
        if frame_idx % 10 == 0:
            print(f"    Frame {frame_idx}/{total_frames}")

    cap.release()
    return all_keypoints, fps


def compute_angles_at_frame(kp_dict, pose_engine, side='right'):
    """Compute joint angles from a keypoint dict."""
    if kp_dict is None:
        return None

    # Convert back to tuple format for pose_engine
    kp_tuples = {}
    for name, data in kp_dict.items():
        kp_tuples[name] = (data['x'], data['y'], data['z'], data['visibility'])

    return pose_engine.get_joint_angles(kp_tuples, side=side)


def find_contact_frame(keypoints_seq, pose_engine):
    """Find the contact frame by detecting peak wrist velocity."""
    velocities = []
    for i in range(1, len(keypoints_seq)):
        curr_kp = keypoints_seq[i]['keypoints']
        prev_kp = keypoints_seq[i-1]['keypoints']

        if curr_kp and prev_kp and 'right_wrist' in curr_kp and 'right_wrist' in prev_kp:
            dx = curr_kp['right_wrist']['x'] - prev_kp['right_wrist']['x']
            dy = curr_kp['right_wrist']['y'] - prev_kp['right_wrist']['y']
            vel = np.sqrt(dx**2 + dy**2)
            velocities.append((i, vel))
        else:
            velocities.append((i, 0))

    if not velocities:
        return len(keypoints_seq) // 2

    peak_frame = max(velocities, key=lambda x: x[1])[0]
    return peak_frame


def normalize_keypoints(keypoints_seq):
    """Normalize keypoints by hip midpoint and torso length for comparison."""
    normalized = []

    for frame_data in keypoints_seq:
        kp = frame_data['keypoints']
        if kp is None:
            normalized.append({'frame': frame_data['frame'], 'keypoints': None})
            continue

        # Hip midpoint as origin
        hip_mid_x = (kp['left_hip']['x'] + kp['right_hip']['x']) / 2
        hip_mid_y = (kp['left_hip']['y'] + kp['right_hip']['y']) / 2

        # Torso length for scale (hip mid to shoulder mid)
        shoulder_mid_x = (kp['left_shoulder']['x'] + kp['right_shoulder']['x']) / 2
        shoulder_mid_y = (kp['left_shoulder']['y'] + kp['right_shoulder']['y']) / 2
        torso_len = np.sqrt((shoulder_mid_x - hip_mid_x)**2 + (shoulder_mid_y - hip_mid_y)**2)

        if torso_len < 1:
            torso_len = 1  # avoid division by zero

        norm_kp = {}
        for name, data in kp.items():
            norm_kp[name] = {
                'x': round((data['x'] - hip_mid_x) / torso_len, 4),
                'y': round((data['y'] - hip_mid_y) / torso_len, 4),
                'z': data['z'],
                'visibility': data['visibility']
            }

        normalized.append({
            'frame': frame_data['frame'],
            'keypoints': norm_kp
        })

    return normalized


def main():
    parser = argparse.ArgumentParser(description='Extract pro player keypoints from YouTube')
    parser.add_argument('--url', type=str, required=True, help='YouTube URL')
    parser.add_argument('--name', type=str, required=True, help='Output name (e.g. djokovic_forehand)')
    parser.add_argument('--start', type=float, default=0, help='Start time (seconds)')
    parser.add_argument('--end', type=float, default=5, help='End time (seconds)')
    parser.add_argument('--side', type=str, default='right', help='Playing hand')
    parser.add_argument('--player', type=str, default='', help='Player name')
    parser.add_argument('--stroke', type=str, default='forehand', help='Stroke type')
    parser.add_argument('--local', type=str, default=None, help='Use local video file instead of YouTube')
    args = parser.parse_args()

    # Setup
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'pros')
    os.makedirs(output_dir, exist_ok=True)

    pose_engine = PoseEngine()

    # Get video
    if args.local:
        video_path = args.local
        print(f"Using local video: {video_path}")
    else:
        video_path = os.path.join(output_dir, f'{args.name}_clip.mp4')
        download_clip(args.url, args.start, args.end, video_path)

    # Extract keypoints
    keypoints_seq, fps = extract_keypoints_from_video(video_path, pose_engine)

    # Find contact frame
    contact_idx = find_contact_frame(keypoints_seq, pose_engine)
    print(f"  Contact frame detected: {contact_idx}")

    # Compute angles at contact
    contact_kp = keypoints_seq[contact_idx]['keypoints']
    contact_angles = compute_angles_at_frame(contact_kp, pose_engine, args.side)

    # Compute loading angles (30% before contact)
    loading_idx = max(0, int(contact_idx * 0.6))
    loading_kp = keypoints_seq[loading_idx]['keypoints']
    loading_angles = compute_angles_at_frame(loading_kp, pose_engine, args.side)

    # Normalize for overlay
    normalized_seq = normalize_keypoints(keypoints_seq)

    # Build output JSON
    output = {
        'player': args.player or args.name.split('_')[0].capitalize(),
        'stroke': args.stroke,
        'hand': args.side,
        'fps': fps,
        'total_frames': len(keypoints_seq),
        'contact_frame': contact_idx,
        'contact_angles': {k: round(v, 1) for k, v in contact_angles.items()} if contact_angles else {},
        'loading_angles': {k: round(v, 1) for k, v in loading_angles.items()} if loading_angles else {},
        'keypoint_sequence': [f['keypoints'] for f in keypoints_seq],
        'normalized_sequence': [f['keypoints'] for f in normalized_seq],
    }

    output_path = os.path.join(output_dir, f'{args.name}.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n  Saved to: {output_path}")
    print(f"  Frames: {len(keypoints_seq)}, FPS: {fps:.0f}")
    print(f"  Contact frame: {contact_idx}")
    if contact_angles:
        print(f"  Contact angles:")
        for k, v in contact_angles.items():
            print(f"    {k}: {v:.1f}")

    pose_engine.release()
    print("  Done!")


if __name__ == '__main__':
    main()
