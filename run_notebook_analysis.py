#!/usr/bin/env python3
"""
Soccer Ball Touch Analysis - Notebook Version
This script runs the same analysis as the Jupyter notebook for demonstration
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path
from dataclasses import dataclass
from ultralytics import YOLO

# Configuration
@dataclass
class Config:
    conf_detection: float = 0.15
    conf_pose: float = 0.3
    imgsz: int = 1280
    touch_distance_px: int = 75
    min_speed_threshold: float = 1.5
    debounce_frames: int = 8
    ball_spin_threshold: float = 0.3
    optical_flow_area: int = 20

class SimpleTracker:
    def __init__(self, max_disappeared: int = 30):
        self.next_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
    
    def register(self, bbox):
        self.objects[self.next_id] = bbox
        self.disappeared[self.next_id] = 0
        self.next_id += 1
    
    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]
    
    def calculate_iou(self, box1, box2):
        x1, y1, x2, y2 = box1
        x1g, y1g, x2g, y2g = box2
        
        xi1, yi1 = max(x1, x1g), max(y1, y1g)
        xi2, yi2 = min(x2, x2g), min(y2, y2g)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0
        
        inter_area = (xi2 - xi1) * (yi2 - yi1)
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2g - x1g) * (y2g - y1g)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def update(self, rects):
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return {}
        
        if len(self.objects) == 0:
            for rect in rects:
                self.register(rect)
        else:
            object_ids = list(self.objects.keys())
            
            ious = np.zeros((len(object_ids), len(rects)))
            for i, object_bbox in enumerate(self.objects.values()):
                for j, rect in enumerate(rects):
                    ious[i, j] = self.calculate_iou(object_bbox, rect)
            
            rows = ious.max(axis=1).argsort()[::-1]
            cols = ious.argmax(axis=1)[rows]
            
            used_row_indices = set()
            used_col_indices = set()
            
            for (row, col) in zip(rows, cols):
                if row in used_row_indices or col in used_col_indices:
                    continue
                
                if ious[row, col] > 0.3:
                    object_id = object_ids[row]
                    self.objects[object_id] = rects[col]
                    self.disappeared[object_id] = 0
                    
                    used_row_indices.add(row)
                    used_col_indices.add(col)
            
            unused_row_indices = set(range(0, ious.shape[0])).difference(used_row_indices)
            unused_col_indices = set(range(0, ious.shape[1])).difference(used_col_indices)
            
            if ious.shape[0] >= ious.shape[1]:
                for row in unused_row_indices:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                for col in unused_col_indices:
                    self.register(rects[col])
        
        return self.objects.copy()

class TouchCounter:
    def __init__(self, config: Config):
        self.config = config
        self.last_touch_frame = {}
        self.touch_events = []
        self.player_positions = {}
        self.ball_positions = []
    
    def detect_touches(self, frame_num: int, ball_center, player_ankles: dict):
        if not ball_center or not player_ankles:
            return []
        
        touches = []
        self.ball_positions.append((frame_num, ball_center))
        
        ball_speed = 0
        if len(self.ball_positions) >= 2:
            prev_pos = self.ball_positions[-2][1]
            curr_pos = ball_center
            ball_speed = np.sqrt((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)
        
        for player_id, ankles in player_ankles.items():
            left_ankle, right_ankle = ankles
            
            for ankle_side, ankle_pos in [('left', left_ankle), ('right', right_ankle)]:
                if ankle_pos is None:
                    continue
                
                distance = np.sqrt((ball_center[0] - ankle_pos[0])**2 + (ball_center[1] - ankle_pos[1])**2)
                
                if (distance < self.config.touch_distance_px and 
                    ball_speed > self.config.min_speed_threshold):
                    
                    last_touch_key = f"{player_id}_{ankle_side}"
                    if (last_touch_key not in self.last_touch_frame or 
                        frame_num - self.last_touch_frame[last_touch_key] > self.config.debounce_frames):
                        
                        touch_event = {
                            'frame': frame_num,
                            'player_id': player_id,
                            'leg': ankle_side,
                            'ball_pos': ball_center,
                            'ankle_pos': ankle_pos,
                            'distance': distance,
                            'ball_speed': ball_speed
                        }
                        
                        touches.append(touch_event)
                        self.touch_events.append(touch_event)
                        self.last_touch_frame[last_touch_key] = frame_num
        
        return touches

def extract_ankle_positions(pose_results):
    player_ankles = {}
    
    for i, pose in enumerate(pose_results):
        if pose.keypoints is not None and len(pose.keypoints.data) > 0:
            keypoints = pose.keypoints.data[0]
            
            left_ankle = keypoints[15][:2] if len(keypoints) > 15 and keypoints[15][2] > 0.3 else None
            right_ankle = keypoints[16][:2] if len(keypoints) > 16 and keypoints[16][2] > 0.3 else None
            
            if left_ankle is not None:
                left_ankle = (int(left_ankle[0]), int(left_ankle[1]))
            if right_ankle is not None:
                right_ankle = (int(right_ankle[0]), int(right_ankle[1]))
            
            player_ankles[i] = (left_ankle, right_ankle)
    
    return player_ankles

def main():
    print("🚀 Soccer Ball Touch Analysis - Notebook Demo")
    print("=" * 50)
    
    # Configuration
    config = Config()
    print(f"⚙️ Configuration:")
    print(f"   • Touch distance: {config.touch_distance_px} pixels")
    print(f"   • Speed threshold: {config.min_speed_threshold} px/frame")
    print(f"   • Debounce frames: {config.debounce_frames}")
    
    # Initialize models
    print("\n🎯 Loading YOLO models...")
    detection_model = YOLO('yolov8n.pt')
    pose_model = YOLO('yolov8n-pose.pt')
    print(f"✅ Models loaded successfully")
    
    # Initialize tracking
    ball_tracker = SimpleTracker()
    player_tracker = SimpleTracker()
    touch_counter = TouchCounter(config)
    
    # Video analysis
    video_path = "data/input_720.mp4"
    output_path = "outputs/annotated_notebook_demo.mp4"
    
    print(f"\n🎬 Analyzing video: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Error: Could not open video")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📊 Video properties: {width}x{height} @ {fps:.1f}fps ({total_frames} frames)")
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_num = 0
    all_touches = []
    progress_interval = max(1, total_frames // 20)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_num % progress_interval == 0:
            progress = (frame_num / total_frames) * 100
            print(f"📊 Progress: {progress:.1f}% (Frame {frame_num}/{total_frames})")
        
        # Object detection
        detection_results = detection_model(frame, conf=config.conf_detection, imgsz=config.imgsz)
        pose_results = pose_model(frame, conf=config.conf_pose, imgsz=config.imgsz)
        
        # Extract detections
        ball_center = None
        ball_bbox = None
        player_bboxes = []
        
        for result in detection_results:
            if result.boxes is not None:
                for box in result.boxes:
                    cls = int(box.cls[0])
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    if cls == 32:  # Ball
                        ball_center = ((x1 + x2) / 2, (y1 + y2) / 2)
                        ball_bbox = (x1, y1, x2, y2)
                    elif cls == 0:  # Person
                        player_bboxes.append((x1, y1, x2, y2))
        
        # Update trackers
        ball_tracks = ball_tracker.update([ball_bbox] if ball_center else [])
        player_tracks = player_tracker.update(player_bboxes)
        
        # Extract ankle positions
        player_ankles = extract_ankle_positions(pose_results)
        
        # Detect touches
        touches = touch_counter.detect_touches(frame_num, ball_center, player_ankles)
        all_touches.extend(touches)
        
        # Draw annotations
        annotated_frame = frame.copy()
        
        # Draw ball
        if ball_center:
            cv2.circle(annotated_frame, (int(ball_center[0]), int(ball_center[1])), 10, (0, 255, 0), -1)
            cv2.putText(annotated_frame, "BALL", (int(ball_center[0] - 20), int(ball_center[1] - 15)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw player ankles
        for player_id, (left_ankle, right_ankle) in player_ankles.items():
            if left_ankle:
                cv2.circle(annotated_frame, left_ankle, 8, (255, 0, 0), -1)
                cv2.putText(annotated_frame, "L", (left_ankle[0] - 10, left_ankle[1] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            if right_ankle:
                cv2.circle(annotated_frame, right_ankle, 8, (0, 0, 255), -1)
                cv2.putText(annotated_frame, "R", (right_ankle[0] - 10, right_ankle[1] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Draw touches
        for touch in touches:
            cv2.circle(annotated_frame, (int(touch['ball_pos'][0]), int(touch['ball_pos'][1])), 25, (0, 255, 255), 3)
            cv2.putText(annotated_frame, f"TOUCH-{touch['leg'].upper()}", 
                       (int(touch['ball_pos'][0] - 40), int(touch['ball_pos'][1] - 30)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Draw frame info
        cv2.putText(annotated_frame, f"Frame: {frame_num}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"Touches: {len(all_touches)}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(annotated_frame)
        frame_num += 1
    
    cap.release()
    out.release()
    
    # Results analysis
    print(f"\n✅ Analysis complete!")
    print(f"📊 Total touches detected: {len(all_touches)}")
    print(f"🎬 Annotated video saved: {output_path}")
    
    if all_touches:
        left_touches = [t for t in all_touches if t['leg'] == 'left']
        right_touches = [t for t in all_touches if t['leg'] == 'right']
        
        print(f"\n🎯 Touch Summary:")
        print(f"   • Left leg touches: {len(left_touches)}")
        print(f"   • Right leg touches: {len(right_touches)}")
        
        # Save results
        df_touches = pd.DataFrame(all_touches)
        df_touches['time_seconds'] = df_touches['frame'] / fps
        
        output_csv = "outputs/touch_events_notebook_demo.csv"
        df_touches.to_csv(output_csv, index=False)
        print(f"📄 Detailed results: {output_csv}")
        
        summary = {
            "total_touches": len(all_touches),
            "left_leg_touches": len(left_touches),
            "right_leg_touches": len(right_touches),
            "avg_ball_speed": float(df_touches['ball_speed'].mean()),
            "avg_touch_distance": float(df_touches['distance'].mean()),
            "analysis_config": {
                "touch_distance_threshold": config.touch_distance_px,
                "speed_threshold": config.min_speed_threshold,
                "debounce_frames": config.debounce_frames
            }
        }
        
        summary_json = "outputs/summary_notebook_demo.json"
        with open(summary_json, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"📊 Summary saved: {summary_json}")
        
    else:
        print("⚠️ No touches detected. Consider adjusting parameters:")
        print("   • Increase touch_distance_px (try 80-100)")
        print("   • Decrease min_speed_threshold (try 1.0)")
        print("   • Lower detection confidence")

if __name__ == "__main__":
    main()
