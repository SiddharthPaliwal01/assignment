from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import argparse
import cv2
import numpy as np
import json
from ultralytics import YOLO


@dataclass
class Config:
    video_path: str
    output_path: str = "outputs/annotated.mp4"
    events_csv: str = "outputs/events.csv"
    model_det: str = "yolov8s.pt"
    model_pose: str = "yolov8n-pose.pt"
    conf_det: float = 0.15
    iou_assign: float = 0.3
    max_missing_frames: int = 15
    touch_dist_px: int = 60
    debounce_frames: int = 8
    flow_win_size: int = 21
    imgsz_det: int = 960
    imgsz_pose: int = 960


class SimpleTracker:
    def __init__(self):
        self.ball_bbox: Optional[Tuple[int, int, int, int]] = None
        self.player_bbox: Optional[Tuple[int, int, int, int]] = None
        self.miss_ball = 0
        self.miss_player = 0
        self.history_player_centers: List[Tuple[int, int]] = []

    @staticmethod
    def iou(a, b) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
        inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
        inter = iw * ih
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a + area_b - inter + 1e-6
        return inter / union

    @staticmethod
    def xywh2xyxy(b):
        x, y, w, h = b
        return int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)

    @staticmethod
    def center(b):
        x1, y1, x2, y2 = b
        return (x1 + x2) // 2, (y1 + y2) // 2

    def update(
        self,
        dets_ball: List[Tuple[int, int, int, int]],
        dets_player: List[Tuple[int, int, int, int]],
        cfg: Config,
    ):
        if dets_ball:
            if self.ball_bbox is None:
                self.ball_bbox = dets_ball[0]
            else:
                bb = max(dets_ball, key=lambda b: self.iou(self.ball_bbox, b))
                if self.iou(self.ball_bbox, bb) > cfg.iou_assign:
                    self.ball_bbox = bb
            self.miss_ball = 0
        else:
            self.miss_ball += 1
            if self.miss_ball > cfg.max_missing_frames:
                self.ball_bbox = None
        if dets_player:
            if self.player_bbox is None:
                self.player_bbox = dets_player[0]
            else:
                pb = max(dets_player, key=lambda b: self.iou(self.player_bbox, b))
                if self.iou(self.player_bbox, pb) > cfg.iou_assign:
                    self.player_bbox = pb
            self.miss_player = 0
        else:
            self.miss_player += 1
            if self.miss_player > cfg.max_missing_frames:
                self.player_bbox = None
        if self.player_bbox:
            self.history_player_centers.append(self.center(self.player_bbox))
            if len(self.history_player_centers) > 1200:
                self.history_player_centers = self.history_player_centers[-1200:]


class TouchCounter:
    def __init__(self, cfg: Config):
        self.right = 0
        self.left = 0
        self.cooldown = 0
        self.cfg = cfg
        self.last_ball_center: Optional[Tuple[int, int]] = None
        self.last_ball_speed = 0.0
        self.events: List[Tuple[int, str, float]] = []  # frame_idx, side, player_speed

    @staticmethod
    def l2(a, b):
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    def update(self, ball_c: Optional[Tuple[int, int]], ankles, frame_idx: int, player_speed_px_s: float):
        event = None
        if self.cooldown > 0:
            self.cooldown -= 1
        if ball_c is not None:
            speed = 0.0
            if self.last_ball_center is not None:
                speed = self.l2(ball_c, self.last_ball_center)
            self.last_ball_center = ball_c
            self.last_ball_speed = speed
            if self.cooldown == 0 and ankles:
                rl = ankles.get("right_ankle")
                ll = ankles.get("left_ankle")
                near_r = self.l2(ball_c, rl) if rl else 1e9
                near_l = self.l2(ball_c, ll) if ll else 1e9
                near = min(near_r, near_l)
                # relax speed gate slightly
                if near < self.cfg.touch_dist_px and speed > 1.5:
                    if near_r < near_l:
                        self.right += 1
                        side = "right"
                    else:
                        self.left += 1
                        side = "left"
                    event = (side, ball_c)
                    self.events.append((frame_idx, side, player_speed_px_s))
                    self.cooldown = self.cfg.debounce_frames
        return event


def estimate_ball_spin(prev_gray: np.ndarray, gray: np.ndarray, ball_bbox: Tuple[int, int, int, int], cfg: Config) -> str:
    x1, y1, x2, y2 = ball_bbox
    x1, y1 = max(0, x1), max(0, y1)
    roi_prev = prev_gray[y1:y2, x1:x2]
    roi_curr = gray[y1:y2, x1:x2]
    if roi_prev.size == 0 or roi_curr.size == 0:
        return "unknown"
    flow = cv2.calcOpticalFlowFarneback(roi_prev, roi_curr, None, 0.5, 3, cfg.flow_win_size, 3, 5, 1.2, 0)
    vx, vy = flow[..., 0], flow[..., 1]
    avg_vy = float(np.nanmean(vy))
    if abs(avg_vy) < 0.05:
        return "unknown"
    return "forward" if avg_vy < 0 else "backward"


def ankles_from_pose(pose_xy: Optional[np.ndarray]):
    res = {}
    if pose_xy is None or pose_xy.shape[0] < 17:
        return res
    def to_t(p):
        return (int(p[0]), int(p[1]))
    # NOTE: Verify indices based on model; these are typical COCO ankle indices
    res["left_ankle"] = to_t(pose_xy[15])
    res["right_ankle"] = to_t(pose_xy[16])
    return res


def draw_overlays(frame, tracker: SimpleTracker, touches: TouchCounter, spin_label: str, fps: float, touch_event, ankles=None, ball_c=None):
    h, w = frame.shape[:2]
    if tracker.ball_bbox:
        cv2.rectangle(frame, tracker.ball_bbox[:2], tracker.ball_bbox[2:], (0, 255, 255), 2)
    if tracker.player_bbox:
        cv2.rectangle(frame, tracker.player_bbox[:2], tracker.player_bbox[2:], (0, 255, 0), 2)
    cv2.rectangle(frame, (10, 10), (460, 160), (0, 0, 0), -1)
    cv2.putText(frame, f"Right touches: {touches.right}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Left touches : {touches.left}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Ball rotation: {spin_label}", (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    if len(tracker.history_player_centers) > 2:
        v = np.linalg.norm(np.diff(np.array(tracker.history_player_centers[-3:], dtype=float), axis=0), axis=1).mean() * fps
        cv2.putText(frame, f"Player speed: {v:.1f} px/s", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    if touch_event:
        _, c = touch_event
        cv2.circle(frame, c, 10, (0, 0, 255), 2)
    # draw ankle and ball markers for debugging
    if ankles:
        if ankles.get("left_ankle"):
            cv2.circle(frame, ankles["left_ankle"], 5, (255, 0, 0), -1)
            cv2.putText(frame, "L-ankle", (ankles["left_ankle"][0]+6, ankles["left_ankle"][1]-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
        if ankles.get("right_ankle"):
            cv2.circle(frame, ankles["right_ankle"], 5, (0, 0, 255), -1)
            cv2.putText(frame, "R-ankle", (ankles["right_ankle"][0]+6, ankles["right_ankle"][1]-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    if ball_c:
        cv2.circle(frame, ball_c, 5, (0, 255, 255), -1)


def run(cfg: Config):
    cap = cv2.VideoCapture(cfg.video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {cfg.video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    Path(Path(cfg.output_path).parent).mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(cfg.output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    det = YOLO(cfg.model_det)
    pose = YOLO(cfg.model_pose)
    tracker = SimpleTracker()
    touches = TouchCounter(cfg)

    ok, prev = cap.read()
    if not ok:
        raise RuntimeError("Empty video")
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detection on full frame
        res_det = det.predict(frame, conf=cfg.conf_det, imgsz=cfg.imgsz_det, verbose=False)[0]
        dets_ball, dets_player = [], []
        for b, c in zip(res_det.boxes.xywh.cpu().numpy(), res_det.boxes.cls.cpu().numpy().astype(int)):
            name = res_det.names[c]
            bb = SimpleTracker.xywh2xyxy(b)
            if name in ("sports ball", "ball"):
                dets_ball.append(bb)
            if name == "person":
                dets_player.append(bb)
        tracker.update(dets_ball, dets_player, cfg)

        # Pose on full frame; pick keypoints overlapping player bbox
        ankles = {}
        res_pose = pose.predict(frame, conf=0.10, imgsz=cfg.imgsz_pose, verbose=False)[0]
        dets_ball, dets_player = [], []
        for b, c in zip(res_det.boxes.xywh.cpu().numpy(), res_det.boxes.cls.cpu().numpy().astype(int)):
            name = res_det.names[c]
            bb = SimpleTracker.xywh2xyxy(b)
            if name in ("sports ball", "ball"):
                dets_ball.append(bb)
            if name == "person":
                dets_player.append(bb)
        tracker.update(dets_ball, dets_player, cfg)

        # Pose on full frame; pick keypoints overlapping player bbox
        ankles = {}
        res_pose = pose.predict(frame, conf=0.10, imgsz=cfg.imgsz_pose, verbose=False)[0]
        if len(res_pose.keypoints) > 0:
            # pick the instance with max IoU to player bbox (if available), else first
            k_all = res_pose.keypoints.xy.cpu().numpy()  # [N,17,2]
            choose_idx = 0
            if tracker.player_bbox is not None:
                px1,py1,px2,py2 = tracker.player_bbox
                pbb = (px1,py1,px2,py2)
                def iou(bb1, bb2):
                    ax1,ay1,ax2,ay2=bb1; bx1,by1,bx2,by2=bb2
                    ix1,iy1=max(ax1,bx1),max(ay1,by1)
                    ix2,iy2=min(ax2,bx2),min(ay2,by2)
                    iw,ih=max(0,ix2-ix1),max(0,iy2-iy1)
                    inter=iw*ih; ua=(ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter+1e-6
                    return inter/ua
                # derive bbox from keypoints
                best, bi = 0.0, 0
                for i, pts in enumerate(k_all):
                    x1,y1 = np.min(pts, axis=0)
                    x2,y2 = np.max(pts, axis=0)
                    iouv = iou((int(x1),int(y1),int(x2),int(y2)), pbb)
                    if iouv > best:
                        best, bi = iouv, i
                choose_idx = bi
            kpts = k_all[choose_idx]
            ankles = ankles_from_pose(kpts)

        # Touch
        ball_c = None
        if tracker.ball_bbox:
            bx1, by1, bx2, by2 = tracker.ball_bbox
            ball_c = ((bx1 + bx2) // 2, (by1 + by2) // 2)

        # Player speed (px/s)
        player_speed = 0.0
        if len(tracker.history_player_centers) > 2:
            v = np.linalg.norm(np.diff(np.array(tracker.history_player_centers[-3:], dtype=float), axis=0), axis=1).mean() * fps
            player_speed = float(v)

        touch_event = touches.update(ball_c, ankles, frame_idx, player_speed)

        # Spin
        spin_label = "unknown"
        if tracker.ball_bbox:
            spin_label = estimate_ball_spin(prev_gray, gray, tracker.ball_bbox, cfg)

        draw_overlays(frame, tracker, touches, spin_label, fps=fps, touch_event=touch_event, ankles=ankles, ball_c=ball_c)
        writer.write(frame)
        prev_gray = gray
        frame_idx += 1

    writer.release()
    cap.release()
    # Save events/summary
    Path("outputs").mkdir(exist_ok=True)
    import csv
    with open(cfg.events_csv, "w", newline="") as f:
        wri = csv.writer(f)
        wri.writerow(["frame", "side", "player_speed_px_s"])
        for fr, side, sp in touches.events:
            wri.writerow([fr, side, f"{sp:.3f}"])
    Path("outputs/summary.json").write_text(json.dumps({"right": touches.right, "left": touches.left}, indent=2))
    print(f"Saved video: {cfg.output_path}\nSaved events: {cfg.events_csv}")


def parse_args() -> Config:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="outputs/annotated.mp4")
    ap.add_argument("--det", default="yolov8s.pt")
    ap.add_argument("--pose", default="yolov8n-pose.pt")
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--touch_dist", type=int, default=60)
    def parse_args() -> Config:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="outputs/annotated.mp4")
    ap.add_argument("--det", default="yolov8s.pt")
    ap.add_argument("--pose", default="yolov8n-pose.pt")
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--touch_dist", type=int, default=60)
    ap.add_argument("--imgsz_det", type=int, default=960)
    ap.add_argument("--imgsz_pose", type=int, default=960)
    args = ap.parse_args()
    return Config(
        video_path=args.video,
        output_path=args.out,
        model_det=args.det,
        model_pose=args.pose,
        conf_det=args.conf,
        touch_dist_px=args.touch_dist,
        imgsz_det=args.imgsz_det,
        imgsz_pose=args.imgsz_pose,
    )


if __name__ == "__main__":
    cfg = parse_args()
    run(cfg)
