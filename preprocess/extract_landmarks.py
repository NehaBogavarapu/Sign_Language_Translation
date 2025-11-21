import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warning, 3=error (used to supress lowlevel warnings)
import json
import cv2
import numpy as np
from tqdm import tqdm
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

POSE_IDS = [11, 13, 15, 12, 14, 16]  # L shoulder, elbow, wrist; R shoulder, elbow, wrist
HAND_L_IDS = list(range(21))
HAND_R_IDS = list(range(21))
MOUTH_INDICES = [
    78, 308, # corners of mouth
    178, 14, 402, # bottom inner lip
    81, 13, 311 # top inner lip
]

def read_missing(missing_path):
    if not os.path.exists(missing_path):
        return set()
    with open(missing_path, 'r') as f:
        return set([line.strip() for line in f if line.strip()])

def build_instance_index(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    vid_to_label = {}
    vid_to_bbox = {}
    vid_to_split = {}
    gloss_to_id = {}
    gloss_id = 0
    for entry in data:
        gloss = entry['gloss']
        if gloss not in gloss_to_id:
            gloss_to_id[gloss] = gloss_id
            gloss_id += 1
        label_id = gloss_to_id[gloss]
        for inst in entry['instances']:
            vid = str(inst['video_id']).zfill(5) if isinstance(inst['video_id'], int) else str(inst['video_id'])
            vid_to_label[vid] = label_id
            vid_to_split[vid] = inst.get('split', 'train')
            bbox = inst.get('bbox', None)
            vid_to_bbox[vid] = bbox
    return vid_to_label, vid_to_bbox, vid_to_split, gloss_to_id

def crop_frame(frame, bbox):
    if not bbox or len(bbox) != 4:
        return frame
    x1, y1, x2, y2 = bbox
    x1 = max(0, int(x1)); y1 = max(0, int(y1))
    x2 = min(frame.shape[1], int(x2)); y2 = min(frame.shape[0], int(y2))
    if x2 <= x1 or y2 <= y1:
        return frame
    return frame[y1:y2, x1:x2]

def normalize_coords(x, y, w, h):
    return x / float(w), y / float(h)

def extract_sequence(video_path, bbox=None, max_frames=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    seq = []
    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=True
    ) as holistic:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            # Only process every N-th frame
            N = 5  # every 5th frame
            if frame_count % N != 0:
                continue

            if max_frames and frame_count > max_frames:
                break

            if bbox:
                frame = crop_frame(frame, bbox)

            h, w = frame.shape[:2]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)

            coords = []

            # Pose arms
            if results.pose_landmarks:
                for pid in POSE_IDS:
                    lm = results.pose_landmarks.landmark[pid]
                    x, y = normalize_coords(lm.x * w, lm.y * h, w, h)
                    z = lm.z  # relative depth
                    coords.extend([x, y, z])
            else:
                coords.extend([0.0, 0.0, 0.0] * len(POSE_IDS))

            # Left hand
            if results.left_hand_landmarks:
                for hid in HAND_L_IDS:
                    lm = results.left_hand_landmarks.landmark[hid]
                    x, y = normalize_coords(lm.x * w, lm.y * h, w, h)
                    z = lm.z
                    coords.extend([x, y, z])
            else:
                coords.extend([0.0, 0.0, 0.0] * len(HAND_L_IDS))

            # Right hand
            if results.right_hand_landmarks:
                for hid in HAND_R_IDS:
                    lm = results.right_hand_landmarks.landmark[hid]
                    x, y = normalize_coords(lm.x * w, lm.y * h, w, h)
                    z = lm.z
                    coords.extend([x, y, z])
            else:
                coords.extend([0.0, 0.0, 0.0] * len(HAND_R_IDS))

            # Mouth subset from face mesh
            if results.face_landmarks:
                face = results.face_landmarks.landmark
                for mid in MOUTH_INDICES:
                    lm = face[mid]
                    x, y = normalize_coords(lm.x * w, lm.y * h, w, h)
                    z = lm.z
                    coords.extend([x, y, z])
            else:
                coords.extend([0.0, 0.0, 0.0] * len(MOUTH_INDICES))

            seq.append(coords)

    cap.release()
    if not seq:
        return None

    seq = np.array(seq, dtype=np.float32)  # [T, V*C]
    V = len(POSE_IDS) + len(HAND_L_IDS) + len(HAND_R_IDS) + len(MOUTH_INDICES)
    C = 3
    seq = seq.reshape(len(seq), V, C)      # [T, V, C]
    return seq

def main():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, 'data')
    videos_dir = os.path.join(data_dir, 'videos')
    out_dir = os.path.join(data_dir, 'sequences')
    os.makedirs(out_dir, exist_ok=True)

    missing = read_missing(os.path.join(data_dir, 'missing.txt'))
    vid_to_label, vid_to_bbox, vid_to_split, gloss_to_id = build_instance_index(
        os.path.join(data_dir, 'WLASL_v0.3.json')
    )

    vid_items = vid_to_label.items()
    processed_count = 0

    # LIMIT FOR TESTING WHETHER PREPROCESSING IS WORKING
    max_videos = 10

    for vid, label in tqdm(vid_items, desc='Extracting landmarks'):
        if processed_count >= max_videos:
            break
        if vid in missing:
            continue
        video_path = os.path.join(videos_dir, f'{vid}.mp4')
        if not os.path.exists(video_path):
            continue
        seq = extract_sequence(video_path, bbox=vid_to_bbox.get(vid), max_frames=None)
        if seq is None:
            continue
        split = vid_to_split.get(vid, 'train')
        save_path = os.path.join(out_dir, f'{vid}.npz')
        np.savez_compressed(save_path, seq=seq, label=label, split=split)

        processed_count += 1

    # Save label mapping for reuse
    np.savez_compressed(os.path.join(data_dir, 'label_map.npz'),
                        gloss_to_id=gloss_to_id)

if __name__ == '__main__':
    main()
