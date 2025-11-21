import cv2
import torch
import numpy as np
from collections import deque
import mediapipe as mp

from models.stgcn_transformer import STGCNTransformer

POSE_IDS = [11, 13, 15, 12, 14, 16]
HAND_L_IDS = list(range(21))
HAND_R_IDS = list(range(21))
MOUTH_INDICES = [
    78, 308, # corners of mouth
    178, 14, 402, # bottom inner lip
    81, 13, 311 # top inner lip
]

def normalize_coords(x, y, w, h):
    return x / float(w), y / float(h)

def frame_to_nodes(results, w, h):
    coords = []

    # Pose arms
    if results.pose_landmarks:
        for pid in POSE_IDS:
            lm = results.pose_landmarks.landmark[pid]
            x, y = normalize_coords(lm.x * w, lm.y * h, w, h)
            z = lm.z
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

    # Mouth subset
    if results.face_landmarks:
        for mid in MOUTH_INDICES:
            lm = results.face_landmarks.landmark[mid]
            x, y = normalize_coords(lm.x * w, lm.y * h, w, h)
            z = lm.z
            coords.extend([x, y, z])
    else:
        coords.extend([0.0, 0.0, 0.0] * len(MOUTH_INDICES))

    V = len(POSE_IDS) + len(HAND_L_IDS) + len(HAND_R_IDS) + len(MOUTH_INDICES)
    return np.array(coords, dtype=np.float32).reshape(V, 3)

def main():
    # Load model
    g2i = np.load('data/label_map.npz', allow_pickle=True)['gloss_to_id'].item()
    num_classes = len(g2i)
    model = STGCNTransformer(num_classes=num_classes)


    checkpoint_file = 'checkpoints/stgcn_transformer.pth' # update this to change the loaded file!

    model.load_state_dict(torch.load(checkpoint_file, map_location='cpu'))
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    cap = cv2.VideoCapture(0)
    mp_holistic = mp.solutions.holistic.Holistic(
        static_image_mode=False, model_complexity=1, smooth_landmarks=True, refine_face_landmarks=True
    )

    window = deque(maxlen=128)
    gloss_names = load_gloss_names()

    while True:
        ret, frame = cap.read()
        if not ret: break
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_holistic.process(rgb)
        nodes = frame_to_nodes(results, w, h)  # [V, 3]
        window.append(nodes)

        if len(window) == window.maxlen:
            seq = np.stack(list(window), axis=0)  # [T, V, C]
            x = torch.from_numpy(seq.astype(np.float32)).permute(2,0,1).unsqueeze(0)  # [1, C, T, V]
            x = x.to(device)
            with torch.no_grad():
                logits = model(x)
                pred = logits.argmax(dim=1).item()
                gloss = gloss_names.get(pred, str(pred))
            cv2.putText(frame, f'{gloss}', (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

        cv2.imshow('Realtime Sign Recognition', frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()
    mp_holistic.close()

def load_gloss_names():
    import numpy as np
    path = 'data/label_map.npz'
    m = np.load(path, allow_pickle=True)
    g2i = m['gloss_to_id'].item()
    # invert
    return {v:k for k,v in g2i.items()}

if __name__ == '__main__':
    main()
