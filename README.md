# Sign_Language_Translation

This project aims to **translate dynamic sign language** using a video classification pipeline.  

### Static vs Dynamic Sign Language
- **Static sign language**: hand gestures that remain in one place (like a single image). Often used for concepts such as fingerspelling.  
- **Dynamic sign language**: involves movement of hands, arms, and other body parts to convey meaning.  
- **Key difference**: static gestures = single pose; dynamic gestures = sequence and motion of poses.

---

## 🔑 Project Goals
The process is broken down into three main steps:

1. **Landmark Extraction**  
   - Extract landmarks of the person’s hands, arms, and mouth from video frames using MediaPipe.

2. **Gloss Prediction**  
   - Train a model that predicts which gloss (word/expression) a sequence of frames corresponds to.

3. **Sentence Translation**  (PENDING)
   - Train another model (LLM-based) that takes a sequence of glosses and predicts the sentence in natural language.

---

## 📂 Core Model File: `stgcn_transformer.py`
This file defines the **ST-GCN + Transformer architecture**:

1. **ST-GCN**: extracts joint-based spatial-temporal features.  
2. **Spatial pooling + projection**: converts features into a temporal sequence.  
3. **Transformer**: captures temporal dependencies across frames.  
4. **Temporal pooling**: aggregates per-frame features into a video-level representation.  
5. **Classifier**: predicts gloss class labels for the sequence.

---

## ▶️ How to Run the Pipeline

### 1. Preprocessing (Landmark Extraction)
Before training, you must run the preprocessing script to convert raw videos into landmark sequences:

```bash
python preprocess/extract_landmarks.py
```

This will:
- Parse `WLASL_v0.3.json` for gloss and video metadata.
- Skip missing videos listed in `missing.txt`.
- Extract pose, hand, and mouth landmarks using MediaPipe.
- Save sequences as `.npz` files in `data/sequences/`.
- Save a `label_map.npz` file with gloss → class ID mapping.

---

### 2. Training
Run the training script:

```bash
python train.py
```

This will:
- Load preprocessed sequences.  
- Set up the ST-GCN + Transformer model.  
- Define loss, optimizer, and scheduler.  
- Train over multiple epochs with GPU support.  
- Perform validation after each epoch.  
- Log training/validation loss and accuracy.  
- Save model weights to `checkpoints/stgcn_transformer.pth`.

Helper:  
- `load_num_classes()` loads the label mapping from preprocessing and returns the number of gloss classes.

---

### 3. Evaluation
Run the evaluation script:

```bash
python evaluate.py
```

This will:
- Load the dataset (validation/test split).  
- Load the trained model weights (update file path if needed).  
- Evaluate accuracy on the test set.  

---

### 4. Real-Time Inference
Run the real-time demo:

```bash
python realtime_infer.py
```

This will:
- Open a webcam window.  
- Continuously extract landmarks from frames.  
- Display the predicted gloss in the top-left corner.  

⚠️ Current limitation:  
- Uses a fixed **128-frame window**.  
- No prediction until 128 frames are accumulated.  
- Signs shorter than 128 frames may get mixed with subsequent frames.  

Example:  
- **Sign A** (50 frames): no prediction until 128 frames are reached.  
- **Sign B** immediately after: prediction may mix frames from A and B.  
- Separate predictions require each sign to occupy a full 128-frame window.

---

## ✅ Summary of Commands
1. **Preprocess videos** → `python preprocess/extract_landmarks.py`  
2. **Train model** → `python train.py`  
3. **Evaluate model** → `python evaluate.py`  
4. **Run real-time demo** → `python realtime_infer.py`  
