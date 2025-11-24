import cv2
import os
import glob
from tqdm import tqdm  # specific library for progress bars

def get_video_stats(video_folder):
    # Get all .mp4 files
    video_paths = glob.glob(os.path.join(video_folder, "*.mp4"))
    
    if not video_paths:
        print(f"No videos found in {video_folder}!")
        return

    print(f"Scanning {len(video_paths)} videos...")

    # Initialize variables to track min/max
    min_frames = float('inf')
    max_frames = 0
    min_seconds = float('inf')
    max_seconds = 0.0

    # Track which files are the shortest/longest
    shortest_file = ""
    longest_file = ""

    for path in tqdm(video_paths):
        cap = cv2.VideoCapture(path)
        
        if not cap.isOpened():
            print(f"Warning: Could not open {path}")
            continue

        # Get stats
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate duration
        if fps > 0:
            duration = frames / fps
        else:
            duration = 0

        # Update Minimums
        if frames < min_frames and frames > 0: 
            min_frames = frames
        if duration < min_seconds and duration > 0: 
            min_seconds = duration
            shortest_file = path

        # Update Maximums
        if frames > max_frames: 
            max_frames = frames
        if duration > max_seconds: 
            max_seconds = duration
            longest_file = path

        cap.release()

    print("\n" + "="*30)
    print("FINAL DATASET STATISTICS")
    print("="*30)
    print(f"Minimum Duration: {min_seconds:.4f} seconds")
    print(f"Maximum Duration: {max_seconds:.4f} seconds")
    print("-" * 30)
    print(f"Minimum Frames:   {min_frames}")
    print(f"Maximum Frames:   {max_frames}")
    print("="*30)
    print(f"Shortest video file: {os.path.basename(shortest_file)}")
    print(f"Longest video file:  {os.path.basename(longest_file)}")
    print(f"Count of Missing videos:  {count}")

if __name__ == "__main__":
    # Make sure this points to your actual video folder
    folder_path = "data/videos"
    get_video_stats(folder_path)