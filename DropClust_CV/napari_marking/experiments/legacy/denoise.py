import cv2
import numpy as np
from tqdm import tqdm

video_path = "input.avi"

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError("Cannot open video")

fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frames.append(gray)
cap.release()

tw = 13
half = tw // 2

if len(frames) < tw:
    raise ValueError("Need at least " + str(tw) + " frames")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out_smoothed = cv2.VideoWriter("smoothed.mp4", fourcc, fps, (w, h), isColor=False)
out_thresh   = cv2.VideoWriter("thresholded.mp4", fourcc, fps, (w, h), isColor=False)
out_cleaned  = cv2.VideoWriter("cleaned.mp4", fourcc, fps, (w, h), isColor=False)

kernel = np.ones((3, 3), np.uint8)
counts = []

for i in tqdm(range(half, len(frames) - half), desc="Processing", unit="frame"):
    stack = frames[i-half:i+half+1]

    smoothed = cv2.fastNlMeansDenoisingMulti(
        srcImgs=stack,
        imgToDenoiseIndex=half,
        temporalWindowSize=tw,
        h=12,
        searchWindowSize=27
    )

    THRESH_VALUE = 127
    _, thresh = cv2.threshold(smoothed, THRESH_VALUE, 255, cv2.THRESH_BINARY)

    # Clean tiny noise
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Count spots
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    min_area = 3
    spot_count = sum(
        1 for lbl in range(1, num_labels)
        if stats[lbl, cv2.CC_STAT_AREA] >= min_area
    )
    counts.append((i, spot_count))

    out_smoothed.write(smoothed)
    out_thresh.write(thresh)
    out_cleaned.write(cleaned)

print("Counts:")
for i, count in counts:
    print(f"{i:5d}: {count:4d}")

out_smoothed.release()
out_thresh.release()
out_cleaned.release()

print("Saved smoothed.mp4, thresholded.mp4, cleaned.mp4")