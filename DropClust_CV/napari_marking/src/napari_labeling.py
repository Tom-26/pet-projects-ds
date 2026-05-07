from pathlib import Path

import cv2
import napari
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter


# ============================================================
# Settings
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
VIDEO_PATH = PROJECT_DIR / "data/raw/input.avi"
ANNOTATIONS_DIR = PROJECT_DIR / "data/annotations"
PREVIEWS_DIR = PROJECT_DIR / "data/previews"

POINTS_LAYER_NAME = "manual_points"

POINT_SIZE = 5
RAW_OPACITY = 0.45
FILTERED_OPACITY = 1.0

SMALL_SIGMA = 1.0
BIG_SIGMA = 8.0

OUTPUT_POINTS_NPY = ANNOTATIONS_DIR / "manual_points.npy"
OUTPUT_POINTS_CSV = ANNOTATIONS_DIR / "manual_points.csv"
OUTPUT_PREVIEW_PNG = PREVIEWS_DIR / "manual_points_preview.png"


# ============================================================
# Video loading
# ============================================================


def load_video_grayscale(video_path: Path) -> np.ndarray:
    """
    Load a video file as a grayscale numpy array.

    Returns
    -------
    np.ndarray
        Array with shape: (frames, height, width)
    """
    if not video_path.exists():
        raise FileNotFoundError(
            f"Video file not found: {video_path}. "
            "Put the video in data/raw/input.avi "
            "or change VIDEO_PATH."
        )

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video file: {video_path}")

    frames = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)

    cap.release()

    if not frames:
        raise RuntimeError(f"No frames were read from video file: {video_path}")

    return np.stack(frames, axis=0)


# ============================================================
# Filtering
# ============================================================


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    """
    Normalize an array to uint8 range 0..255 for visualization.
    """
    image = image.astype(np.float32)
    image = image - np.min(image)

    max_value = np.max(image)

    if max_value <= 0:
        return np.zeros_like(image, dtype=np.uint8)

    image = image / max_value
    image = image * 255

    return image.astype(np.uint8)


def make_bandpass_video(
    frames: np.ndarray,
    small_sigma: float = SMALL_SIGMA,
    big_sigma: float = BIG_SIGMA,
) -> np.ndarray:
    """
    Create a filtered video that suppresses slow background and leaves bright spots.

    small blur removes pixel noise.
    big blur estimates slow-changing background.
    small_blur - big_blur highlights small bright objects.
    """
    frames_float = frames.astype(np.float32)

    small_blur = gaussian_filter(frames_float, sigma=(0, small_sigma, small_sigma))
    big_blur = gaussian_filter(frames_float, sigma=(0, big_sigma, big_sigma))

    bandpass = small_blur - big_blur

    return normalize_to_uint8(bandpass)
# ============================================================
# Saving
# ============================================================


def save_points(points: np.ndarray, output_dir: Path) -> None:
    """
    Save napari points to NPY and CSV.

    Napari stores points for video as:
        frame, y, x
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    npy_path = OUTPUT_POINTS_NPY
    csv_path = OUTPUT_POINTS_CSV

    np.save(npy_path, points)

    if points.size == 0:
        df = pd.DataFrame(columns=["frame", "y", "x"])
    else:
        df = pd.DataFrame(points, columns=["frame", "y", "x"])
        df["frame"] = df["frame"].round().astype(int)
        df["y"] = df["y"].round(2)
        df["x"] = df["x"].round(2)

    df.to_csv(csv_path, index=False)

    print(f"Saved NPY: {npy_path}")
    print(f"Saved CSV: {csv_path}")
    print(f"Points saved: {len(df)}")



def save_preview_image(
    frames: np.ndarray,
    points: np.ndarray,
    output_dir: Path,
    frame_id: int = 0,
) -> None:
    """
    Save a PNG preview with marked points for one selected frame.
    """
    frame_id = int(np.clip(frame_id, 0, frames.shape[0] - 1))
    image = cv2.cvtColor(frames[frame_id], cv2.COLOR_GRAY2BGR)

    if points.size > 0:
        for point in points:
            point_frame, y, x = point

            if int(round(point_frame)) != frame_id:
                continue

            cv2.circle(
                image,
                center=(int(round(x)), int(round(y))),
                radius=4,
                color=(0, 0, 255),
                thickness=1,
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    preview_path = OUTPUT_PREVIEW_PNG
    cv2.imwrite(str(preview_path), image)

    print(f"Saved preview: {preview_path}")


# ============================================================
# Main marking app
# ============================================================


def main() -> None:
    frames = load_video_grayscale(VIDEO_PATH)
    filtered = make_bandpass_video(frames)

    print(f"Video loaded: {VIDEO_PATH}")
    print(f"Frames shape: {frames.shape}")
    print("Napari point format: [frame, y, x]")
    print("Use layer 'manual_points' and point-add mode to mark real spots.")
    print("After closing napari, points will be saved automatically.")

    viewer = napari.Viewer()

    viewer.add_image(
        frames,
        name="raw_video",
        colormap="gray",
        opacity=RAW_OPACITY,
        blending="additive",
    )

    viewer.add_image(
        filtered,
        name="filtered_video",
        colormap="gray",
        opacity=FILTERED_OPACITY,
        blending="additive",
    )

    points_layer = viewer.add_points(
        name=POINTS_LAYER_NAME,
        size=POINT_SIZE,
        face_color="red",
        ndim=3,
    )

    points_layer.mode = "add"

    napari.run()

    points = viewer.layers[POINTS_LAYER_NAME].data

    save_points(points, ANNOTATIONS_DIR)

    if points.size > 0:
        first_marked_frame = int(round(points[0][0]))
    else:
        first_marked_frame = 0

    save_preview_image(
        frames=frames,
        points=points,
        output_dir=PREVIEWS_DIR,
        frame_id=first_marked_frame,
    )


if __name__ == "__main__":
    main()
