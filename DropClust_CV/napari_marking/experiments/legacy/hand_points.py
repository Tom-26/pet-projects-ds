import marimo

__generated_with = "0.22.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import napari
    import cv2

    return cv2, napari


@app.cell
def _(napari):
    print(napari.__file__)
    print(napari.__version__)
    return


@app.cell
def _(cv2, napari):

    cap = cv2.VideoCapture("input.avi")

    frames = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frames.append(gray)

    viewer = napari.Viewer()

    viewer.add_image(frames)

    viewer.add_points()


    napari.run()

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
