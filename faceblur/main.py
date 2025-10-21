import logging
from pathlib import Path
from typing import Annotated

import cv2
import typer
from tqdm import tqdm
from ultralytics import YOLO

app = typer.Typer()

LOGGER = logging.getLogger(__name__)


@app.command()
def faces(
    input: Annotated[
        Path,
        typer.Argument(
            help="Path to input video.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
):
    """
    Args:
        input (Path): Path to input video.
    """

    # Open the input movie file
    input_movie = cv2.VideoCapture(str(input))
    nb_frames = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(input_movie.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(input_movie.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = input_movie.get(cv2.CAP_PROP_FPS)

    # Create an output movie file (make sure resolution/frame rate matches input video!)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
    # fourcc = cv2.VideoWriter_fourcc(*"XVID")  # type: ignore

    outfile = input.with_stem(input.stem + "_blured").with_suffix(".mp4")

    out = cv2.VideoWriter(str(outfile), fourcc, fps, (width, height))

    model = YOLO("yolov12n-face.pt")

    frames = 0
    with tqdm(
        total=nb_frames,
        desc=f"Processing {input} with {nb_frames} frames to {outfile}...",
    ) as pbar:
        while input_movie.isOpened():
            ret, frame = input_movie.read()
            frames += 1

            # Quit when the input video file ends
            if not ret:
                break

            results = model.predict(frame, show=False, verbose=False)

            # Process each detection
            for r in results:
                for box in r.boxes:
                    y1, x1, y2, x2 = box.xyxy[0].detach().to(int).tolist()

                    # Extract the region of the image that contains the face
                    face_image = frame[x1:x2, y1:y2]

                    # Blur the face image
                    face_image = cv2.GaussianBlur(face_image, (99, 99), 30)

                    # Put the blurred face region back into the frame image
                    frame[x1:x2, y1:y2] = face_image

            # write the blurred frame
            out.write(frame)
            pbar.update(1)

    # All done!
    input_movie.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    app()
