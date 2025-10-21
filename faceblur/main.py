import logging
from enum import Enum
from pathlib import Path
from typing import Annotated, Literal

import cv2
import typer
from rich.logging import RichHandler
from tqdm import tqdm
from ultralytics import YOLO


FORMAT = "%(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=FORMAT,
    datefmt="[%X]",
    handlers=[RichHandler(markup=True)],
)
app = typer.Typer()
LOGGER = logging.getLogger(__name__)


class FaceDetectionModel(str, Enum):
    n = "yolov12n-face.pt"
    s = "yolov12s-face.pt"
    m = "yolov12m-face.pt"
    l = "yolov12l-face.pt"  # noqa: E741


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
    model_size: Annotated[
        Literal["n", "s", "m", "l"],
        typer.Option("--model", "-m", help="Model size.", case_sensitive=False),
    ] = "n",
):
    """
    Args:
        input (Path): Path to input video.
    """

    # Open the input movie file and read props
    video = cv2.VideoCapture(str(input))
    if not video.isOpened():
        error = "Unable to open video file {input}."
        LOGGER.error(error)
        raise RuntimeError(error)

    nb_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)

    # Create an output file with same resolution & frame rate as input
    outfile = input.with_stem(input.stem + "_blured").with_suffix(".mp4")
    out = cv2.VideoWriter(
        str(outfile), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)  # type: ignore
    )

    # Create face detection model
    model_name = FaceDetectionModel[model_size]
    model = YOLO(Path() / "resources/models" / model_name.value)

    LOGGER.info(
        f"Processing [bold green]{input}[/] video file with {nb_frames} frames "
        f"to [bold red]{outfile}[/] using {model_name.value} model..."
    )

    frames = 0
    with tqdm(total=nb_frames, unit="frame") as pbar:
        while video.isOpened():
            ret, frame = video.read()
            frames += 1

            # Quit when the input video file ends
            if not ret:
                if frames < nb_frames:
                    pbar.update(nb_frames - frames)
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
    video.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    app()
