# Yolo Face Blur - Automatically detect and blur faces in a video

Simple Python script to provide a CLI to automatically detect and blur all visible faces in a video input file.

<div align="center">
    <video src="https://github.com/user-attachments/assets/a98515b1-2576-4d86-a2be-bcdc15a5720b" width="640" controls></video>
</div>

## Install

This project requires a recent version of python: ![python_version](https://img.shields.io/badge/Python-%3E=3.12-blue).

### Install from github

Clone the repository and install the project in your python environment, either using `pip`

```bash
git clone https://github.com/jonasrenault/yolo-face-blur.git
cd yolo-face-blur
pip install --editable .
```

or [uv](https://docs.astral.sh/uv/)

```bash
git clone https://github.com/jonasrenault/yolo-face-blur.git
cd yolo-face-blur
uv sync
```

## Usage

### CLI

When you install the project in a virtual environment, it creates a CLI script called `blur`. To blur the faces in the video, simply call the `blur` script with the path of the video.

```bash
blur resources/videos/rugby.mp4
```

You can select the model size and confidence threshold as options for the CLI command:

```bash
blur resources/videos/rugby.mp4 -m l -c 0.7
```

## Models

The models used for face detection are the [Yolov12](https://docs.ultralytics.com/) models from [Yolo-Face](https://github.com/YapaLab/yolo-face). Available sizes for the face detection model are `nano (n)`, `small (s)`, `medium (s)` and `large (l)`.
