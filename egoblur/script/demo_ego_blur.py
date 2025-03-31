# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import os
from functools import lru_cache
from typing import Annotated, List

import cv2
import numpy as np
import torch
import torchvision
from moviepy.editor import ImageSequenceClip
from moviepy.video.io.VideoFileClip import VideoFileClip
from typing import Union
from fastapi import FastAPI, File, UploadFile, responses

app = FastAPI()

lp_model_path = 'ego_blur_assets/ego_blur_lp.jit'
lp_model_score_threshold = 0.9
nms_iou_threshold = 0.3
scale_factor_detections = 1
# -----------------------------------------

face_model_path = None

def create_output_directory(file_path: str) -> None:
    """
    parameter file_path: absolute path to the directory where we want to create the output files
    Simple logic to create output directories if they don't exist.
    """
    print(
        f"Directory {os.path.dirname(file_path)} does not exist. Attempting to create it..."
    )
    os.makedirs(os.path.dirname(file_path))
    if not os.path.exists(os.path.dirname(file_path)):
        raise ValueError(
            f"Directory {os.path.dirname(file_path)} didn't exist. Attempt to create also failed. Please provide another path."
        )


@lru_cache
def get_device() -> str:
    """
    Return the device type
    """
    return (
        "cpu"
        if not torch.cuda.is_available()
        else f"cuda:{torch.cuda.current_device()}"
    )


def read_image(file: Annotated[UploadFile, File()]) -> np.ndarray:
    # Lê os bytes do arquivo
    file_bytes = file.file.read()
    
    # Converte os bytes em um array numpy
    nparr = np.frombuffer(file_bytes, np.uint8)
    
    # Decodifica a imagem usando OpenCV
    bgr_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if bgr_image is None:
        raise ValueError("Erro ao ler a imagem. Verifique o formato do arquivo.")
    
    return bgr_image

def write_image(image: np.ndarray, image_path: str) -> None:
    """
    parameter image: np.ndarray in BGR format
    parameter image_path: absolute path where we want to save the visualized image
    """
    cv2.imwrite(image_path, image)


def get_image_tensor(bgr_image: np.ndarray) -> torch.Tensor:
    """
    parameter bgr_image: image on which we want to make detections

    Return the image tensor
    """
    bgr_image_transposed = np.transpose(bgr_image, (2, 0, 1))
    image_tensor = torch.from_numpy(bgr_image_transposed).to(get_device())

    return image_tensor


def get_detections(
    detector: torch.jit._script.RecursiveScriptModule,
    image_tensor: torch.Tensor,
    model_score_threshold: float,
    nms_iou_threshold: float,
) -> List[List[float]]:
    """
    parameter detector: Torchscript module to perform detections
    parameter image_tensor: image tensor on which we want to make detections
    parameter model_score_threshold: model score threshold to filter out low confidence detection
    parameter nms_iou_threshold: NMS iou threshold to filter out low confidence overlapping boxes

    Returns the list of detections
    """
    with torch.no_grad():
        detections = detector(image_tensor)

    boxes, _, scores, _ = detections  # returns boxes, labels, scores, dims

    nms_keep_idx = torchvision.ops.nms(boxes, scores, nms_iou_threshold)
    boxes = boxes[nms_keep_idx]
    scores = scores[nms_keep_idx]

    boxes = boxes.cpu().numpy()
    scores = scores.cpu().numpy()

    score_keep_idx = np.where(scores > model_score_threshold)[0]
    boxes = boxes[score_keep_idx]
    return boxes.tolist()


def scale_box(
    box: List[List[float]], max_width: int, max_height: int, scale: float
) -> List[List[float]]:
    """
    parameter box: detection box in format (x1, y1, x2, y2)
    parameter scale: scaling factor

    Returns a scaled bbox as (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    w = x2 - x1
    h = y2 - y1

    xc = x1 + w / 2
    yc = y1 + h / 2

    w = scale * w
    h = scale * h

    x1 = max(xc - w / 2, 0)
    y1 = max(yc - h / 2, 0)

    x2 = min(xc + w / 2, max_width)
    y2 = min(yc + h / 2, max_height)

    return [x1, y1, x2, y2]


def visualize(
    image: np.ndarray,
    detections: List[List[float]],
    scale_factor_detections: float,
) -> np.ndarray:
    """
    parameter image: image on which we want to make detections
    parameter detections: list of bounding boxes in format [x1, y1, x2, y2]
    parameter scale_factor_detections: scale detections by the given factor to allow blurring more area, 1.15 would mean 15% scaling

    Visualize the input image with the detections and save the output image at the given path
    """
    image_fg = image.copy()
    mask_shape = (image.shape[0], image.shape[1], 1)
    mask = np.full(mask_shape, 0, dtype=np.uint8)

    for box in detections:
        if scale_factor_detections != 1.0:
            box = scale_box(
                box, image.shape[1], image.shape[0], scale_factor_detections
            )
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        w = x2 - x1
        h = y2 - y1

# Garante que o kernel seja ímpar e maior que zero
        ksize = (max(1, (image.shape[0] // 20) | 1), max(1, (image.shape[1] // 20) | 1))
        print('----------------------------')
        print("KSIZE: ")
        print(ksize)
        print('----------------------------')

        image_fg[y1:y2, x1:x2] = cv2.GaussianBlur(image_fg[y1:y2, x1:x2], ksize,3)
        cv2.ellipse(mask, (((x1 + x2) // 2, (y1 + y2) // 2), (w, h), 1), (255), -1)

    inverse_mask = cv2.bitwise_not(mask)
    image_bg = cv2.bitwise_and(image, image, mask=inverse_mask)
    image_fg = cv2.bitwise_and(image_fg, image_fg, mask=mask)
    image = cv2.add(image_bg, image_fg)

    return image


def visualize_image(
    image_file: str,
    face_detector: torch.jit._script.RecursiveScriptModule,
    lp_detector: torch.jit._script.RecursiveScriptModule,
    face_model_score_threshold: float,
    lp_model_score_threshold: float,
    nms_iou_threshold: float,
    output_image_path: str,
    scale_factor_detections: float,
):
    bgr_image = read_image(image_file)
    image = bgr_image.copy()

    image_tensor = get_image_tensor(bgr_image)
    image_tensor_copy = image_tensor.clone()
    detections = []
    # get face detections
    if face_detector is not None:
        detections.extend(
            get_detections(
                face_detector,
                image_tensor,
                face_model_score_threshold,
                nms_iou_threshold,
            )
        )

    # get license plate detections
    if lp_detector is not None:
        detections.extend(
            get_detections(
                lp_detector,
                image_tensor_copy,
                lp_model_score_threshold,
                nms_iou_threshold,
            )
        )
    image = visualize(
        image,
        detections,
        scale_factor_detections,
    ) 
    write_image(image, output_image_path)
    return image


def visualize_video(
    input_video_path: str,
    face_detector: torch.jit._script.RecursiveScriptModule,
    lp_detector: torch.jit._script.RecursiveScriptModule,
    face_model_score_threshold: float,
    lp_model_score_threshold: float,
    nms_iou_threshold: float,
    output_video_path: str,
    scale_factor_detections: float,
    output_video_fps: int,
):
    """
    parameter input_video_path: absolute path to the input video
    parameter face_detector: face detector model to perform face detections
    parameter lp_detector: face detector model to perform face detections
    parameter face_model_score_threshold: face model score threshold to filter out low confidence detection
    parameter lp_model_score_threshold: license plate model score threshold to filter out low confidence detection
    parameter nms_iou_threshold: NMS iou threshold
    parameter output_video_path: absolute path where the visualized video will be saved
    parameter scale_factor_detections: scale detections by the given factor to allow blurring more area
    parameter output_video_fps: fps of the visualized video

    Perform detections on the input video and save the output video at the given path.
    """
    visualized_images = []
    video_reader_clip = VideoFileClip(input_video_path)
    for frame in video_reader_clip.iter_frames():
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        image = frame.copy()
        bgr_image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        image_tensor = get_image_tensor(bgr_image)
        image_tensor_copy = image_tensor.clone()
        detections = []
        # get face detections
        if face_detector is not None:
            detections.extend(
                get_detections(
                    face_detector,
                    image_tensor,
                    face_model_score_threshold,
                    nms_iou_threshold,
                )
            )
        # get license plate detections
        if lp_detector is not None:
            detections.extend(
                get_detections(
                    lp_detector,
                    image_tensor_copy,
                    lp_model_score_threshold,
                    nms_iou_threshold,
                )
            )
        visualized_images.append(
            visualize(
                image,
                detections,
                scale_factor_detections,
            )
        )

    video_reader_clip.close()

    if visualized_images:
        video_writer_clip = ImageSequenceClip(visualized_images, fps=output_video_fps)
        video_writer_clip.write_videofile(output_video_path)
        video_writer_clip.close()

@app.post("/{user}/file/new")
async def create_file(user: str,file: Annotated[UploadFile, File()]):

    output_image_path = './saves/' + user + '/' + file.filename

    print('----------------------------')
    print("OUTPUT: " + output_image_path)
    print('----------------------------')

    face_detector = None

    if lp_model_path is not None:
        lp_detector = torch.jit.load(lp_model_path, map_location="cpu").to(
            get_device()
        )
        lp_detector.eval()
    else:
        lp_detector = None

    
    if output_image_path is not None and not os.path.exists(
        os.path.dirname(output_image_path)
    ):
        create_output_directory(output_image_path)

    if file is not None:
        visualize_image(
            file,
            face_detector,
            lp_detector,
            0,
            lp_model_score_threshold,
            nms_iou_threshold,
            output_image_path,
            scale_factor_detections,
        )

    # if args.input_video_path is not None:
    #     visualize_video(
    #         input_video_path,
    #         face_detector,
    #         lp_detector,
    #         face_model_score_threshold,
    #         lp_model_score_threshold,
    #         nms_iou_threshold,
    #         output_video_path,
    #         scale_factor_detections,
    #         output_video_fps,
    #     )
    return responses.FileResponse(output_image_path,200)