# Third-Party Component Notices

## Project's Own License

- The project's own code is released under the **MIT License**; the full text is in the `LICENSE` file in the repository root;
- Third-party components keep their own licenses (see below) and are not changed by the project's MIT license.

## Bringing Old Photos Back to Life (restoration/detection/face enhancement models)

- Source: https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life
  (CVPR 2020 oral, Microsoft Corporation)
- License: MIT License (full text in `LICENSE-Bringing-Old-Photos-Back-to-Life`)
- Usage scope: the `Global/`, `Face_Detection/`, `Face_Enhancement/` directories of this project

## DDColor (old photo colorization module)

- Source: https://github.com/piddnad/DDColor (ICCV 2023, DAMO Academy / Alibaba Group)
- License: Apache License 2.0 (full text in `ddcolor/LICENSE`)
- Usage scope: the `ddcolor/` and `basicsr/archs/ddcolor_arch_utils/` directories of this project
- Weight source: ModelScope `damo/cv_ddcolor_image-colorization/pytorch_model.pt`
  (or HuggingFace `piddnad/ddcolor_modelscope`), officially released by the Apache-2.0 project

## Pre-trained model weights (not distributed in this repository)

This repository contains **code only**; every weight file is downloaded from its
official source by `python -m scripts.download_weights download` and verified
against the SHA-256 manifest in `config/weights_manifest.json`.

| Weight | Source | License / terms |
| --- | --- | --- |
| `Global/checkpoints/**`, `Face_Enhancement/checkpoints/**` | Microsoft `facevc.blob.core.windows.net/zhanbo/old_photo/pretrain/` (official checkpoints of Bringing Old Photos Back to Life) | Released with the MIT-licensed upstream project |
| `weights/ddcolor/pytorch_model.pt` | ModelScope / HuggingFace (DDColor) | Apache License 2.0 |
| `weights/yunet/face_detection_yunet_2023mar.onnx` | OpenCV Zoo (YuNet face detector) | Apache License 2.0 |
| `Face_Detection/shape_predictor_68_face_landmarks.dat` | dlib.net (68-point facial landmark model) | dlib itself is Boost Software License, but **the pre-trained landmark model is restricted to research/non-commercial use** (trained on the iBUG 300-W dataset). Replace it before commercial deployment |

## Sample and documentation images

Every demonstration image is either an upstream asset or a derivative of one:

| Path | Origin |
| --- | --- |
| `test_images/old/*` | Evaluation photos published by Bringing Old Photos Back to Life (byte-identical copies) |
| `test_images/old_w_scratch/*` | Same upstream repository, scratch samples (byte-identical copies) |
| `examples/old/*` | Copies/resizes of the upstream evaluation photos, used by the web UI example galleries |
| `examples/old_w_scratch/*` | Byte-identical copies of the upstream scratch samples |
| `examples/color/*` | Grayscale conversions of the upstream evaluation photos, used as colorization demo inputs |
| `docs/upstream/*` | README/pipeline illustrations of the upstream projects listed above (a few are re-encoded to reduce size) |
| `docs/examples/*` | Inputs taken from the rows above; output images produced by this project's pipeline on this machine |

These images are included for documentation and demonstration purposes only and are
not covered by this project's MIT license. If you redistribute this project or use it
commercially, replace them with assets you are allowed to publish.
