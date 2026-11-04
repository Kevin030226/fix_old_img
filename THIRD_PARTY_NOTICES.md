# Third-Party Component Notices

## Project's Own License

- The project's own code is released under the **MIT License**; the full text is in the `LICENSE` file in the repository root;
- Third-party components keep their own licenses (see below) and are not changed by the project's MIT license.

## Bringing Old Photos Back to Life (code vendored in this repository)

- Source: https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life
  (CVPR 2020 oral, Microsoft Corporation)
- License: MIT License (full text in `LICENSE-Bringing-Old-Photos-Back-to-Life`); this covers the
  **code** vendored under `Global/`, `Face_Detection/` and `Face_Enhancement/`.
- The **pre-trained weights** published by the same project are a separate question; see
  "Pre-trained model weights" below (the README's MIT sentence and its "academic research
  use only" sentence conflict and remain unresolved).

## DDColor (code vendored in this repository)

- Source: https://github.com/piddnad/DDColor (ICCV 2023, DAMO Academy / Alibaba Group)
- License: Apache License 2.0 (full text in `ddcolor/LICENSE`); this covers the **code**
  vendored under `ddcolor/` and `basicsr/archs/ddcolor_arch_utils/`.
- The **weights** are published separately on ModelScope; see "Pre-trained model weights" below.

## Pre-trained model weights (not distributed in this repository)

This repository contains **code only**. No weight file is committed, packaged into a wheel,
or baked into the container images. Weights are fetched by
`python -m fiximg.cli.download_weights download`; the legacy
`scripts/download_weights.sh` now delegates to that CLI and no longer downloads anything by
itself. The committed `config/weights_manifest.json` is the trust anchor: every declared
path, byte size and SHA-256 must match, any mismatch exits non-zero with the per-entry
detail, and a normal download never rewrites the manifest. `generate` is a maintainer-only
command for deliberately rebuilding that baseline.

A code license does not license the model weights by itself. The table below records what
each rights-holder actually says about the **weights** and marks unresolved questions
instead of assuming a permission. **Nothing here states that commercial deployment or
redistribution is authorised.**

| Weight | Official source | Weight-level licensing evidence | Assessment |
| --- | --- | --- | --- |
| `Global/checkpoints/**`, `Face_Enhancement/checkpoints/**` | Microsoft, GitHub release `v1.0` assets `global_checkpoints.zip` / `face_checkpoints.zip` (the README's documented download). The older `facevc.blob.core.windows.net` Azure URLs no longer resolve. | Upstream README "License": *"The codes and the pretrained model in this repository are under the MIT license as specified by the LICENSE file."* The same README "Maintenance" section also says the project *"is for academic research use only"*. The LICENSE text itself only says "Software"; there is no model card and no official checksum. | Official source, **conflicting statements**: MIT vs academic-only is unresolved. Obtain written clarification from Microsoft before commercial use or redistribution. |
| `weights/yunet/face_detection_yunet_2023mar.onnx` | OpenCV Zoo `models/face_detection_yunet` | The **model directory** carries its own MIT LICENSE and its README says *"All files in this directory are licensed under [MIT License]"*; the repository-level Apache-2.0 covers the surrounding code, not the model as such. Training data is not documented in the model card. | Weight-level MIT is explicit; training-data terms are **unknown**. |
| `weights/ddcolor/pytorch_model.pt` | ModelScope model repo `damo/cv_ddcolor_image-colorization` | Model-repo metadata declares `License: "Apache License 2.0"`; the model repo has no separate LICENSE file and the Hugging Face fallback has not been verified as an equivalent artifact. The upstream GitHub repository's Apache-2.0 covers code only. The model card names ImageNet as training data and states no additional restriction. | Weight-level Apache-2.0 is declared; redistribution notice obligations and training-data terms need review. |
| `Face_Detection/shape_predictor_68_face_landmarks.dat` | dlib.net/files (author's dlib-models repository) | dlib code is Boost 1.0, which covers "software and accompanying documentation" only. The dlib-models README states the iBUG 300-W dataset license excludes commercial use, that the trained model *"can't be used in a commercial product"*, and refers commercial users to Imperial College London; the author declines to give a legal opinion. | **Non-commercial research only.** Commercial use is not authorised: replace the model or obtain written permission. |
| `Face_Detection/dlib_face_recognition_resnet_model_v1.dat` | dlib.net/files (author's dlib-models repository) | The dlib-models repository LICENSE is **CC0-1.0**, and the README/blog state the models are public domain (*"you can use it for anything you want"*). The network was trained on FaceScrub (CC BY-NC-ND 3.0), VGG Face (CC BY-NC 4.0) and scraped images; the author explicitly does not opine on whether those dataset terms flow through to the weights. | The author's weight grant is clear; **training-data transitivity is unresolved** — legal review before commercial use. |

Local audit evidence for these rows (produced for the v3.1.2 review; not distributed with
the repository): `_repair/_weights_license_review/WEIGHTS-29-MATRIX.md` with
`evidence-bob.md`, `evidence-dlib.md`, `evidence-ddcolor-yunet.md`; unresolved items are
tracked in `_repair/_weights_license_review/ACQUISITION-PLAN.md` section 0.

Two consequences for this project: (1) the container images deliberately ship no weights —
some artifacts are restricted and none has unconditional redistribution clearance;
(2) `tests/gpu` and real-inference gates stay blocked until the rights-holder
confirmations above are obtained and all 29 weights are fetched and verified (29/29)
against the committed manifest.

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
