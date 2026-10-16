#!/usr/bin/env bash
# Download all model weights (BOB restoration chain + DDColor colorization).
# Usage: bash scripts/download_weights.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==> Face_Detection: dlib 68-point face landmarks"
if [ ! -f Face_Detection/shape_predictor_68_face_landmarks.dat ]; then
  curl -L -o /tmp/shape_predictor_68_face_landmarks.dat.bz2 \
    http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
  bzip2 -d -c /tmp/shape_predictor_68_face_landmarks.dat.bz2 \
    > Face_Detection/shape_predictor_68_face_landmarks.dat
  rm -f /tmp/shape_predictor_68_face_landmarks.dat.bz2
else
  echo "    already exists, skipping"
fi

echo "==> Face_Detection: dlib 128-D face embeddings (identity metric, plan §17)"
# Part of the committed weights manifest, so a deployment that skips it fails the
# startup integrity check with "Missing file". The metric falls back to a
# dependency-free descriptor, but the check does not fall back at all.
FACE_REC="Face_Detection/dlib_face_recognition_resnet_model_v1.dat"
if [ ! -f "$FACE_REC" ]; then
  curl -L -o /tmp/dlib_face_recognition_resnet_model_v1.dat.bz2 \
    http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2
  bzip2 -d -c /tmp/dlib_face_recognition_resnet_model_v1.dat.bz2 > "$FACE_REC"
  rm -f /tmp/dlib_face_recognition_resnet_model_v1.dat.bz2
else
  echo "    $FACE_REC exists, skipping"
fi

echo "==> Face_Enhancement: face enhancement weights"
# Test a weight file, not the directory. The checkout ships
# Face_Enhancement/checkpoints/ (empty), so `-d` was true, the download was skipped
# and the face stage came up with no checkpoint. A wrong-looking path re-downloads,
# which is the safe direction if upstream ever changes its layout.
FACE_CKPT="Face_Enhancement/checkpoints/Setting_9_epoch_100/latest_net_G.pth"
if [ ! -f "$FACE_CKPT" ]; then
  curl -L -o /tmp/face_checkpoints.zip \
    https://facevc.blob.core.windows.net/zhanbo/old_photo/pretrain/Face_Enhancement/checkpoints.zip
  (cd Face_Enhancement && unzip -o /tmp/face_checkpoints.zip)
  rm -f /tmp/face_checkpoints.zip
else
  echo "    $FACE_CKPT exists, skipping"
fi

echo "==> Global: global restoration/scratch detection weights"
GLOBAL_CKPT="Global/checkpoints/detection/FT_Epoch_latest.pt"
if [ ! -f "$GLOBAL_CKPT" ]; then
  curl -L -o /tmp/global_checkpoints.zip \
    https://facevc.blob.core.windows.net/zhanbo/old_photo/pretrain/Global/checkpoints.zip
  (cd Global && unzip -o /tmp/global_checkpoints.zip)
  rm -f /tmp/global_checkpoints.zip
else
  echo "    $GLOBAL_CKPT exists, skipping"
fi

echo "==> YuNet: face detector ONNX (OpenCV Zoo; the manifest requires it)"
# 232 KB, read once per process. Same reason as the dlib embeddings: it is in
# config/weights_manifest.json, and verify_weights() refuses to start without it.
YUNET="weights/yunet/face_detection_yunet_2023mar.onnx"
mkdir -p weights/yunet
if [ ! -f "$YUNET" ]; then
  curl -L --retry 3 -o "$YUNET" \
    https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx \
  || curl -L --retry 3 -o "$YUNET" \
    https://gitee.com/mirrors/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
else
  echo "    $YUNET exists, skipping"
fi

echo "==> DDColor: colorization weights (ModelScope; falls back to HuggingFace on failure)"
mkdir -p weights/ddcolor
if [ ! -f weights/ddcolor/pytorch_model.pt ]; then
  curl -L --retry 3 -o weights/ddcolor/pytorch_model.pt \
    https://modelscope.cn/models/damo/cv_ddcolor_image-colorization/resolve/master/pytorch_model.pt \
  || curl -L --retry 3 -o weights/ddcolor/pytorch_model.pt \
    https://huggingface.co/piddnad/ddcolor_modelscope/resolve/main/pytorch_model.pt
else
  echo "    already exists, skipping"
fi

echo "==> Done. If this is a first deployment, re-generate the weights manifest:"
echo "    python -m fiximg.cli.download_weights generate"

