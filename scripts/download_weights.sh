#!/usr/bin/env bash
# 下载全部模型权重（BOB 修复链路 + DDColor 上色）。
# 用法：bash scripts/download_weights.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==> Face_Detection: dlib 68 点人脸关键点"
if [ ! -f Face_Detection/shape_predictor_68_face_landmarks.dat ]; then
  curl -L -o /tmp/shape_predictor_68_face_landmarks.dat.bz2 \
    http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
  bzip2 -d -c /tmp/shape_predictor_68_face_landmarks.dat.bz2 \
    > Face_Detection/shape_predictor_68_face_landmarks.dat
  rm -f /tmp/shape_predictor_68_face_landmarks.dat.bz2
else
  echo "    已存在，跳过"
fi

echo "==> Face_Enhancement: 人脸增强权重"
if [ ! -d Face_Enhancement/checkpoints ]; then
  curl -L -o /tmp/face_checkpoints.zip \
    https://facevc.blob.core.windows.net/zhanbo/old_photo/pretrain/Face_Enhancement/checkpoints.zip
  (cd Face_Enhancement && unzip -o /tmp/face_checkpoints.zip)
  rm -f /tmp/face_checkpoints.zip
else
  echo "    已存在，跳过"
fi

echo "==> Global: 整体修复/划痕检测权重"
if [ ! -d Global/checkpoints ]; then
  curl -L -o /tmp/global_checkpoints.zip \
    https://facevc.blob.core.windows.net/zhanbo/old_photo/pretrain/Global/checkpoints.zip
  (cd Global && unzip -o /tmp/global_checkpoints.zip)
  rm -f /tmp/global_checkpoints.zip
else
  echo "    已存在，跳过"
fi

echo "==> DDColor: 上色权重（ModelScope，失败则回退 HuggingFace）"
mkdir -p pretrained/ddcolor
if [ ! -f pretrained/ddcolor/pytorch_model.pt ]; then
  curl -L --retry 3 -o pretrained/ddcolor/pytorch_model.pt \
    https://modelscope.cn/models/damo/cv_ddcolor_image-colorization/resolve/master/pytorch_model.pt \
  || curl -L --retry 3 -o pretrained/ddcolor/pytorch_model.pt \
    https://huggingface.co/piddnad/ddcolor_modelscope/resolve/main/pytorch_model.pt
else
  echo "    已存在，跳过"
fi

echo "==> 完成。如首次部署请重新生成权重清单："
echo "    python -m config.weights_check generate"

