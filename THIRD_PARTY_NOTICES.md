# 第三方组件说明

## 项目自身许可

- 本项目自身代码以 **MIT License** 发布，完整文本见根目录 `LICENSE`；
- 第三方组件保留各自许可证（见下文），不因项目 MIT 许可而改变。

## Bringing Old Photos Back to Life（修复/检测/面部增强模型）

- 来源：https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life
  （CVPR 2020 oral，Microsoft Corporation）
- 许可证：MIT License（完整文本见 `LICENSE-Bringing-Old-Photos-Back-to-Life`）
- 使用范围：本项目 `Global/`、`Face_Detection/`、`Face_Enhancement/` 目录

## DDColor（老照片上色模块）

- 来源：https://github.com/piddnad/DDColor （ICCV 2023，DAMO Academy / Alibaba Group）
- 许可证：Apache License 2.0（完整文本见 `ddcolor/LICENSE`）
- 使用范围：本项目 `ddcolor/` 与 `basicsr/archs/ddcolor_arch_utils/` 目录
- 权重来源：ModelScope `damo/cv_ddcolor_image-colorization/pytorch_model.pt`
  （或 HuggingFace `piddnad/ddcolor_modelscope`），Apache-2.0 项目官方发布
