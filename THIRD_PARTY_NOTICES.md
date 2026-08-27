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
