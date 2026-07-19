# Vision Language Models for Context-Aware Scene Understanding in Autonomous Systems

MSc thesis implementation — Erasmus Mundus Joint Master in Intelligent Field Robotic Systems (IFRoS), Eötvös Loránd University (ELTE), Budapest.

This repository contains the ROS2 Humble implementation of three semantic mapping pipelines for indoor mobile robotics, developed and benchmarked as part of the thesis *"Vision Language Models for Context-Aware Scene Understanding in Autonomous Systems."*

> **Note on repository contents:** This repository contains source code only. Rosbag recordings, generated maps, detection outputs, and other experiment result/data folders have been intentionally excluded to keep the repository lightweight and focused on the implementation. Result summaries and metrics are described below; the full experimental data is documented in the thesis itself.

## Overview

The thesis implements and benchmarks three semantic mapping pipelines, all built on a shared spatial estimation backbone:

| Pipeline | Detection / Segmentation | Semantic Labelling |
|---|---|---|
| **Pipeline 1** | YOLO26 (classical baseline) | Fixed label set |
| **Pipeline 2** | YOLO26 | Qwen2.5-VL 3B post-processing |
| **Pipeline 3** | SAM2-small (open-vocabulary) | Qwen2.5-VL 3B |

**Shared spatial estimation backbone:** ray casting → pairwise triangulation → DBSCAN clustering, used to localize detected/segmented objects in 3D from 2D image detections and robot pose.

**Evaluation:** 5 Gazebo simulation environments + 1 real-world lab setting, assessed across 5 metrics — SPE, SLA, F1-Score, MCS, and SSS.

**Key findings:**
- Pipeline 2 achieves perfect SLA (Semantic Labelling Accuracy) in two simulation environments.
- Pipeline 3 achieves MCS = 1.000 in the real-world experiment, but its performance degrades on textureless Gazebo objects — a substantive finding on the sim-to-real visual gap for open-vocabulary VLM-based perception.

## Tech Stack

- **Middleware:** ROS2 Humble
- **Simulation:** Gazebo
- **Navigation:** Nav2
- **Models:** YOLO26 (Ultralytics), SAM2-small, Qwen2.5-VL 3B (4-bit quantized, local deployment), Moondream2, Llama 3.2 3B (via Ollama)
- **Sensors:** Fisheye camera (170° FoV, equidistant projection)

## Repository Structure

```
src/                    # ROS2 packages: robot description, Gazebo simulation models,
                         # detection/mapping pipelines, spatial estimation backbone
.gitignore
LICENSE                 # Apache License 2.0
CITATION.cff             # Citation metadata (see "Cite this repository" on GitHub)
```

## Branches

Each branch corresponds to a distinct experiment, pipeline variant, or post-thesis extension developed during this work. `main` holds the core/consolidated implementation; other branches contain the specific pipeline runs and extensions described below.

| Branch | Purpose |
|---|---|
| `main` | Core implementation and consolidated development (depth estimation approach and pipeline tuning) |
| `SIM_YOLO_MappingROS` | Pipeline 1 (YOLO26 baseline) evaluation across the 5 Gazebo simulation environments |
| `VLM_LabellingROS` | Pipeline 2 (YOLO26 + Qwen2.5-VL) semantic labelling experiments |
| `ViT_ROS` | Pipeline 3 (SAM2 + Qwen2.5-VL) open-vocabulary experiments and results |
| `RealWorld_ROS` | Real-world lab deployment and result recordings |
| `two_cam_improvement` | Post-thesis extension: dual-camera architecture (simulation) |
| `two_cam_improvement_realworld` | Post-thesis extension: dual-camera architecture validated on real-world data |
| `live_stream_mapping` | Post-thesis extension: live/streaming semantic mapping |
| `llm_orchestrator` | Post-thesis extension: natural-language-to-navigation-goal orchestration via Llama 3.2 3B (Ollama) and Nav2 |

> Branch descriptions above are inferred from commit history and naming for this README — please verify each mapping is accurate before treating this table as final.

## Setup

```bash
# Clone
git clone https://github.com/ByteMastee/vlm-thesis.git
cd vlm-thesis

# Build (inside a ROS2 Humble environment / Docker container)
colcon build --symlink-install
source install/setup.bash
```

Model weights for Qwen2.5-VL, SAM2-small, and Llama 3.2 3B are not included in this repository and should be obtained separately (see thesis for exact versions and deployment configuration).

## Results Summary

Full quantitative results (SPE, SLA, F1-Score, MCS, SSS across all 5 simulation environments and the real-world setting) are reported in the thesis. Headline results:

- Pipeline 2 (YOLO26 + Qwen2.5-VL): perfect SLA in 2/5 simulation environments.
- Pipeline 3 (SAM2 + Qwen2.5-VL): MCS = 1.000 in real-world deployment; reduced performance on textureless simulation objects, highlighting a sim-to-real gap for open-vocabulary VLM perception.

## Citation

If you use this code, or build on the ideas presented here, please cite this repository:

```bibtex
@software{sampathkumar2026vlmthesis,
  author  = {Sampath Kumar, Hariram},
  title   = {Vision Language Models for Context-Aware Scene Understanding in Autonomous Systems},
  year    = {2026},
  url     = {https://github.com/ByteMastee/vlm-thesis},
  version = {1.0.0}
}
```

This work was completed as part of an Erasmus Mundus Joint Master's thesis in Intelligent Field Robotic Systems (IFRoS) at Eötvös Loránd University (ELTE), Budapest. A citation file ([`CITATION.cff`](./CITATION.cff)) is included in this repository — GitHub surfaces a **"Cite this repository"** option automatically on the repo page.

*A peer-reviewed publication based on this work is in preparation; this section will be updated with the corresponding paper citation once available.*

## Acknowledgments

This work was supervised by Dr. Zoltán Istenes (ELTE) and Dr. Dániel Horváth and Mr. Renatto Tommasi (European Knowledge Centre Ltd.) as part of the author's Erasmus Mundus Joint Master's thesis in Intelligent Field Robotic Systems (IFRoS).

## License

This project is licensed under the [Apache License 2.0](./LICENSE) — see the LICENSE file for details.
