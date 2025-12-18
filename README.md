# **LiDAR-V2X: Robust Cooperative Perception Under Spoofing Attacks**

This repository contains a simulation framework for evaluating the robustness of cooperative LiDAR-based perception under V2X spoofing attacks. It provides attack injectors, occupancy-grid models, metrics computation, and configuration files to reproduce the experiments described in the associated academic work.

---

## 🚗 **Overview**

Modern connected and autonomous vehicles rely on cooperative perception via V2X communication to overcome occlusions and expand situational awareness. However, the same communication channel broadens the attack surface, enabling adversarial manipulation of perception pipelines.

This framework simulates:

* Point-cloud spoofing over V2X,
* Phantom objects and malicious LiDAR injections,
* Baseline, attack, and attack-with-defense scenarios,
* Low-complexity mitigation strategies compatible with real-time perception.

It operates atop CARLA and generates occupancy grids for quantitative analysis of robustness.

---

## 🧮 **Mathematical Models**

The perception subsystem implements three occupancy-grid formulations:

| Model  | Description                                      | Purpose                                            |
| ------ | ------------------------------------------------ | -------------------------------------------------- |
| **M₀** | Geometric projection and grid quantization       | Baseline LiDAR occupancy                           |
| **M₁** | Temporal accumulation with exponential smoothing | Reduces intermittent noise                         |
| **M₂** | Morphological post-processing                    | Eliminates sparse artifacts and enhances structure |

---

## 🎯 **Metrics**

Three categories of metrics quantify the effects of spoofing and mitigation:

| Category      | Metric           | Description                       |
| ------------- | ---------------- | --------------------------------- |
| Communication | PDR, latency p95 | NR-V2X constraints                |
| Perception    | IoU, FP, FN      | Occupancy-grid consistency        |
| Robustness    | Reaction time    | Frames until spoofing suppression |

These metrics are automatically logged to CSV during execution.

To summarize results:

```bash
python -m scripts.metrics_summary
```

---

## ▶ **How to Run**

### **Prerequisites**

* Python ≥ 3.10
* CARLA Simulator (≥ 0.9.14)
* Linux or Windows

### **Execution**

Baseline run:

```bash
python -m scripts.run_intermediate
```

Attack logs, metrics, and occupancy-grid figures will be produced automatically.

---
