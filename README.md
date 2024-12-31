# Fed-PFR
Federated Learning-based Road Surveillance System in Distributed CCTV Environment: Pedestrian Fall Recognition using Spatio-Temporal Attention Networks

![Concept of federated learning](Figs/Figure1.png)

**Figure 1: Concept of Federated Learning**

The figure demonstrates the core concept of federated learning in a distributed CCTV environment:
- Local Training: Each CCTV node processes local video data and trains a model using spatio-temporal attention mechanisms without sharing raw data.
- Federated Aggregation: The local models periodically send updated parameters to a central server for aggregation.
- Global Model: The server builds a global model by combining the parameters, enabling collaborative learning across all nodes.
- Privacy Preservation: This approach ensures data privacy by keeping sensitive video data on local devices while benefiting from collective intelligence.

This architecture is designed to address the challenges of scalability, real-time performance, and data privacy in pedestrian fall detection systems.

## Abstract
Intelligent CCTV systems are highly effective in monitoring pedestrian and vehicular traffic and identifying anomalies in the roadside environment. In particular, it is necessary to develop an effective recognition system to address the problem of pedestrian falls, which is a major cause of injury in road traffic environments. However, the existing systems have challenges such as communication constraints and performance instability. In this paper, we propose a novel fall recognition system based on Federated Learning (FL) to solve these challenges. The proposed system utilizes a GAT combined with LSTM and attention layers to extract spatio-temporal features, which can more accurately identify pedestrian falls. Each road CCTV works as an independent client to generate local data, and the server aggregates these models to learn a global model. This ensures robust operation in different views and environments, and solves the bottleneck of data communication and security challenges. We validated the feasibility and applicability of the FL-based fall recognition method by implementing the prototype and applying it to the UP-FALL benchmark dataset, which is widely used for fall recognition.

![The overall structure of the proposed FL-based fall recognition model](Figs/Figure2.png)

**Figure 2: The overall structure of the proposed FL-based fall recognition model**
The overall structure of the proposed FL-based fall recognition model is as follows: First, at the beginning of each round $t$, client $c$ downloads the global model weights $w_{t-1}^s$ of the previous round. Then, each client trains a local model and uploads the trained weights $w_t^{c_i}$ ($i=1, 2, \ldots, n$) to the server. The server uses the FedAvg algorithm to aggregate the global model weights $w_t^s$. Each local model consists of a network that recognizes pedestrian falls (see local model, right in the figure).

This repository contains demo code for implementing the system. The dataset can be downloaded from the site [Link](https://sites.google.com/up.edu.mx/har-up/).

## System Requirements
- **Operating System**: Ubuntu 22.04.3 LTS
- **CPU**: 12th Gen Intel(R) Core(TM) i7-12700F
- **GPU**: NVIDIA GeForce RTX 4070 (12GB VRAM)
- **System Memory (RAM)**: 32GB

## Usage

### Requirements
First, clone the repository
```bash
git clone https://github.com/Kim-Byeong-Hun/Fed-PFR.git
```
Then download the required packages.
```bash
pip install -r requirements.txt
```
### Preparation
First, download the dataset from [Link](https://sites.google.com/up.edu.mx/har-up/).

The dataset folder should be saved in the following format.

### Pre-processing


## Reference
- [YOLOv8](https://github.com/ultralytics/ultralytics)
