# Fed-PFR
Federated Learning-based Road Surveillance System in Distributed CCTV Environment: Pedestrian Fall Recognition using Spatio-Temporal Attention Networks

## Abstract
Intelligent CCTV systems are highly effective in monitoring pedestrian and vehicular traffic and identifying anomalies in the roadside environment. In particular, it is necessary to develop an effective recognition system to address the problem of pedestrian falls, which is a major cause of injury in road traffic environments. However, the existing systems have challenges such as communication constraints and performance instability. In this paper, we propose a novel fall recognition system based on Federated Learning (FL) to solve these challenges. The proposed system utilizes a GAT combined with LSTM and attention layers to extract spatio-temporal features, which can more accurately identify pedestrian falls. Each road CCTV works as an independent client to generate local data, and the server aggregates these models to learn a global model. This ensures robust operation in different views and environments, and solves the bottleneck of data communication and security challenges. We validated the feasibility and applicability of the FL-based fall recognition method by implementing the prototype and applying it to the UP-FALL benchmark dataset, which is widely used for fall recognition.

![Concept of federated learning](fig/Figure1.png)
![The overall structure of the proposed FL-based fall recognition model](fig/Figure2.png)

This repository contains demo code for implementing the system. The dataset can be downloaded from the site [Link](https://sites.google.com/up.edu.mx/har-up/).
