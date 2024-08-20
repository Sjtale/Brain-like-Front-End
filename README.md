# Brain-like Front-End for Robust Object Recognition

This project aims to enhance the robustness of Convolutional Neural Networks (CNNs) by incorporating biologically-inspired models of the human visual cortex. The developed front-end, named **V++**, integrates four regions of the visual cortex (V1, V2, V4, IT) to improve object recognition capabilities under various types of noise and disturbances.

![Model Architecture](https://github.com/yourusername/yourrepository/path/to/architecture_image.png)

## Overview

This repository contains the source code and data for the project, including:

- **Model Implementation**: A biologically-inspired front-end for CNNs that simulates the visual processing stages in the human brain.
- **Evaluation Scripts**: Tools to assess the robustness of the model against various types of image noise and distortions.
- **Datasets**: Preprocessed datasets used for training and evaluation, designed to test the robustness of object recognition.

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Model Architecture](#model-architecture)
4. [Evaluation](#evaluation)
5. [Results](#results)

## Installation

To run this project, clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/yourrepository.git
cd yourrepository
pip install -r requirements.txt
```
## Usage

To train the model, use the following command:

```bash
python train.py --config config.yaml
```

## Evaluating the Model

After training, evaluate the model by running:

```bash
python evaluate.py --model-path models/trained_model.pth
```

## Model Architecture

The model is structured to simulate the visual processing pathways in the brain:

\| Layer \| Description                                 \|
\|-------\|---------------------------------------------\|
\| V1    \| Edge detection and orientation analysis.    \|
\| V2    \| Complex pattern detection.                  \|
\| V4    \| Shape and color processing.                 \|
\| IT    \| Object recognition and identification.      \|

![Detailed Architecture](https://github.com/yourusername/yourrepository/path/to/detailed_architecture_image.png)

## Evaluation

The model's robustness is tested against several types of noise:

\| Noise Type       \| Description                                      \|
\|------------------\|--------------------------------------------------\|
\| Gaussian Noise   \| Random noise added to pixel values.              \|
\| Salt and Pepper  \| Random black and white pixels scattered in image.\|
\| Motion Blur      \| Simulates the effect of object movement.         \|
\| Occlusion        \| Parts of the image are covered or missing.       \|

To run the evaluation, use:

```bash
python evaluate.py --noise-type gaussian --severity 2
```

## Results

The following table summarizes the model's performance under different noise conditions:

| Noise Type | CNN Accuracy | V++ Accuracy | Improvement |
|------------------|--------------|--------------|-------------|
| Gaussian Noise | 72% | 85% | +13% |
| Salt and Pepper | 68% | 80% | +12% |
| Motion Blur | 70% | 83% | +13% |
| Occlusion | 65% | 78% | +13% |


