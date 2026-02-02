---
trigger: always_on
---

# Genie: Generative Interactive Environments - Technical Summary

This document summarizes the essential architecture, components, and hyperparameters of the Genie paper to guide the development of Mini-Genie.

## 1. Core Architecture Overview
Genie is a world model comprised of three main components trained in an unsupervised manner from video data:
1.  **Video Tokenizer (VQ-VAE)**: Compresses video frames into discrete tokens.
2.  **Latent Action Model (LAM)**: Learns latent actions $\textbf{a}$ between pairs of frames $(\textbf{x}_t, \textbf{x}_{t+1})$ without ground truth labels.
3.  **Dynamics Model**: An autoregressive transformer that predicts the next frame tokens given past tokens and latent actions.

## 2. Component Details & Hyperparameters

### 2.1 Video Tokenizer (ST-VQ-VAE)
A Spatiotemporal VQ-VAE that compresses video sequences into discrete tokens.

*   **Architecture**: ST-ViViT (Spatiotemporal Video Vision Transformer).
*   **Compression**:
    *   Compresses $T \times H \times W$ video into $T \times h \times w$ discrete tokens.
    *   Typical patch size: $4 \times 4$.
*   **Objective**: Reconstruction loss + VQ codebook loss.
*   **Reproducible Case (CoinRun) Config**:
    *   **Encoder/Decoder**: 8 layers, `d_model`=512, 8 heads.
    *   **Codebook**: 1024 codes, latent dimension = 32.
    *   **Batch Size**: 48 sequences of length 16 (768 images total).

### 2.2 Latent Action Model (LAM)
Learns a discrete set of latent actions to control the environment.

*   **Input**: Past frame $\textbf{x}_t$ and future frame $\textbf{x}_{t+1}$ (quantized tokens).
*   **Output**: Latent action $\textbf{a}_t$.
*   **Architecture**: VQ-VAE style encoder-decoder.
*   **Key Constraint**: The number of latent codes (`num_codes`) should be small to limit the action space (e.g., 6-10 for simple games).
*   **Reproducible Case (CoinRun) Config**:
    *   **Encoder/Decoder**: 8 layers, `d_model`=512, 8 heads.
    *   **Action Codebook**: **6-10 codes** (crucial for learnable controls).
    *   **Latent Dim**: 32.

### 2.3 Dynamics Model (MaskGIT Transformer)
Predicts the next frame's tokens based on history and actions.

*   **Architecture**: Decoder-only Transformer (MaskGIT).
*   **Input**: Sequence of frame tokens $\textbf{z}_{1:t}$ and latent actions $\textbf{a}_{1:t}$.
*   **Prediction**: Autoregressive prediction of $\textbf{z}_{t+1}$.
*   **Reproducible Case (CoinRun) Config**:
    *   **Layers**: 12.
    *   **Model Dimension**: 512.
    *   **Heads**: 8.
*   **Inference**: Uses MaskGIT iterative decoding (typically 25 steps) for high-quality generation.

## 3. Training Pipeline
1.  **Phase 1**: Train the **Video Tokenizer** on raw video clips.
2.  **Phase 2**: Freeze the tokenizer. Train the **Latent Action Model (LAM)** and **Dynamics Model** jointly (or LAM first, then Dynamics).
    *   LAM infers actions $\hat{a}_t$ from $(\textbf{z}_t, \textbf{z}_{t+1})$.
    *   Dynamics model predicts $\textbf{z}_{t+1}$ given $(\textbf{z}_{1:t}, \hat{a}_{1:t})$.

## 4. Reproducible "Mini-Genie" Config (16GB VRAM target)
Based on the CoinRun case study in the paper:

| Component | Hyperparameter | Value |
| :--- | :--- | :--- |
| **Data** | Sequence Length | 16 frames |
| **Tokenizer** | Patch Size | 4 |
| | Codebook Size | 1024 |
| | Embed Dim | 32-64 |
| **LAM** | Action Codes | **6 - 8** |
| | Embed Dim | 32-64 |
| **Dynamics** | Layers | 8 - 12 |
| | Embed Dim | 512 |
| | Heads | 8 |
| **Training** | Optimizer | AdamW |
| | LR | ~1e-4 |
