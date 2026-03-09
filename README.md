# Diff2SP

This repository is the submission package for the paper `diff2sp.pdf`.

The paper studies **scenario generation for stochastic programming** using diffusion models. The main goal is not only to generate statistically realistic uncertainty scenarios, but also to make those scenarios more useful for the downstream optimization problem.

## Core Idea

Traditional scenario generation methods usually optimize only data fidelity. Diff2SP adds a second requirement: generated scenarios should also support high-quality downstream decisions.

The method combines:
- a **conditional diffusion model** for scenario generation
- a **transformer-based denoiser** to capture cross-variable dependence
- an **optimization-guided loss** to align generation with stochastic programming objectives

In the paper, the motivating application is power systems, where uncertainty comes from correlated wind, solar, and load variables.

## Method Overview

Diff2SP has four main components.

### 1. Conditional diffusion generation

The model learns to generate class-conditional scenarios by reversing a noise process. Compared with GANs, diffusion training is more stable and better suited to covering multimodal and correlated distributions.

### 2. Transformer denoiser

Instead of using a purely local architecture, Diff2SP uses a transformer-based denoiser so the reverse model can capture long-range dependence and cross-feature correlation in multivariate scenarios.

### 3. Optimization-guided training

This is the key part of the paper. The training objective is not only:
- noise prediction loss
- reconstruction loss

It also includes an **optimization loss** derived from the downstream stochastic programming task. This makes the generated scenarios better aligned with decision quality, not just statistical similarity.

### 4. Ablation design

The empirical study compares four variants:
- `gan`: GAN baseline
- `noise`: vanilla diffusion baseline
- `noopt`: diffusion without optimization loss
- `full`: full Diff2SP with optimization-guided training

This ablation is used to isolate the value of diffusion itself and the additional value of optimization-aware learning.

## Experimental Structure

The submission is organized around the two main experimental sections in the paper.

### `numerical/`

This directory corresponds to **Section 5.1**, the synthetic or numerical experiments.

Its purpose is to test the generative behavior of the model in a more controlled setting, where structural properties of the generated scenarios can be analyzed more directly.

### `opf/`

This directory corresponds to **Section 5.2**, the real-world power systems case study.

In this setting:
- the uncertainty input is an 18-dimensional regional feature vector
- generated scenarios are mapped to a 30-bus demand representation
- the downstream task is evaluated with DC optimal power flow
- scenario quality is measured by both distributional fidelity and optimization performance

This directory contains the main implementation of the paper's optimization-guided Diff2SP framework.

## Summary

Diff2SP is a diffusion-based scenario generation framework for stochastic programming. Its main contribution is to move beyond pure data generation and train the model in a way that is explicitly aware of downstream optimization quality. The paper shows this through both controlled numerical experiments and a real-world power systems case study.
