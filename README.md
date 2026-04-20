# Hybrid Quantum Convolutional Neural Network for Pneumonia Classification

**2026 Yearly Homework Project (YHP)**

This is the code repository for my YHP as explained in my final submission!

My YHP  explores whether quantum computing can improve medical image
classification. A Hybrid Quantum Convolutional Neural Network (QCNN) is
trained on pediatric chest X-ray images to classify pneumonia vs. normal
cases, and its performance is compared against a classical CNN baseline.

---

## Overview

The hybrid model combines a classical convolutional feature extractor with
two quantum circuits — one using **amplitude encoding** and one using
**angle encoding** — whose outputs are fused for final binary classification.
Both models were trained under identical conditions for a fair comparison.

**Result:** The Hybrid QCNN achieved **81.3% accuracy** vs. **77.2%** for
the classical CNN, with statistical significance confirmed via a one-sided 
Wilcoxon signed-rank test (p = 0.033 < 0.05).
