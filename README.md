OLEGUS
======


### 1. Introduction
This project consists of classes and utility, that can be assembled into a trainable model.

### 2. Content
- Models (like perceptron and multi head attention) are in `/parts`
- Utility (preparing and analyzing data) is in `/utility`
- Files without special folder are too abstract (constants, universal classes and examples)

### 3. Universal
File `universal.py` contains:
- Class Model. Does nothing, used to search class names
- Class Optimizable. Defines `drop_gradient()` and `descent()` as `self.optimizer`'s corresponding methods
- Class NoDropout. Defines `predict()` as `forward()`