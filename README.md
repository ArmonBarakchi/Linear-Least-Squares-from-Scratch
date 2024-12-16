# Linear-Least-Squares-from-Scratch


This repository implements linear least squares classification for supervised learning, built entirely from scratch, using both one-vs-one and one-vs-all approaches. The implementation is applied to classify handwritten digits from the MNIST dataset and demonstrates key concepts in binary and multi-class classification.

Features
Binary Least Squares Classifier: Implements a linear model for binary classification using the least squares optimization method.
One-vs-All Multi-Class Classification: Trains a separate binary classifier for each class to distinguish it from all others, effectively transforming the multi-class problem into multiple binary problems.
One-vs-One Multi-Class Classification: Constructs binary classifiers for each pair of classes, resolving classifications by majority voting from all pairwise comparisons.
MNIST Dataset Integration: Leverages the widely recognized MNIST dataset of handwritten digits, providing a practical benchmark for evaluating the classifiers.
Feature Mapping and Non-Linear Extensions
Beyond linear classification in the input space, this project explores randomized feature-based least squares classifiers, where the input data is mapped into a higher-dimensional feature space before applying the least squares method. This step allows the implementation to approximate non-linear decision boundaries, significantly enhancing classification performance.

The feature mapping process involves:

Randomized Feature Transformation: A random matrix and bias vector are generated with entries drawn from Gaussian distributions. These parameters transform the input data into a new feature space.
Non-Linear Activation Functions: The transformed data passes through a non-linear activation function, enabling the classifier to capture complex relationships in the data. Supported activation functions include:
Identity Function: 
𝑔
(
𝑥
)
=
𝑥
g(x)=x, preserving linearity.
Sigmoid Function: 
𝑔
(
𝑥
)
=
1
1
+
𝑒
−
𝑥
g(x)= 
1+e 
−x
 
1
​
 , useful for smooth transitions.
Sinusoidal Function: 
𝑔
(
𝑥
)
=
sin
⁡
(
𝑥
)
g(x)=sin(x), capturing periodic patterns.
ReLU (Rectified Linear Unit): 
𝑔
(
𝑥
)
=
max
⁡
(
0
,
𝑥
)
g(x)=max(0,x), ideal for sparse activation in feature spaces.
Analysis of Feature Mappings
The project examines the performance of classifiers trained on feature-mapped data across several dimensions:

Accuracy on Training and Testing Data: Comparison of feature mappings to determine which non-linearities generalize best to unseen data.
Error Rate vs. Number of Features: Investigates how the dimensionality of the feature space influences classifier performance.
Robustness to Noise: Evaluates the classifiers' ability to handle noisy inputs, identifying thresholds where performance deteriorates.
Highlights
Built Entirely from Scratch: Implements core machine learning concepts without relying on external libraries, providing an educational deep dive into least squares classification.
Performance Evaluation: Includes error rate calculations, confusion matrices, and detailed comparisons between one-vs-one and one-vs-all strategies.
Comprehensive Feature Mapping Analysis: Extends linear classifiers to non-linear problems through randomized feature transformations and non-linear activations.
Detailed Report: Explains the mathematical foundation, simulation results, and classifier interpretations.
Getting Started
Load the Dataset: Use the provided scripts to import and preprocess the MNIST dataset.
Train Classifiers:
Implement binary least squares classifiers as building blocks.
Train both one-vs-one and one-vs-all multi-class classifiers.
Extend the design to randomized feature spaces with non-linear activations.
Evaluate Performance:
Measure accuracy using confusion matrices and error rates.
Compare results across feature mappings and assess robustness to noise.
