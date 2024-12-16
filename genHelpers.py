import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from scipy.special import expit

#calculates the error for each class i.e. how many 7's did you get wrong? etc.
def classwise_error_rate(predictions, true_labels, num_classes=10):
    # Ensure predictions and true_labels are numpy arrays and are 1-dimensional
    predictions = np.asarray(predictions).flatten()
    true_labels = np.asarray(true_labels).flatten()

    error_rates = {}

    for cls in range(num_classes):
        # Get indices where the true label is the current class
        class_indices = (true_labels == cls)

        # Use the boolean mask directly on the 1D arrays
        class_predictions = predictions[class_indices]
        class_true_labels = true_labels[class_indices]

        # Calculate error rate for the current class
        error_rate = np.mean(class_predictions != class_true_labels)

        # Store the error rate for this class formatted as a percentage
        error_rates[cls] = f"{error_rate:.2%}"

    return error_rates

def error_rate(predictions, true_labels):
    return np.mean(predictions != true_labels)


import numpy as np


# Define activation functions
def identity(x):
    return x


def sigmoid(x):
    return expit(x)


def sinusoidal(x):
    return np.sin(x)


def relu(x):
    return np.maximum(x, 0)


# Randomized feature mapping function

def randomized_feature_mapping(X, L, g=np.identity):
    """
    Generate a random feature mapping for the input data X using the function g.

    Parameters:
    - X: numpy array of shape (n_samples, n_features), input data
    - L: int, number of random features to generate
    - g: function, activation function to use in the feature mapping

    Returns:
    - h_X: numpy array of shape (n_samples, L), transformed feature space
    - W: numpy array of shape (L, n_features), random weight matrix
    - b: numpy array of shape (L,), random bias vector
    """
    n_samples, n_features = X.shape

    # Generate random matrix W and bias vector b
    W = np.random.normal(0, 1, (L, n_features))  # W ~ N(0, 1) with shape (L, n_features)
    b = np.random.normal(0, 1, L)  # b ~ N(0, 1) with shape (L,)

    # Compute random feature mapping
    h_X = g(np.dot(X, W.T) + b)  # h(X) = g(W * X^T + b)

    return h_X, W, b

def randomized_feature_mapping_with_params(X, W, b, g=np.identity):
    """
    Apply a precomputed random feature mapping to new data X using the given W and b.

    Parameters:
    - X: numpy array of shape (n_samples, n_features), input data
    - W: numpy array of shape (L, n_features), random weight matrix
    - b: numpy array of shape (L,), random bias vector
    - g: function, activation function to use in the feature mapping

    Returns:
    - h_X: numpy array of shape (n_samples, L), transformed feature space
    """
    return g(np.dot(X, W.T) + b)  # h(X) = g(W * X^T + b)


def plot_confusion_matrix(true_labels, predicted_labels, num_classes):
    """
    Generate and plot a confusion matrix for the given true and predicted labels,
    including row and column totals.

    Parameters:
    true_labels (numpy.ndarray): The ground truth labels.
    predicted_labels (numpy.ndarray): The predicted labels.
    num_classes (int): The number of classes in the dataset.
    """
    # Compute the confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=np.arange(num_classes))

    # Compute row and column totals
    row_totals = cm.sum(axis=1)
    col_totals = cm.sum(axis=0)
    overall_total = cm.sum()

    # Extend the confusion matrix to include totals
    cm_with_totals = np.zeros((num_classes + 1, num_classes + 1), dtype=int)
    cm_with_totals[:num_classes, :num_classes] = cm
    cm_with_totals[:num_classes, -1] = row_totals  # Add row totals
    cm_with_totals[-1, :num_classes] = col_totals  # Add column totals
    cm_with_totals[-1, -1] = overall_total  # Add overall total

    # Plot the extended confusion matrix as a heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(cm_with_totals, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix with Totals")
    plt.colorbar()

    # Label the axes
    tick_marks = np.arange(num_classes + 1)
    labels = [str(i) for i in range(num_classes)] + ['Total']
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)

    # Add the numerical values to the confusion matrix cells
    for i in range(num_classes + 1):
        for j in range(num_classes + 1):
            plt.text(j, i, f"{cm_with_totals[i, j]}",
                     horizontalalignment="center",
                     color="white" if cm_with_totals[i, j] > cm_with_totals.max() / 2 else "black")

    # Add axis labels
    plt.ylabel("True Labels")
    plt.xlabel("Predicted Labels")
    plt.tight_layout()
    plt.show()
