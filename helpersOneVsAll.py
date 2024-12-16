import numpy as np

#if label is equal to the class, set y(i) = 1, otherwise y(i) = -1
def labelBinaryData(classifierNum, uncleanedLabels):
    cleanedLabels = []
    for label in uncleanedLabels[0]:
        if label == classifierNum:
            cleanedLabels.append(1)
        else:
            cleanedLabels.append(-1)
    return cleanedLabels

#solves the Normal equations
def train_binary_classifier(X, y):
    # Ensure y is a NumPy array, then reshape it to be a column vector
    y = np.array(y).reshape(-1, 1)

    # Add a bias term by augmenting X with a column of ones
    X_augmented = np.hstack([X, np.ones((X.shape[0], 1))])

    # Use the pseudo-inverse instead of the regular inverse
    params = np.linalg.pinv(X_augmented.T @ X_augmented) @ X_augmented.T @ y

    # Separate the β and α from the computed parameters
    beta = params[:-1].flatten()  # All but the last element
    alpha = params[-1, 0]  # Last element (bias term)

    return beta, alpha

#uses betas and alphas to predict the images
def predict_one_vs_all_full(X, betas, alphas):
    # Collect scores for each class
    scores = np.array([X @ beta + alpha for beta, alpha in zip(betas, alphas)])

    # Choose the class with the highest score
    predicted_labels = np.argmax(scores, axis=0)

    return predicted_labels



