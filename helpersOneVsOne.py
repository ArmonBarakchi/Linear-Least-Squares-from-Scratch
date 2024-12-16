import numpy as np

#solve normal equation for every binary classifier
def train_ovo_classifier(X, Y, class_i, class_j):
    # Ensure Y is a 1D array for proper indexing
    y = Y.flatten()

    # Select only the samples belonging to class_i and class_j
    indices = (y == class_i) | (y == class_j)
    X_filtered = X[indices]
    y_filtered = y[indices]

    # Relabel: +1 for class_i, -1 for class_j
    y_binary = np.where(y_filtered == class_i, 1, -1)

    # Add a bias term to X for the linear model
    X_augmented = np.hstack([X_filtered, np.ones((X_filtered.shape[0], 1))])

    # Solve for [β; α] using the pseudo-inverse
    params = np.linalg.pinv(X_augmented.T @ X_augmented) @ X_augmented.T @ y_binary

    # Separate the β and α from params
    beta = params[:-1]  # All but the last element
    alpha = params[-1]  # Last element is the bias term

    return beta, alpha


#loop through every combination of binary classifiers with class combos i,j
#such that i<j and create binary classifier
def train_ovo_classifiers(X, y, num_classes=10):
    classifiers = {}

    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            # Train the classifier for class i vs class j
            beta, alpha = train_ovo_classifier(X, y, i, j)
            classifiers[(i, j)] = (beta, alpha)

    return classifiers

#count votes up from the various classifiers and pick the number with the most votes
def predict_ovo(X, classifiers, num_classes=10):
    votes = np.zeros((X.shape[0], num_classes), dtype=int)  # Matrix to count votes for each class

    # Iterate through each classifier (i, j)
    for (i, j), (beta, alpha) in classifiers.items():
        # Calculate the decision for class i vs class j
        scores = X @ beta + alpha
        predictions = np.sign(scores)

        # Increment votes based on predictions
        votes[:, i] += (predictions == 1)  # Vote for class i
        votes[:, j] += (predictions == -1)  # Vote for class j

    # Choose the class with the maximum votes for each sample
    predicted_labels = np.argmax(votes, axis=1)

    return predicted_labels


