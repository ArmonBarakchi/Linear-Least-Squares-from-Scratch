import helpersOneVsAll as helpers
import numpy as np
from scipy.io import loadmat
import genHelpers
import matplotlib.pyplot as plt

#which part of the program to run
problem1 = 0
withFeatureMapping = 1
problem2_2 = 0
problem2_2Test = 0

#initialize data
data = loadmat('mnist.mat')
images = data['trainX']  # 60000 images -> each an array of 784 pixels
test_images = data['testX']
labels = data['trainY']  # 60000 labels
test_labels = data['testY']

# Convert to float32 and normalize
images = images.astype(np.float32) / 255.0
test_images = test_images.astype(np.float32) / 255.0
num_classes = 10  # For MNIST, this would be 10 for digits 0-9
betas = []
alphas = []

if (problem1):
    #train classifiers
    for k in range(num_classes):
        # Label data for class k vs all others
        binary_labels = helpers.labelBinaryData(k, labels)

        # Train a binary classifier for this class
        beta, alpha = helpers.train_binary_classifier(images, binary_labels)

        # Store the classifier's parameters
        betas.append(beta)
        alphas.append(alpha)

    #get predictions for test/training
    predicted_test = helpers.predict_one_vs_all_full(test_images, betas, alphas)
    predicted_training = helpers.predict_one_vs_all_full(images, betas, alphas)

    #get classwise and total errors for test/training
    error_test = genHelpers.classwise_error_rate(predicted_test, test_labels)
    total_test_error = genHelpers.error_rate(predicted_test, test_labels)
    error_training = genHelpers.classwise_error_rate(predicted_training, labels)
    total_training_error = genHelpers.error_rate(predicted_training,labels)

    #generate confusion matrices
    true_test_labels = test_labels.flatten()
    true_training_labels = labels.flatten()
    genHelpers.plot_confusion_matrix(true_test_labels, predicted_test, num_classes=10)
    genHelpers.plot_confusion_matrix(true_training_labels, predicted_training, num_classes=10)

    #print results
    print(error_test)
    print(total_test_error)
    print(error_training)
    print(total_training_error)

if (withFeatureMapping):
    activationFunctions = [genHelpers.identity, genHelpers.relu,
                           genHelpers.sigmoid, genHelpers.sinusoidal]
    for activationFunction in activationFunctions:
        alphas.clear()
        betas.clear()
        #apply feature mapping to training images
        transformed_training_images, W, b = genHelpers.randomized_feature_mapping(images, L=1000, g=activationFunction)

        #train classifiers
        for k in range(num_classes):
            # Label data for class k vs all others
            binary_labels = helpers.labelBinaryData(k, labels)

            # Train a binary classifier for this class
            beta, alpha = helpers.train_binary_classifier(transformed_training_images, binary_labels)

            # Store the classifier's parameters
            betas.append(beta)
            alphas.append(alpha)

        #apply same feature mapping to test images
        transformed_images = genHelpers.randomized_feature_mapping_with_params(test_images, W, b,  g=activationFunction)

        # Predict using the one-versus-all classifiers on test/training data
        predicted_labels = helpers.predict_one_vs_all_full(transformed_images, betas, alphas)
        predicted_training_labels = helpers.predict_one_vs_all_full(transformed_training_images, betas, alphas)

        #calc test errors
        error = genHelpers.classwise_error_rate(predicted_labels, test_labels)
        total_error = genHelpers.error_rate(predicted_labels, test_labels)
        #calc training errors
        error2 = genHelpers.classwise_error_rate(predicted_training_labels, labels)
        total_error2 = genHelpers.error_rate(predicted_training_labels, labels)
        #print results
        print("Stats for {} non-linearity for test data: ".format(activationFunction.__name__))
        print("The total error for the test data was: {}".format(total_error))
        for num, error in error.items():
            print("Error rate for class {}: {}".format(num, error))

        print("Stats for {} non-linearity for training data: ".format(activationFunction.__name__))
        print("The total error for the training data was: {}".format(total_error2))
        for num, error in error2.items():
            print("Error rate for class {}: {}".format(num, error))


#problem 2-2 for the training data
if (problem2_2):
    #initialize variables
    L_values = range(100, 1501, 50)
    num_classes = 10  # For MNIST, this would be 10 for digits 0-9
    betas = []
    alphas = []
    error_rates = []
    activationFunctions = [genHelpers.sigmoid, genHelpers.relu, genHelpers.sinusoidal, genHelpers.identity]

    for activationFunction in activationFunctions:
        error_rates.clear()
        for L in L_values:
            betas.clear()
            alphas.clear()
            #apply feature mapping with current L value
            transformed_training_images, W, b = genHelpers.randomized_feature_mapping(images, L=L, g=activationFunction)
            #train classifiers
            for k in range(num_classes):
                # Label data for class k vs all others
                binary_labels = helpers.labelBinaryData(k, labels)

                # Train a binary classifier for this class
                beta, alpha = helpers.train_binary_classifier(transformed_training_images, binary_labels)

                # Store the classifier's parameters
                betas.append(beta)
                alphas.append(alpha)
            #get predictions and total errors
            predicted = helpers.predict_one_vs_all_full(transformed_training_images, betas, alphas)
            total_error = genHelpers.error_rate(predicted, labels)
            error_rates.append(total_error)
            print(L)
            print(error_rates)
        #plot results
        plt.figure(figsize=(10, 6))
        plt.plot(L_values, error_rates, marker='o')
        plt.xlabel("Number of Random Features (L)")
        plt.ylabel("Error Rate")
        plt.title("OneVsAll Error Rate vs L for {} function on Training Data".format(activationFunction.__name__))
        plt.grid()
        plt.show()

#problem 2-2 for the test data
if (problem2_2Test):
    #initialize variables
    L_values = range(100, 1501, 50)
    num_classes = 10  # For MNIST, this would be 10 for digits 0-9
    betas = []
    alphas = []
    error_rates = []
    activationFunctions = [genHelpers.sigmoid, genHelpers.relu, genHelpers.sinusoidal, genHelpers.identity]

    for activationFunction in activationFunctions:
        error_rates.clear()
        for L in L_values:
            betas.clear()
            alphas.clear()
            #apply feature mapping to training images
            transformed_images, W, b = genHelpers.randomized_feature_mapping(images, L, g=activationFunction)
            #apply identical feature mapping to test images
            transformed_test_images = genHelpers.randomized_feature_mapping_with_params(test_images, W, b , g=activationFunction)
            #train classifiers
            for k in range(num_classes):
                # Label data for class k vs all others
                binary_labels = helpers.labelBinaryData(k, labels)

                # Train a binary classifier for this class
                beta, alpha = helpers.train_binary_classifier(transformed_images, binary_labels)

                # Store the classifier's parameters
                betas.append(beta)
                alphas.append(alpha)

            #get predictions for test images
            predicted = helpers.predict_one_vs_all_full(transformed_test_images, betas, alphas)
            total_error = genHelpers.error_rate(predicted, test_labels)
            error_rates.append(total_error)
            print(L)
            print(error_rates)
        #plot results
        plt.figure(figsize=(10, 6))
        plt.plot(L_values, error_rates, marker='o')
        plt.xlabel("Number of Random Features (L)")
        plt.ylabel("Error Rate")
        plt.title("OneVsAll Error Rate vs L for {} function for Test Data".format(activationFunction.__name__))
        plt.grid()
        plt.show()
