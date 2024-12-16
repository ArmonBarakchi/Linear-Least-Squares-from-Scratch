import helpersOneVsOne as helpers
import numpy as np
from scipy.io import loadmat
import genHelpers
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# which part to run
withFeatureMapping = 1
problem1 = 0
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



if (problem1):
    # Train all one-versus-one classifiers
    binary_classifiers = helpers.train_ovo_classifiers(images, labels, num_classes=10)

    #get predictons for test and training images
    predicted_test = helpers.predict_ovo(test_images, binary_classifiers, num_classes=10)
    predicted_training = helpers.predict_ovo(images, binary_classifiers, num_classes=10)

    #Plot confusion matrices
    true_test_labels = test_labels.flatten()
    true_training_labels = labels.flatten()
    genHelpers.plot_confusion_matrix(true_test_labels, predicted_test, num_classes=10)
    genHelpers.plot_confusion_matrix(true_training_labels, predicted_training, num_classes=10)

    #calc errors
    error_training = genHelpers.classwise_error_rate(predicted_training, labels)
    total_training_error = genHelpers.error_rate(predicted_training, labels)
    error_test = genHelpers.classwise_error_rate(predicted_test, test_labels)
    total_test_error = genHelpers.error_rate(predicted_test, test_labels)

    #print errors
    print(error_training)
    print(total_training_error)
    print(error_test)
    print(total_test_error)

if (withFeatureMapping):
    activationFunctions = [genHelpers.identity, genHelpers.relu,
                           genHelpers.sigmoid, genHelpers.sinusoidal]

    for activationFunction in activationFunctions:
        #apply feature mapping to training and test images
        transformed_training_images, W, b = genHelpers.randomized_feature_mapping(images, L=1000, g=activationFunction)
        transformed_images = genHelpers.randomized_feature_mapping_with_params(test_images, W, b, g=activationFunction)
        #train classifiers
        binary_classifiers = helpers.train_ovo_classifiers(transformed_training_images, labels, num_classes=10)


        # Predict using the one-versus-one classifiers on test data
        predicted_labels = helpers.predict_ovo(transformed_images, binary_classifiers, num_classes=10)
        predicted_training_labels = helpers.predict_ovo(transformed_training_images, binary_classifiers, num_classes=10)
        #calc errors
        error = genHelpers.classwise_error_rate(predicted_labels, test_labels)
        total_error = genHelpers.error_rate(predicted_labels, test_labels)
        error2 = genHelpers.classwise_error_rate(predicted_training_labels, labels)
        total_error2 = genHelpers.error_rate(predicted_training_labels, labels)

        print("Stats for {} non-linearity for test data: ".format(activationFunction.__name__))
        print("The total error for the test data was: {}".format(total_error))
        for num, error in error.items():
            print("Error rate for class {}: {}".format(num, error))

        print("Stats for {} non-linearity for training data: ".format(activationFunction.__name__))
        print("The total error for the training data was: {}".format(total_error2))
        for num, error in error2.items():
            print("Error rate for class {}: {}".format(num, error))

if (problem2_2):
    #initialize variables for graph
    L_values = range(100, 1501, 50)
    error_rates = []
    activationFunctions = [genHelpers.sigmoid, genHelpers.identity, genHelpers.relu, genHelpers.sinusoidal]
    for activationFunction in activationFunctions:
        print(activationFunction.__name__)
        error_rates.clear()
        #loop through L values
        for L in L_values:
            #create random features and train classifiers on them
            transformed_training_images, W, b = genHelpers.randomized_feature_mapping(images, L, g=activationFunction)

            # Train classifiers
            binary_classifiers = helpers.train_ovo_classifiers(transformed_training_images, labels, num_classes=10)
            #predict labels for training data
            predicted_labels = helpers.predict_ovo(transformed_training_images, binary_classifiers, num_classes=10)

            #calc error and reset for next loop
            total_error = genHelpers.error_rate(predicted_labels, labels)
            error_rates.append(total_error)
            predicted_labels = np.array([])
            binary_classifiers = np.array([])

            print(L)
            print(error_rates)

        #plot L vs Error Rates
        plt.figure(figsize=(10, 6))
        plt.plot(L_values, error_rates, marker='o')
        plt.xlabel("Number of Random Features (L)")
        plt.ylabel("Error Rate")
        plt.title("OneVsOne Error Rate vs L for {} function on Training Data".format(activationFunction.__name__))
        plt.grid()
        plt.show()

if (problem2_2Test):
    # initialize variables for graph
    L_values = range(100, 1501, 50)
    error_rates = []
    error_rates.clear()
    activationFunctions = [genHelpers.sigmoid, genHelpers.identity, genHelpers.relu, genHelpers.sinusoidal]
    for activationFunction in activationFunctions:
        print(activationFunction.__name__)
        error_rates.clear()
        # loop through L values
        for L in L_values:
            # create random features and train classifiers on them
            transformed_images, W, b = genHelpers.randomized_feature_mapping(images, L, g=activationFunction)
            binary_classifiers = helpers.train_ovo_classifiers(transformed_images, labels, num_classes=10)

            transformed_test_images = genHelpers.randomized_feature_mapping_with_params(test_images, W, b, g=activationFunction)        # predict labels for test data
            predicted_labels = helpers.predict_ovo(transformed_test_images, binary_classifiers, num_classes=10)

            # calc error and reset for next loop
            total_error = genHelpers.error_rate(predicted_labels, test_labels)
            error_rates.append(total_error)

            print(L)
            print(error_rates)

        # plot L vs Error Rates
        plt.figure(figsize=(10, 6))
        plt.plot(L_values, error_rates, marker='o')
        plt.xlabel("Number of Random Features (L)")
        plt.ylabel("Error Rate")
        plt.title("OneVsOne Error Rate vs L for {} function on Test Data".format(activationFunction.__name__))
        plt.grid()
        plt.show()

