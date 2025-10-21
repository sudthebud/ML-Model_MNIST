###########
# IMPORTS #
###########
from os import path

import numpy as np
import matplotlib.pyplot as plt

from ML_Model_General_Purpose_SudTheBud import *



##########
# CONSTS #
##########
DATA_DIRPATH = "data"
RESULTS_DIRPATH = "results"
MODEL_NAME = "mnist_model"



##########
# SCRIPT #
##########

# Get data (should be shape of (n samples) x (1 output + 784 features))
# 784 features comes from the 28 x 28 = 784 pixels
print("Retrieving data...")

def retrieve_data(csvName):
    data = np.genfromtxt(path.join(DATA_DIRPATH, csvName), delimiter=",", dtype=int)

    input = data[:, 1:]
    output_raw = data[:, 0:1]

    output = np.zeros((output_raw.shape[0], 10))
    output[range(len(output_raw)), np.reshape(output_raw, output_raw.shape[0])] = 1 # Thanks to @myz540 on StackOverflow for this

    return input, output

if not path.isfile(path.join(DATA_DIRPATH, "mnist_np.npz")):
    trainData_input, trainData_output = retrieve_data("mnist_train.csv")
    testData_input, testData_output = retrieve_data("mnist_test.csv")

    np.savez_compressed(path.join(DATA_DIRPATH, "mnist_np.npz"), trainData_input = trainData_input, trainData_output = trainData_output, testData_input = testData_input, testData_output = testData_output)
else:
    mnistData_np = np.load(path.join(DATA_DIRPATH, "mnist_np.npz"))

    trainData_input, trainData_output = mnistData_np['trainData_input'], mnistData_np['trainData_output']
    testData_input, testData_output = mnistData_np['testData_input'], mnistData_np['testData_output']


# Create or load model
if path.isfile(path.join(RESULTS_DIRPATH, MODEL_NAME + ".sudml")): 
    print("\nLoading model...")

    model = load_model(path.join(RESULTS_DIRPATH, MODEL_NAME + ".sudml"))
else: 
    print("\nCreating model...")

    model = Model(
        numInputNodes=784,
        numHiddenLayerNodes=[128],
        numOutputNodes=10,
        normalize=True,
        activationFunc=[ActivationFunc.RELU, ActivationFunc.SOFTMAX],
        costFunc=CostFunc.CATEGORICAL_CROSS_ENTROPY,
        weightInitFunc=WeightInitFunc.HE_NORMAL,
        biasInitFunc=BiasInitFunc.SMALL_ALPHA
    )


    # Train model
    print("\nTraining model...")

    costs, _ = model.train(trainData_input, 
                        trainData_output,
                        batchSize = 64,
                        epochs = 100,
                        learningRateSchedulerFunc=LearningRateSchedulerFunc.EXPONENTIAL_DECAY,
                        learningRate=0.001,
                        epochPrintInterval=20)
    costs = 1/costs.shape[1] * np.sum(costs, axis=1)
    epochs = np.arange(1, len(costs) + 1)

    plt.plot(epochs, costs)
    plt.xlim(1, len(epochs) + 1)
    plt.ylim(0, max(np.max(costs), 1))
    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    plt.title("Model Training Performance by Epoch")
    plt.savefig(path.join(RESULTS_DIRPATH, "model_performance.png"))


# Test model
print ("\nTesting model...")

predictions = model.predict(testData_input)

predictedClass = np.argmax(predictions, axis = 1)
trueClass = np.argmax(testData_output, axis = 1)

accuracy = np.sum(np.where(predictedClass == trueClass, 1, 0)) / trueClass.shape[0]
print(accuracy)


# Save model
if not path.isfile(path.join(RESULTS_DIRPATH, MODEL_NAME + ".sudml")): 
    print("\nSaving model...")

    model.save_model(path.join(RESULTS_DIRPATH, MODEL_NAME))