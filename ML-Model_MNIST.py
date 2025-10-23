###########
# IMPORTS #
###########
import os

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from ML_Model_General_Purpose_SudTheBud import *



##########
# CONSTS AND FLAGS #
##########
DATA_DIRPATH = "data"
RESULTS_DIRPATH = "results"
PREDICTION_INPUTS_DIRPATH = "prediction_inputs"

MODEL_NAME = "mnist_model"
RETRAIN_MODEL = False



##########
# SCRIPT #
##########

# Get data (should be shape of (n samples) x (1 output + 784 features))
# 784 features comes from the 28 x 28 = 784 pixels
print("Retrieving data...")


def retrieve_data(csvName):
    data = np.genfromtxt(os.path.join(DATA_DIRPATH, csvName), delimiter=",", dtype=int)

    input = data[:, 1:]
    output_raw = data[:, 0:1]

    output = np.zeros((output_raw.shape[0], 10))
    output[range(len(output_raw)), np.reshape(output_raw, output_raw.shape[0])] = 1 # Thanks to @myz540 on StackOverflow for this

    return input, output


if not os.path.isfile(os.path.join(DATA_DIRPATH, "mnist_np.npz")):
    trainData_input, trainData_output = retrieve_data("mnist_train.csv")
    testData_input, testData_output = retrieve_data("mnist_test.csv")

    np.savez_compressed(os.path.join(DATA_DIRPATH, "mnist_np.npz"), trainData_input = trainData_input, trainData_output = trainData_output, testData_input = testData_input, testData_output = testData_output)
else:
    mnistData_np = np.load(os.path.join(DATA_DIRPATH, "mnist_np.npz"))

    trainData_input, trainData_output = mnistData_np['trainData_input'], mnistData_np['trainData_output']
    testData_input, testData_output = mnistData_np['testData_input'], mnistData_np['testData_output']


# Create or load model
if not RETRAIN_MODEL and os.path.isfile(os.path.join(RESULTS_DIRPATH, MODEL_NAME + ".sudml")): 
    print("\nLoading model...")

    model = load_model(os.path.join(RESULTS_DIRPATH, MODEL_NAME + ".sudml"))
else: 
    print("\nCreating model...")

    model = Model(
        numInputNodes=784,
        numHiddenLayerNodes=[512],
        numOutputNodes=10,
        normalize=True,
        activationFunc=[ActivationFunc.RELU, ActivationFunc.SOFTMAX],
        costFunc=CostFunc.CATEGORICAL_CROSS_ENTROPY,
        weightInitFunc=WeightInitFunc.HE_NORMAL,
        biasInitFunc=BiasInitFunc.SMALL_ALPHA
    )


    # Train model
    print("\nTraining model...")


    costs, epochOutputs = model.train(trainData_input, 
                                      trainData_output,
                                      batchSize = 64,
                                      epochs = 100,
                                      shuffleDataset = True,
                                      learningRateSchedulerFunc=LearningRateSchedulerFunc.EXPONENTIAL_DECAY,
                                      learningRate=0.001,
                                      epochPrintInterval=20,
                                      returnOutput=True)
    
    costs = 1/costs.shape[1] * np.sum(costs, axis=1)
    epochAccuracies = np.empty((epochOutputs.shape[0], 1))
    for i in range(epochOutputs.shape[0]):
        accuracy, _, _, _, _ = classification_metrics(epochOutputs[i], trainData_output)
        epochAccuracies[i] = accuracy
    epochs = np.arange(1, epochOutputs.shape[0] + 1)


    plt.plot(epochs, costs, label = "Cost")
    plt.plot(epochs, epochAccuracies, label = "Accuracy")
    plt.xlim(1, len(epochs) + 1)
    plt.ylim(0, max(np.max(costs), np.max(epochAccuracies), 1))
    plt.xlabel("Epoch")
    plt.ylabel("Cost / Accuracy")
    plt.legend(loc = "lower left")
    plt.title("Model Training Performance by Epoch")
    plt.savefig(os.path.join(RESULTS_DIRPATH, "model_performance.png"))


# Test model
print ("\nTesting model...")


predictions = model.predict(testData_input)

accuracy, recall, fpr, precision, f1 = classification_metrics(predictions, testData_output)
accuracy = np.round(accuracy, 4)
recall = np.round(np.mean(recall), 4)
fpr = np.round(np.mean(fpr), 4)
precision = np.round(np.mean(precision), 4)
f1 = np.round(np.mean(f1), 4)
print(f"Accuracy:\t\t{accuracy}\nRecall:\t\t\t{recall}\nFalse Positive Rate:\t{fpr}\nPrecision:\t\t{precision}\nF1 Score:\t\t{f1}")


confusionMatrix = np.zeros((predictions.shape[1], testData_output.shape[1]))
for trueVal, predVal in zip(np.argmax(testData_output, axis = 1), np.argmax(predictions, axis = 1)):
    confusionMatrix[predVal, trueVal] += 1

plt.clf()
plt.figure(figsize = (8, 8))
plt.imshow(confusionMatrix)
for (i, j), val in np.ndenumerate(confusionMatrix): # Thanks to @Joe Kington on StackOverflow for this (https://stackoverflow.com/questions/20998083/show-the-values-in-the-grid-using-matplotlib)
    plt.text(j, i, f"{int(val)}", ha='center', va='center', bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
plt.xticks(np.arange(testData_output.shape[1]))
plt.yticks(np.arange(predictions.shape[1]))
plt.xlabel("True Digit")
plt.ylabel("Predicted Digit")
plt.title("Model Test Data Confusion Matrix")
plt.savefig(os.path.join(RESULTS_DIRPATH, "model_confusion_matrix.png"))


# Predict
if len(os.listdir(PREDICTION_INPUTS_DIRPATH)) > 0:
    print("\nPredicting model on test images...")

    for predictionInput_File in os.listdir(PREDICTION_INPUTS_DIRPATH):
        predictionInput = os.path.join(PREDICTION_INPUTS_DIRPATH, predictionInput_File)
        if not (os.path.isfile(predictionInput) and os.path.splitext(predictionInput)[1] == ".png"): continue

        pixels = np.array(Image.open(predictionInput))
        pixels = np.mean(pixels, axis = 2)
        pixels = np.ravel(pixels)
        
        print(f"{predictionInput_File}: Predicted as \"{np.argmax(model.predict(pixels))}\"")


# Save model
if RETRAIN_MODEL: 
    print("\nSaving model...")

    model.save_model(os.path.join(RESULTS_DIRPATH, MODEL_NAME))