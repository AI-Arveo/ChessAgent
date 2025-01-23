import argparse
import os
import time
from math import floor
from typing import Type

from matplotlib import pyplot as pt
import torch as t
import torch.cuda
import torch.nn as nn
import torch.optim
from torch.cuda.amp import autocast
from torch.optim import Adam, Optimizer

import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

from chess_project.project.chess_neuralNetwork.neural_network import NeuralNetwork, FullPerspectiveHeuristic, Heuristic
from chess_project.project.chess_neuralNetwork.parser import Loader, DataParser


class Criterion:
    def __call__(self, results: t.Tensor, targets: t.Tensor) -> t.Tensor:
        pass

class EarlyStopper:
    def __init__(self, patience, delta):
        self.minLoss = float('inf')
        self.counter = 0
        self.patience = patience
        self.delta = delta

    def early_stop(self, loss: float):
        if loss < self.minLoss:
            self.counter = 0
            self.minLoss = loss
        elif loss > self.minLoss + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def train(model: nn.Module, optimizer: Optimizer, criterion: Criterion, dataLoader: Loader) -> float:
    _runningLoss = 0
    _reportLoss = 0
    reportingPeriod = 1000
    batch = 0
    model.train()
    for j, data in enumerate(dataLoader):
        inputs, targets = data
        optimizer.zero_grad()

        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        sumLoss = loss.item()
        _runningLoss += sumLoss
        _reportLoss += sumLoss

        # Report progress
        if (j + 1) % floor(reportingPeriod / 20) == 0:
            percentage = floor((j % reportingPeriod) / reportingPeriod * 100)
            print("Training: {", "=" * percentage,
                  " " * (100 - percentage), "}", end='\r')

        # Report loss
        if (j + 1) % reportingPeriod == 0:
            batch += 1
            print(" " * 130, end="\r")
            batchLoss = _reportLoss / reportingPeriod
            _reportLoss = 0
            print(
                f"{batch}: Average loss over last {reportingPeriod} batches: {round(batchLoss, 5)} ")
    print(" " * 130, end="\r")
    _averageTrainingLoss = _runningLoss / max(1, (len(dataLoader) // batchSize))
    return round(_averageTrainingLoss, 5)


def validate(model: nn.Module, criterion: Criterion, validationDataLoader: Loader) -> float:
    _validationLoss = 0
    model.eval()
    with torch.no_grad():
        for j, data in enumerate(validationDataLoader):
            inputs, targets = data

            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            _validationLoss += loss.item()
            if j % 20:
                percentage = floor(
                    j / (len(validationDataLoader) // batchSize) * 100)
                print("Evaluating: {", "=" * percentage,
                      " " * (100 - percentage), "}", end='\r')

    return round(_validationLoss / (len(validationDataLoader) // batchSize), 5)


def collectData(folder_path: str, heuristic: Type[Heuristic], batchSize: int) -> Loader:
    files = os.listdir(folder_path)
    dataParsers: [DataParser] = []
    for file in files:
        if not file.split(".")[-1] == "cache":
            dataParser = DataParser(folder_path + "/" + file)
            dataParser.parse()
            dataParsers.append(dataParser)
    return Loader(dataParsers, batchSize, heuristic)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="DataTrainer", description="Train our model using datasets located in /project/data/raw")
    parser.add_argument('-l', '--learning-rate', default=0.0001, type=float)
    parser.add_argument('-b', '--batch-size', default=32, type=int)
    parser.add_argument('-e', '--epochs', default=5, type=int)
    parser.add_argument('--preload', default=None, type=str,
                        help="Continue training on a previous model, value is location to model")
    args = parser.parse_args()
    learningRate = args.learning_rate
    batchSize = args.batch_size
    numberOfEpochs = args.epochs
    preload = args.preload

    print(
        f"The learning parameters are:\n- Learning rate:\t{learningRate}\n- batchSize:\t\t{batchSize}\n- number of epochs:\t{numberOfEpochs}\n- preload:\t\t{preload}\n")

    # TODO use the dataset to train a NeuralNetworkHeuristic, afterwards save it.

    """
    vvv Insert model here vvv
    """
     #model: nn.Module = CanCaptureHeuristicBit(128, 64, 0, nn.LeakyReLU(), 0.3)
    model: Heuristic = FullPerspectiveHeuristic()
    earlyStopper: EarlyStopper = EarlyStopper(1, 0.005)
    if preload:
        model = torch.load(preload)
        model.train()
    if t.cuda.is_available():
        # print("Cuda was available, transferring data to GPU")
        model.to(device='cuda')

    optimizer = Adam(model.parameters(), lr=learningRate)
    criterion: Criterion = nn.MSELoss()


    # Specify the folder path you want to get filepaths from
    trainingFolderPath = r"D:\PythonProjects\ChessAgent\LichessEliteDatabase"
    validationFolderPath = r"D:\PythonProjects\ChessAgent\LichessEliteDatabase"

    trainDataLoader = collectData(
        trainingFolderPath, model.__class__, batchSize)
    validationDataLoader = collectData(
        validationFolderPath, model.__class__, batchSize)
    print(
        f"Total amount of batches:\n- training:\t{len(trainDataLoader)} datapoints -> {len(trainDataLoader) // batchSize} batches\n- validation:\t{len(validationDataLoader)} datapoints -> {len(validationDataLoader) // batchSize} batches\n")

    print("Start training: \n" + "=" * 100, '\n')
    startTime = time.perf_counter()

    trainingLossValues = []
    validationLossValues = []

    for i in range(numberOfEpochs):
        print(f"Starting epoch {i + 1}:\n" + "-" * 100)
        averageTrainingLoss = train(model, optimizer, criterion, trainDataLoader)
        print(
            f"Finished training for epoch {i + 1} with average training loss: {averageTrainingLoss}")
        trainingLossValues.append(averageTrainingLoss)

        averageValidationLoss = validate(model, criterion, validationDataLoader)
        validationLossValues.append(averageValidationLoss)
        print(
            f"Finished validating for epoch {i + 1} with average validation loss: {averageValidationLoss}" + " " * 50 + "\n")
        torch.save(
            model, f"../../{model._get_name()}_{i + 1}_0,{round(averageValidationLoss * 10000)}.pth")
        if earlyStopper is not None and earlyStopper.early_stop(averageValidationLoss):
            print(
                f"Stopped early because previous loss {earlyStopper.minLoss} was lower than current loss {averageValidationLoss}")
            break

    endTime = time.perf_counter()
    seconds = endTime - startTime
    minutes = seconds // 60
    seconds = seconds - minutes * 60
    print(
        f"Time passed training: {round(minutes)} minutes {round(seconds)} seconds")

    # Plot the loss for every epoch
    pt.plot(trainingLossValues, 'r', label="training loss")
    pt.plot(validationLossValues, 'g', label="validation loss")
    pt.title("Loss over epochs")
    pt.legend(loc="upper right")

    pt.show()
    pass
