import os
import argparse
from argparse import ArgumentParser
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.autograd import Variable
import torchvision
from torchvision import transforms
import torchvision.datasets as dset
from sklearn.metrics import roc_auc_score, PrecisionRecallDisplay
from sklearn import tree
import pickle
import numpy as np

# Run through datasets
def FindCommonElements(masterList):
    common = masterList[0]
    for i in range(len(masterList) - 1):
        common = set(masterList[i]).intersection(masterList[i + 1])
    return list(common)

def RunTestMethod100(modelList, masterTreeList, runList, args, useFixedSoftmaxThreshold, fixedSoftmaxValue):
    # Define parameters
    cuda = torch.cuda.is_available()
    normMean = [0.507, 0.4865, 0.4409]
    normStd = [0.2673, 0.2564, 0.2761]
    normTransform = transforms.Normalize(normMean, normStd)
    testTransformer = transforms.Compose([
        transforms.ToTensor(),
        normTransform
    ])
    
    classifiedLowestScore = 0
    classifiedHighestScore = 1
    nonClassifiedLowestScore = 2
    nonClassifiedHighestScore = 3

    tprList = []

    for run in runList:
        print("=======" + run + "=======")

        dataName = run
        if run == "Gaussian" or run == "Uniform":
            dataName = "Imagenet"

        testLoader = None
        if run == "Cifar100_validation":
            with open(os.path.join(args.cifar_dir, "validation100_indices"), 'rb') as f:
                indices = pickle.load(f)
            testSampler = SubsetRandomSampler(indices)
            testLoader = DataLoader(dset.CIFAR100(root=args.cifar_dir, train=False, download=True, transform=testTransformer), 
                batch_size=1, sampler=testSampler, shuffle=False, num_workers=4)
        elif run == "Cifar100_test":
            with open(os.path.join(args.cifar_dir, "test100_indices"), 'rb') as f:
                indices = pickle.load(f)
            testSampler = SubsetRandomSampler(indices)
            testLoader = DataLoader(dset.CIFAR100(root=args.cifar_dir, train=False, download=True, transform=testTransformer), 
                batch_size=1, sampler=testSampler, shuffle=False, num_workers=4)
        else:
            testDataset = torchvision.datasets.ImageFolder(args.ood_dir + "/{}".format(dataName), transform=testTransformer)
            testLoader = torch.utils.data.DataLoader(testDataset, batch_size=1, shuffle=False, num_workers=4)

        count = 0
        oodCount = 0
        idCount = 0
        yAlwaysHighScoreList = []
        yAlwaysLowScoreList = []

        for j, data in enumerate(testLoader):
            if run == "Gaussian":
                images = torch.randn(1, 3, 32, 32) + 0.5
                images = torch.clamp(images, 0, 1)
                images[0][0] = (images[0][0] - 125.3/255) / (63.0/255)
                images[0][1] = (images[0][1] - 123.0/255) / (62.1/255)
                images[0][2] = (images[0][2] - 113.9/255) / (66.7/255)
            elif run == "Uniform":
                images = torch.rand(1, 3, 32, 32)
                images[0][0] = (images[0][0] - 125.3/255) / (63.0/255)
                images[0][1] = (images[0][1] - 123.0/255) / (62.1/255)
                images[0][2] = (images[0][2] - 113.9/255) / (66.7/255)
            else:
                images, _ = data

            inputs = Variable(images.cuda(), requires_grad=True)
            softmaxOutput = torch.nn.Softmax(dim=1)

            predList = []
            probList = []

            for i in range(args.no_models):
                output = modelList[i](inputs)
                outputNp = output.cpu().detach().numpy()[0]
                softenOutput = softmaxOutput(output).detach().cpu().numpy()
                pred = np.argsort(outputNp)[::-1][:args.top_n]
                predList.append(pred)
                probList.append(np.sum(softenOutput[0][pred]))


            classifiedClass = -1

            yBest = [1, -1, 1, -1]
            passCount = int(args.no_models / 2) + 1
            for i in range(args.no_models):
                validList = []
                indexList = []
                predictionList = []
                for j in range(passCount):
                    validList.append(False)
                    index = (i + j) % args.no_models
                    indexList.append(index)
                    predictionList.append(predList[index])

                commonElements = FindCommonElements(predictionList)
                for ele in commonElements:
                    totalProb = 0

                    for j in range(passCount):
                        curIndex = indexList[j]
                        totalProb += probList[curIndex]
                        if useFixedSoftmaxThreshold:
                            if probList[curIndex] > fixedSoftmaxValue: # Fix softmax threshold to reach target TPR
                                validList[j] = True
                        elif masterTreeList[curIndex][ele].predict([[probList[curIndex]]])[0] == 1:
                            validList[j] = True

                    avgScore = totalProb / passCount
                    if all(validList):
                        classifiedClass = ele
                        if yBest[classifiedLowestScore] > avgScore:
                            yBest[classifiedLowestScore] = avgScore
                        if yBest[classifiedHighestScore] < avgScore:
                            yBest[classifiedHighestScore] = avgScore
                    else:
                        if yBest[nonClassifiedLowestScore] > avgScore:
                            yBest[nonClassifiedLowestScore] = avgScore
                        if yBest[nonClassifiedHighestScore] < avgScore:
                            yBest[nonClassifiedHighestScore] = avgScore

            if classifiedClass == -1:
                oodCount += 1
                yAlwaysHighScoreList.append(yBest[nonClassifiedHighestScore])
                yAlwaysLowScoreList.append(yBest[nonClassifiedLowestScore])
            else:
                idCount += 1
                yAlwaysHighScoreList.append(yBest[classifiedHighestScore])
                yAlwaysLowScoreList.append(yBest[classifiedLowestScore])

            count += 1
            if count % 100 == 0:
                print("{} Progress: {} \ {}".format(run, count, len(testLoader)))

        allList = [yAlwaysHighScoreList, yAlwaysLowScoreList]
        filename = os.path.join(args.y_score_dir, 'noofmodels_' + str(args.no_models), run + '_lists.npy')
        np.save(filename, np.array(allList))

        print("{} OOD: {}".format(run, oodCount))
        print("{} ID: {}".format(run, idCount))

        tprList.append(idCount / len(testLoader))

    return tprList

def main(args):
    # Define parameters
    cuda = torch.cuda.is_available()
    
    # Load models
    modelNameList = ["shufflenetv2_x2_0", "repvgg_a2", "vgg13_bn"]
    modelList = []

    for modelName in modelNameList:
        modelPath = os.path.join(args.weights_dir, modelName + ".pt")
        model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_" + modelName, pretrained=False)
        model.load_state_dict(torch.load(modelPath))
        model.eval()
        if cuda:
            model.cuda()
        modelList.append(model)

    masterTreeList = []

    for modelName in modelNameList:
        treeList = []
        for i in range(100):
            filename = os.path.join(args.trees_dir, str(args.top_n), modelName + "_" + str(i) + "_model.sav")
            treeModel = pickle.load(open(filename, 'rb'))
            treeList.append(treeModel)
        masterTreeList.append(treeList)
        
    runList = [
            "Gaussian", 
            "Uniform", 
            "Imagenet", 
            "Imagenet_resize", 
            "LSUN", 
            "LSUN_resize",
            "Cifar100_test",
            "Cifar100_validation"
            ]

    RunTestMethod100(modelList, masterTreeList, runList, args, args.use_fixed_softmax_threshold, args.fixed_softmax_threshold_value)
    
    oodList = [
            "Gaussian", 
            "Uniform", 
            "Imagenet", 
            "Imagenet_resize", 
            "LSUN", 
            "LSUN_resize"
            ]

    run = "Cifar100_test"
    filename = os.path.join(args.y_score_dir, 'noofmodels_' + str(args.no_models), run + '_lists.npy')
    file = open(filename, "rb")
    testNpAllLists = np.load(file, allow_pickle=True)
    totalAuroc = 0
    totalInvAuroc = 0
    totalAupr = 0
    totalInvAupr = 0

    for ood in oodList:
        print("========" + ood + "========")
        run = ood
        filename = os.path.join(args.y_score_dir, 'noofmodels_' + str(args.no_models), run + '_lists.npy')
        file = open(filename, "rb")
        oodNpAllLists = np.load(file, allow_pickle=True)
        yAlwaysHighScoreList = np.random.choice(oodNpAllLists[0], 5000, replace=False)

        yHigh = np.concatenate((testNpAllLists[0], yAlwaysHighScoreList))
        yTrue = np.concatenate((np.full((5000), True), np.full((5000), False)))
        yTrueInverse = np.concatenate((np.full((5000), False), np.full((5000), True)))
        aurocYHigh = roc_auc_score(yTrue, yHigh)
        print("{}  auroc (ID True): {}".format(ood, aurocYHigh))

        auprY = PrecisionRecallDisplay.from_predictions(yTrue, yHigh, name="yHigh")
        print("{} average precision (ID True): {}".format(ood, auprY.average_precision))
        auprYInv = PrecisionRecallDisplay.from_predictions(yTrueInverse, yHigh * -1, name="yInvHigh")
        print("{} inverse average precision (OOD True): {}".format(ood, auprYInv.average_precision))

        totalAuroc += aurocYHigh
        totalAupr += auprY.average_precision
        totalInvAupr += auprYInv.average_precision

    print("Average auroc (ID True): {}".format(totalAuroc / len(oodList)))
    print("Average aupr (ID True): {}".format(totalAupr / len(oodList)))
    print("Average inverse aupr (OOD True): {}".format(totalInvAupr / len(oodList)))

def restricted_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]" % (x,))
    return x

if __name__ == "__main__":
    parser = ArgumentParser()

    root_dir = "/storage/"
    
    parser.add_argument("--cifar_dir", type=str, default=root_dir+"data/cifar/")
    parser.add_argument("--weights_dir", type=str, default=root_dir+"models/cifar100/")
    parser.add_argument("--trees_dir", type=str, default=root_dir+"models/trees/cifar100/")
    parser.add_argument("--ood_dir", type=str, default=root_dir+"data/")
    parser.add_argument("--y_score_dir", type=str, default=root_dir+"data/modelrunresult/cifar100/")
    parser.add_argument("--top_n", type=int, default=1, choices=[1,2,3,4,5])
    parser.add_argument("--no_models", type=int, default=3, choices=[1,3])
    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--fix_softmax_threshold', dest='use_fixed_softmax_threshold', action='store_true')
    feature_parser.add_argument('--no-fix_softmax_threshold', dest='use_fixed_softmax_threshold', action='store_false')
    parser.set_defaults(use_fixed_softmax_threshold=False)
    parser.add_argument("--fixed_softmax_threshold_value", type=restricted_float, default=0.85)

    args = parser.parse_args()
    main(args)