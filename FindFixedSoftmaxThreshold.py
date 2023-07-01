import os
import argparse
from argparse import ArgumentParser
import torch
import pickle
from Test import RunTestMethod
from Test100 import RunTestMethod100

def main():
    # Define parameters
    cuda = torch.cuda.is_available()

    modelList = []
    masterTreeList = []
    runList = []

    if not args.use_CIFAR100:
        # Load models
        modelNameList = ["shufflenetv2_x2_0", "repvgg_a2", "vgg13_bn", "resnet44", "vgg16_bn", "mobilenetv2_x1_4", "resnet20", "repvgg_a0", "vgg11_bn"]

        for modelName in modelNameList:
            modelPath = os.path.join(args.weights_dir, modelName + ".pt")
            model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_" + modelName, pretrained=False)
            model.load_state_dict(torch.load(modelPath))
            model.eval()
            if cuda:
                model.cuda()
            modelList.append(model)

        for modelName in modelNameList:
            treeList = []
            for i in range(10):
                filename = os.path.join(args.trees_dir, modelName + "_" + str(i) + "_model.sav")
                treeModel = pickle.load(open(filename, 'rb'))
                treeList.append(treeModel)
            masterTreeList.append(treeList)
            
        runList = [
                "Cifar10_validation"
                ]
    else:
        modelNameList = ["shufflenetv2_x2_0", "repvgg_a2", "vgg13_bn"]

        for modelName in modelNameList:
            modelPath = os.path.join(args.weights_dir, modelName + ".pt")
            model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_" + modelName, pretrained=False)
            model.load_state_dict(torch.load(modelPath))
            model.eval()
            if cuda:
                model.cuda()
            modelList.append(model)

        for modelName in modelNameList:
            treeList = []
            for i in range(100):
                filename = os.path.join(args.trees_dir, str(args.top_n), modelName + "_" + str(i) + "_model.sav")
                treeModel = pickle.load(open(filename, 'rb'))
                treeList.append(treeModel)
            masterTreeList.append(treeList)

        runList = [
                "Cifar100_validation"
                ]
        
    numberOfPredictions = args.top_n # Top N Accuracy
    numberOfModels = args.no_models # No of models to use in ensemble

    curThreshold = 1
    bestThreshold = -1
    bestSoftmax = 2
    swingFactor = 0.1
    stepFactor = 0.01

    while curThreshold >= 0:
        tprList = []
        if not args.use_CIFAR100:
            tprList = RunTestMethod(modelList, masterTreeList, runList, args, True, curThreshold)
        else:
            tprList = RunTestMethod100(modelList, masterTreeList, runList, args, True, curThreshold)
        
        if tprList[0] > args.true_positive_rate_value:
            bestThreshold = curThreshold
            bestSoftmax = tprList[0]
            swingFactor -= stepFactor
            curThreshold += swingFactor
            break
        else:
            curThreshold -= swingFactor
            bestSoftmax = tprList[0]
            
    while True and curThreshold >= 0:
        tprList = []
        if not args.use_CIFAR100:
            tprList = RunTestMethod(modelList, masterTreeList, runList, args, True, curThreshold)
        else:
            tprList = RunTestMethod100(modelList, masterTreeList, runList, args, True, curThreshold)
        
        if tprList[0] > args.true_positive_rate_value:
            if tprList[0] < bestSoftmax:
                bestThreshold = curThreshold
                bestSoftmax = tprList[0]
            swingFactor -= stepFactor
            curThreshold += swingFactor
            
        else:
            swingFactor -= stepFactor
            curThreshold -= swingFactor
        
        if swingFactor < 0 or curThreshold > 1 or curThreshold < 0:
            if bestThreshold == -1:
                bestThreshold = curThreshold
                bestSoftmax = tprList[0]
            break

    if bestThreshold < 1 and bestThreshold > 0 and bestSoftmax > args.true_positive_rate_value:
        print("Minimum softmax threshold to reach {}% true positive rate is {} with {} softmax score".format(args.true_positive_rate_value * 100, round(bestThreshold, 2), round(bestSoftmax, 2)))
    else:
        print("{}% true positive rate is not viable".format(args.true_positive_rate_value * 100))
        print("The minimum softmax threshold tested is {} with {} softmax score".format(max(round(bestThreshold, 2), 0), round(bestSoftmax, 2)))

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
    parser.add_argument("--weights_dir", type=str, default=root_dir+"models/cifar10/")
    parser.add_argument("--trees_dir", type=str, default=root_dir+"models/trees/cifar10/")
    parser.add_argument("--ood_dir", type=str, default=root_dir+"data/")
    parser.add_argument("--y_score_dir", type=str, default=root_dir+"data/modelrunresult/cifar10/")
    parser.add_argument("--top_n", type=int, default=1, choices=[1,2,3,4,5])
    parser.add_argument("--no_models", type=int, default=3, choices=[1,3,5,7,9])
    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--use_CIFAR100', dest='use_CIFAR100', action='store_true')
    feature_parser.add_argument('--no_use_CIFAR100', dest='use_CIFAR100', action='store_false')
    parser.set_defaults(use_CIFAR100=False)
    parser.add_argument("--true_positive_rate_value", type=restricted_float, default=0.95)

    args = parser.parse_args()
    main()