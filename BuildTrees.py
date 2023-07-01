import os
import argparse
from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms
import torchvision.datasets as dset
from sklearn import tree
import pickle
import numpy as np

def main(args):
    modelNameList = ["shufflenetv2_x2_0", "repvgg_a2", "vgg13_bn", "resnet44", "vgg16_bn", "mobilenetv2_x1_4", "resnet20", "repvgg_a0", "vgg11_bn"]
    cuda = torch.cuda.is_available()
    normMean = [0.49139968, 0.48215827, 0.44653124]
    normStd = [0.24703233, 0.24348505, 0.26158768]
    normTransform = transforms.Normalize(normMean, normStd)
    testTransformer = transforms.Compose([
        transforms.ToTensor(),
        normTransform
    ])

    for modelName in modelNameList:
        print("=======" + modelName + "=======")
        
        modelPath = os.path.join(args.weights_dir, modelName + ".pt")
        model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_" + modelName, pretrained=False)
        model.load_state_dict(torch.load(modelPath))
        model.eval()
        if cuda:
            model.cuda()

        with open(os.path.join(args.cifar_dir, "validation_indices"), 'rb') as f:
            indices = pickle.load(f)
        testSampler = SubsetRandomSampler(indices)

        testLoader = DataLoader(
                dset.CIFAR10(root=args.cifar_dir, train=False, download=True, transform=testTransformer),
                    batch_size=1, sampler=testSampler, shuffle=False)

        predList = []
        targetList = []
        allXList = []

        for data, target in testLoader:
            if cuda:
                data, target = data.cuda(), target.cuda()

            output = model(data)
            softmax = torch.nn.Softmax(dim=1)
            softenOutput = softmax(output).detach().cpu().numpy()
            pred = softenOutput.argmax(axis=1)[0]
            x = softenOutput[0][pred]

            predList.append([pred])
            targetList.append(target.item())
            allXList.append(x)

        predNp = np.array(predList)
        targetNp = np.array(targetList)
        allXNp = np.array(allXList)

        for i in range(10):
            classNo = i
            print("Building {} class {} tree".format(modelName, classNo))

            filteredPredIndex = np.where(np.any(predNp == classNo, axis = 1))[0]
            filteredTargetList = targetNp[filteredPredIndex]
            filteredXList = allXNp[filteredPredIndex]
            xList = []
            yList = []

            for j in range(len(filteredPredIndex)):
                if filteredTargetList[j] != classNo:
                    # False Positive
                    xList.append([filteredXList[j]])
                    yList.append(0)
                elif filteredTargetList[j] == classNo:
                    # True Positive
                    xList.append([filteredXList[j]])
                    yList.append(1)

            x0 = np.array(xList)[np.where(np.array(yList) == 0)]
            y0 = np.array(yList)[np.where(np.array(yList) == 0)]
            x1 = np.array(xList)[np.where(np.array(yList) == 1)]
            y1 = np.array(yList)[np.where(np.array(yList) == 1)]
            class0weight = 1
            class1weight = 1
            if len(x1) > 0:
                class0weight = len(x1) / (len(x0) + len(x1))
                class1weight = len(x0) / (len(x0) + len(x1))
            else:
                xList = [[0], [1]]
                yList = [[0], [1]]

            modelTree = tree.DecisionTreeClassifier(max_depth=1, class_weight={0:class0weight, 1:class1weight})
            modelTree.fit(xList, yList)
            
            filename = os.path.join(args.trees_dir, modelName + '_' + str(classNo) + '_model.sav')
            pickle.dump(modelTree, open(filename, 'wb'))

            print("Complete building {} class {} trees".format(modelName, classNo))
    
    print("Complete building all trees")

if __name__ == "__main__":
    parser = ArgumentParser()

    root_dir = "/storage/"
    
    parser.add_argument("--cifar_dir", type=str, default=root_dir+"data/cifar/")
    parser.add_argument("--weights_dir", type=str, default=root_dir+"models/cifar10/")
    parser.add_argument("--trees_dir", type=str, default=root_dir+"models/trees/cifar10/")

    args = parser.parse_args()
    main(args)