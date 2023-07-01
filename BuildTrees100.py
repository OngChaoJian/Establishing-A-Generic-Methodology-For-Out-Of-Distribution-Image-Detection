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
    modelNameList = ["shufflenetv2_x2_0", "repvgg_a2", "vgg13_bn"]
    cuda = torch.cuda.is_available()
    normMean = [0.507, 0.4865, 0.4409]
    normStd = [0.2673, 0.2564, 0.2761]
    normTransform = transforms.Normalize(normMean, normStd)
    testTransformer = transforms.Compose([
        transforms.ToTensor(),
        normTransform
    ])

    for modelName in modelNameList:
        print("=======" + modelName + "=======")

        modelPath = os.path.join(args.weights_dir, modelName + ".pt")
        model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_" + modelName, pretrained=False)
        model.load_state_dict(torch.load(modelPath))
        model.eval()
        if cuda:
            model.cuda()

        with open(os.path.join(args.cifar_dir, "validation100_indices"), 'rb') as f:
            indices = pickle.load(f)
        testSampler = SubsetRandomSampler(indices)

        testLoader = DataLoader(
                dset.CIFAR100(root=args.cifar_dir, train=False, download=True, transform=testTransformer),
                    batch_size=1, sampler=testSampler, shuffle=False)

        indicesList = []
        targetList = []
        allXList = []

        for data, target in testLoader:
            if cuda:
                data, target = data.cuda(), target.cuda()

            output = model(data)
            output_np = output.cpu().detach().numpy()[0]
            maxIndex = np.argmax(output_np)
            indices = np.argsort(output_np)[::-1][:args.top_n]
            softmax = torch.nn.Softmax(dim=1)
            soften_output = softmax(output).detach().cpu().numpy()[0]
            x = np.sum(soften_output[indices])

            indicesList.append(indices)
            targetList.append(target.item())
            allXList.append(x)

        indicesNp = np.array(indicesList)
        targetNp = np.array(targetList)
        allXNp = np.array(allXList)

        for i in range(100):
            classNo = i
            print("Building {} class {} trees".format(modelName, classNo))

            filteredIndicesIndex = np.where(np.any(indicesNp == classNo, axis = 1))[0]
            filteredTargetList = targetNp[filteredIndicesIndex]
            filteredXList = allXNp[filteredIndicesIndex]
            xList = []
            yList = []

            for j in range(len(filteredIndicesIndex)):
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

            filename = os.path.join(args.trees_dir, str(args.top_n), modelName + '_' + str(classNo) + '_model.sav')
            pickle.dump(modelTree, open(filename, 'wb'))

            print("Complete building {} class {} trees".format(modelName, classNo))

    print("Complete building all trees")

if __name__ == "__main__":
    parser = ArgumentParser()

    root_dir = "/storage/"
    
    parser.add_argument("--cifar_dir", type=str, default=root_dir+"data/cifar/")
    parser.add_argument("--weights_dir", type=str, default=root_dir+"models/cifar100/")
    parser.add_argument("--trees_dir", type=str, default=root_dir+"models/trees/cifar100/")
    parser.add_argument("--top_n", type=int, default=1, choices=[1,2,3,4,5])

    args = parser.parse_args()
    main(args)