from __future__ import print_function, division

import torch
torch.manual_seed(42)
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms
import time
import copy
import os
import glob

import blur.models_lpf.resnet
from src._logging import get_logger
from src.dataset import TrainDataset, TestDataset
from src.configs import *
from src.snapshot import CosineAnnealingLR_with_Restart
from src.ensemble import get_base_predictions_model

logger = get_logger(LOGGER)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25):

    if not os.path.exists(LOGGER):
        os.makedirs(LOGGER)

    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        logger.info("Epoch {}/{}".format(epoch, num_epochs - 1))
        logger.info("-" * 10)
        scheduler.step()

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:

                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs.double())
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            logger.info("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), LOGGER + "/best" + ".pth")
            if phase == "val":
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    logger.info("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    logger.info("Best val Acc: {:4f}".format(best_acc))

    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def initialize_model(num_classes=13, num_channels=3):
    if BLUR:
        model_ft = blur.models_lpf.resnet.resnet18(filter_size=FILTER_SIZE, num_channels=num_channels)
        state_dict = torch.load('weights/resnet18_lpf%i.pth.tar' % FILTER_SIZE, map_location=torch.device(device))['state_dict']
    else:
        model_ft = models.resnet18(num_channels=num_channels)
        state_dict = torch.load(WEIGHTS_PATH, map_location=torch.device(device))
    if num_channels == 1:
        conv1_weight = state_dict["conv1.weight"]
        state_dict["conv1.weight"] = conv1_weight.sum(dim=1, keepdim=True)
    model_ft.load_state_dict(state_dict)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    ct = 0
    for name, child in model_ft.named_children():
        ct += 1
        if ct < NUM_FREEZE_LAYERS:
            for name2, params in child.named_parameters():
                params.requires_grad = False
    input_size = 224

    return model_ft, input_size


def load_model(path, num_channels, num_classes):
    model, input_size = initialize_model(num_channels=num_channels, num_classes=num_classes)
    model = model.double().to(device)
    ch = torch.load(path, map_location=torch.device(device))
    model.load_state_dict(ch["state_dict"])
    return model.to(device)


if __name__ == "__main__":
    if GREY:
        num_channels = 1
    else:
        num_channels = 3

    dataset_trn = TrainDataset("data/train/", grey=GREY)
    dataset_trn.resample()
    dataset_trn.augment()
    dataset_trn.augment(first=False)
    dataset_val = TrainDataset("datasplit/val/", grey=GREY)
    logger.info("{} training samples, {} validation samples".format(len(dataset_trn), len(dataset_val)))
    dataloaders_dict = {"train": DataLoader(dataset=dataset_trn, batch_size=BATCHSIZE), "val": DataLoader(dataset=dataset_val, batch_size=BATCHSIZE)}
    num_classes = len(dataset_trn.counter.keys())

    model_ft, input_size = initialize_model(num_channels=num_channels, num_classes=num_classes)
    if len(CHECKPOINT_PATH) > 0:
        model_ft.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=torch.device(device)))
    model_ft = model_ft.to(device).double()

    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-3)
    scheduler = CosineAnnealingLR_with_Restart(optimizer_ft, T_max=4, T_mult=1.5, model=model_ft, out_dir=LOGGER, take_snapshot=True, eta_min=1e-6)

    criterion = nn.CrossEntropyLoss()

    model_ft, hist = train_model(
        model_ft, dataloaders_dict, criterion, optimizer_ft, scheduler, num_epochs=EPOCHS)


    checkpoints = sorted(glob.glob(LOGGER + "/*.tar"))
    models = [load_model(ch, num_channels=num_channels, num_classes=num_classes) for ch in checkpoints]

    if not os.path.exists('xgbdata/'):
        os.makedirs('xgbdata/')

    xgb_trn = TrainDataset("data/train/", grey=GREY, subclass=SUBCLASS)
    xgb_trn.resample()
    xgb_trn.augment()
    xgb_trn.augment(first=False)
    xgb_val = TrainDataset("datasplit/val/", grey=GREY, subclass=SUBCLASS)

    get_base_predictions_model(xgb_trn, "xgbdata/" + LOGGER + "_train.csv", models)
    logger.info("Saved xgb train data")
    get_base_predictions_model(xgb_val, "xgbdata/" + LOGGER + "_val.csv", models)
    logger.info("Saved xgb val data")

    test_set = TestDataset("data/test/")
    test_loader = DataLoader(dataset=test_set, batch_size=100)

    get_base_predictions_model(test_set, "xgbdata/" + LOGGER + "_test.csv", models, test=True)
    logger.info("Saved xgb test data")
