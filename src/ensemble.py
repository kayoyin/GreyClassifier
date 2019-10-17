import numpy as np
from numpy import loadtxt
import xgboost as xgb
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import DataLoader
from src.configs import *
from src._logging import get_logger
from scipy.special import softmax

logger = get_logger(LOGGER)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_base_predictions_model(dataset, output, models, test=False):
    """
    Save predictions from base learners as new dataset for meta learner
    :param output: path where new data will be stored
    :return:
    """

    def filename_to_int(file):
        return int(file[-4:])

    dataloader = DataLoader(dataset=dataset, batch_size=BATCHSIZE)

    with open(output, "ab") as file:
        if not test:
            for batch, labels in dataloader:
                batch_pred = torch.DoubleTensor()
                batch = batch.to(device).double()
                for model in models:
                    model.eval()
                    model_pred = model(batch)  # (batchsize,num_classes)
                    batch_pred = torch.cat((batch_pred, model_pred.cpu()), dim=1)
                batch_pred = torch.cat((batch_pred.cpu(), torch.tensor(labels).double().view(-1, 1)), dim=1)  # (batchsize, num_models * num_classes)
                np.savetxt(file, batch_pred.detach().numpy())

        else:
            for batch, filenames in dataloader:
                batch_pred = torch.DoubleTensor()
                batch = batch.to(device).double()
                for model in models:
                    model.eval()
                    model_pred = model(batch)  # (batchsize,num_classes)
                    batch_pred = torch.cat((batch_pred, model_pred.cpu()), dim=1)
                batch_pred = torch.cat((batch_pred.cpu(), torch.tensor(list(map(filename_to_int, filenames))).double().view(-1, 1)), dim=1)
                np.savetxt(file, batch_pred.detach().numpy())

def get_pred(model, output_margin=False):
    """
    Obtain predictions by XGB classifier on a certain subclass
    :param model:
    :param output_margin:
    :return:
    """
    dataset_val = loadtxt("xgbdata/" + model + "_val.csv", delimiter=" ")
    test_data = loadtxt("xgbdata/" + model + "_test.csv", delimiter=" ")
    x_val, y_val = dataset_val[:, :-1], dataset_val[:, -1]
    x_test, filenames = test_data[:, :-1], test_data[:, -1]

    xgbmodel = xgb.Booster({"nthread": 4})
    xgbmodel.load_model("weights/" + model + ".model")

    valmat = xgb.DMatrix(data=x_val, label=y_val)
    testmat = xgb.DMatrix(data=x_test)

    pred_val = xgbmodel.predict(valmat, output_margin=output_margin)
    pred_test = xgbmodel.predict(testmat, output_margin=output_margin)

    return pred_val, y_val, pred_test, filenames


def subclass_predictions(preds, subpreds):
    """
    Obtain final prediction by combining subclass predictions
    :param preds:
    :param subpreds:
    :return:
    """
    superpred = []
    room_dict = {0: 0, 1: 5, 2: 6, 3: 8}
    nature_dict = {0: 1, 1: 2, 2: 7, 3: 9, 4: 3}
    urban_dict = {0: 12, 1: 4, 2: 10}
    for i, p in enumerate(preds):

        p = softmax(p)
        argmaxp = np.argmax(p)
        if np.max(p) < 0.5:
            if argmaxp in urban_dict.values():
                superpred.append(urban_dict[subpreds["urban"][i]])
            elif argmaxp in room_dict.values():
                superpred.append(room_dict[subpreds["rooms"][i]])
            elif argmaxp in nature_dict.values():
                superpred.append(nature_dict[subpreds["nature"][i]])
            else:
                superpred.append(argmaxp)
        else:
            superpred.append(argmaxp)
    return superpred


if __name__ == "__main__":

    if TRAIN:
        dataset_trn = loadtxt("xgbdata/" + LOGGER + "_train.csv", delimiter=" ")
        dataset_val = loadtxt("xgbdata/" + LOGGER + "_val.csv", delimiter=" ")

        x_train, y_train = dataset_trn[:, :-1], dataset_trn[:, -1]
        x_val, y_val = dataset_val[:, :-1], dataset_val[:, -1]

        trainmat = xgb.DMatrix(data=x_train, label=y_train)
        valmat = xgb.DMatrix(data=x_val, label=y_val)

        steps = EPOCHS  # The number of training iterations

        model = xgb.train(PARAMS, trainmat, steps)
        trainpreds = model.predict(trainmat)
        logger.info("Training accuracy: %.2f%%" % (accuracy_score(trainpreds, y_train) * 100.0))
        valpreds = model.predict(valmat, output_margin=True)
        logger.info("Validation accuracy: %.2f%%" % (accuracy_score(np.argmax(valpreds, axis=1), y_val) * 100.0))

        model.save_model("weights/" + LOGGER + ".model")
        logger.info("Model saved!")

    else:
        resnet_val, y_val, resnet_test, filenames = get_pred("resnet", output_margin=True)
        rooms_val, y_val, rooms_test, filenames = get_pred("rooms")
        nature_val, y_val, nature_test, filenames = get_pred("nature")
        urban_val, y_val, urban_test, filenames = get_pred("urban")

        head = "image_0000"
        filenames = [head[: -len(str(f)[:-2])] + str(f)[:-2] for f in filenames]

        pred_val = subclass_predictions(resnet_val, {"rooms": rooms_val, "nature": nature_val, "urban": urban_val})
        logger.info("Validation accuracy: %.2f%%" % (accuracy_score(pred_val, y_val) * 100.0))

        pred_test = subclass_predictions(resnet_test, {"rooms": rooms_test, "nature": nature_test, "urban": urban_test})
        with open("xgb.csv", "w") as file:
            file.write("id,label\n")
            for name, p in zip(filenames, pred_test):
                file.write(name + "," + LABELS[p] + "\n")
        logger.info("Submission saved!")

