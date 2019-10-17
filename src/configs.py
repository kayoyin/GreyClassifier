LOGGER = "resnet"
PARAMS = {"eta": 0.05, "max_depth": 1, "min_child_weight": 9, "subsample": 1, "col_sample_by_tree": 1, "objective": "multi:softmax", "num_class": 4} # for XGB model
EPOCHS = 20
BATCHSIZE = 64
BLUR = True
WEIGHTS_PATH = "weights/resnet18-5c106cde.pth"
CHECKPOINT_PATH = ""
TRAIN = True
NUM_FREEZE_LAYERS = 7
GREY = False
IMG_SIZE = 256
FILTER_SIZE = 3
SUBCLASS = []
# SUBCLASS = ['bedroom', 'kitchen', 'livingroom', 'office']
# SUBCLASS = ['coast', 'forest', 'mountain', 'opencountry', 'highway']
# SUBCLASS = ['tallbuilding', 'insidecity', 'street']
CLASSES = {
    "bedroom": 0,
    "coast": 1,
    "forest": 2,
    "highway": 3,
    "insidecity": 4,
    "kitchen": 5,
    "livingroom": 6,
    "mountain": 7,
    "office": 8,
    "opencountry": 9,
    "street": 10,
    "suburb": 11,
    "tallbuilding": 12,
}

LABELS = {
    0: "bedroom",
    1: "coast",
    2: "forest",
    3: "highway",
    4: "insidecity",
    5: "kitchen",
    6: "livingroom",
    7: "mountain",
    8: "office",
    9: "opencountry",
    10: "street",
    11: "suburb",
    12: "tallbuilding",
}

CLASSNAMES = [
    "bedroom",
    "coast",
    "forest",
    "highway",
    "insidecity",
    "kitchen",
    "livingroom",
    "mountain",
    "office",
    "opencountry",
    "street",
    "suburb",
    "tallbuilding",
]
CLASSINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
