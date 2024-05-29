COCO_CLASSES_LIST = [
    'person',
    'bicycle',
    'car',
    'motorbike',
    'aeroplane',
    'bus',
    'train',
    'truck',
    'boat',
    'traffic light',
    'fire hydrant',
    'stop sign',
    'parking meter',
    'bench',
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
    'backpack',
    'umbrella',
    'handbag',
    'tie',
    'suitcase',
    'frisbee',
    'skis',
    'snowboard',
    'sports ball',
    'kite',
    'baseball bat',
    'baseball glove',
    'skateboard',
    'surfboard',
    'tennis racket',
    'bottle',
    'wine glass',
    'cup',
    'fork',
    'knife',
    'spoon',
    'bowl',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'hot dog',
    'pizza',
    'donut',
    'cake',
    'chair',
    'sofa',
    'pottedplant',
    'bed',
    'diningtable',
    'toilet',
    'tvmonitor',
    'laptop',
    'mouse',
    'remote',
    'keyboard',
    'cell phone',
    'microwave',
    'oven',
    'toaster',
    'sink',
    'refrigerator',
    'book',
    'clock',
    'vase',
    'scissors',
    'teddy bear',
    'hair drier',
    'toothbrush',
]

OBSTACLE_15 = [
    "person",
    "bicycle",
    "bus",
    "car",
    "carrier",
    "motorcycle",
    "movable_signage",
    "truck",
    "bollard",
    "chair",
    "potted_plant",
    "table",
    "tree_trunk	",
    "pole",
    "fire_hydrant",
]

OBSTACLE_PERSON = [
    'person'
]

OBJECT_DATASET_CLASSES = {
    "coco": COCO_CLASSES_LIST,
    "obstacle-15": OBSTACLE_15,
}

def get_cls_dict(category_num):
    """Get the class ID to name translation dictionary."""
    if category_num == 80:
        return {i: n for i, n in enumerate(COCO_CLASSES_LIST)}
    elif category_num == 15: # obstacle
        return {i: n for i, n in enumerate(OBSTACLE_15)}
    elif category_num == 'obstacle_person':
        return {i: n for i, n in enumerate(OBSTACLE_PERSON)}
    else:
        return {i: 'CLS%d' % i for i in range(category_num)}

def get_class(dataset_name):
    dataset_class = OBJECT_DATASET_CLASSES[dataset_name]
    return {i: n for i, n in enumerate(dataset_class)}, dataset_class

JOINT_PAIR = {
    "coco": [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16],
             [20, 21], [22, 23], [24, 25], [26, 42], [27, 41], [28, 40], [29, 39], [30, 38],
             [31, 37], [32, 36], [33, 35], [43, 52], [44, 51], [45, 50], [46, 49], [47, 48],
             [62, 71], [63, 70], [64, 69], [65, 68], [66, 73], [67, 72], [57, 61], [58, 60],
             [74, 80], [75, 79], [76, 78], [87, 89], [93, 91], [86, 90], [85, 81], [84, 82],
             [94, 115], [95, 116], [96, 117], [97, 118], [98, 119], [99, 120], [100, 121],
             [101, 122], [102, 123], [103, 124], [104, 125], [105, 126], [106, 127], [107, 128],
             [108, 129], [109, 130], [110, 131], [111, 132], [112, 133], [113, 134], [114, 135]]
}