import os
import logging
import datetime
import json
import random
import numpy as np
from tensorboardX import SummaryWriter

# My imports
import matplotlib.pyplot as plt
import numpy as np
    
def plot_grid(img, vPoints):
    plt.figure(figsize=(8, 8))
    plt.imshow(img, cmap='gray')
    plt.scatter(vPoints[:, 0], vPoints[:, 1], s=10, c='red', marker='o')
    plt.axis('off')
    plt.show()


def get_logger(logdir):
    logger = logging.getLogger('emotion')
    ts = str(datetime.datetime.now()).split('.')[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-","_")
    file_path = os.path.join(logdir, 'run_{}.log'.format(ts))
    hdlr = logging.FileHandler(file_path)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger

def save_config(logdir, config):
    param_path = os.path.join(logdir, "params.json")
    print("[*] PARAM path: %s" % param_path)
    with open(param_path, 'w') as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)


def recursive_glob(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    image_paths = []
    for looproot, _, filenames in os.walk(rootdir):
        for filename in filenames:
            if filename.endswith(suffix):
                image_paths.append(os.path.join(looproot, filename))
    return image_paths



class averageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count