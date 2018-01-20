import torch
import numpy as np
import visdom


class Visualizer:
    def __init__(self):
        self.vis = visdom.Visdom()

    def batch_images(self, batch):
        self.vis.images(batch, win='batch', nrow=16)

