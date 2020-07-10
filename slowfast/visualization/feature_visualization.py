#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import numpy as np
import torch

import slowfast.datasets.utils as data_utils
import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc
import slowfast.visualization.tensorboard_vis as tb
from slowfast.datasets import loader
from slowfast.models import build_model

logger = logging.get_logger(__name__)


def run_visualization(vis_loader, model, cfg, writer=None):
    n_devices = cfg.NUM_GPUS * cfg.NUM_SHARDS

    global_idx = 0
    for inputs, _, _, _ in vis_loader:
        # Transfer the data to the current GPU device.
        if isinstance(inputs, (list,)):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda(non_blocking=True)
        else:
            inputs = inputs.cuda(non_blocking=True)

        inputs = du.all_gather_unaligned(inputs)

        if writer is not None:
            total_vids = 0
            for i in range(n_devices):
                cur_input = inputs[i]
                cur_batch_size = cur_input[0].shape[0]

                for cur_batch_idx in range(cur_batch_size):
                    global_idx += 1
                    total_vids += 1
                    for path_idx, input_pathway in enumerate(cur_input):
                        if cfg.TEST.DATASET == "ava" and cfg.AVA.BGR:
                            video = input_pathway[cur_batch_idx, [2, 1, 0], ...]
                        else:
                            video = input_pathway[cur_batch_idx]
                        # Permute to (T, H, W, C) from (C, T, H, W).
                        video = video.permute(1, 2, 3, 0)
                        video = data_utils.revert_tensor_normalize(
                            video.cpu(), cfg.DATA.MEAN, cfg.DATA.STD
                        )
                        video = video.permute(0, 3, 1, 2).unsqueeze(0)
                        writer.add_video(
                            video,
                            tag="Input {}/Input from pathway {}".format(
                                global_idx, path_idx + 1
                            ),
                        )

            logger.info("Visualized {} videos...".format(total_vids))


def visualize(cfg):
    """
    Perform layer weights and activations visualization on the model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    if cfg.TENSORBOARD.ENABLE and cfg.TENSORBOARD.MODEL_VIS.ENABLE:
        # Set up environment.
        du.init_distributed_training(cfg)
        # Set random seed from configs.
        np.random.seed(cfg.RNG_SEED)
        torch.manual_seed(cfg.RNG_SEED)

        # Setup logging format.
        logging.setup_logging(cfg.OUTPUT_DIR)

        # Print config.
        logger.info("Model Visualization with config:")
        logger.info(cfg)

        # Build the video model and print model statistics.
        model = build_model(cfg)
        if du.is_master_proc() and cfg.LOG_MODEL_INFO:
            misc.log_model_info(model, cfg, is_train=False)

        cu.load_test_checkpoint(cfg, model)

        # Create video testing loaders.
        vis_loader = loader.construct_loader(cfg, "test")
        logger.info(
            "Visualize model for {} data points".format(len(vis_loader))
        )

        if cfg.DETECTION.ENABLE:
            assert cfg.NUM_GPUS == cfg.TEST.BATCH_SIZE

        # Set up writer for logging to Tensorboard format.
        if du.is_master_proc(cfg.NUM_GPUS * cfg.NUM_SHARDS):
            writer = tb.TensorboardWriter(cfg)
        else:
            writer = None

        # Run visualization on the model
        run_visualization(vis_loader, model, cfg, writer)

        if writer is not None:
            writer.close()
=======
"""
Created on Sat Nov 18 23:12:08 2017

@author: Utku Ozbulak - github.com/utkuozbulak

Modified by WooJu Lee
"""

import os
import numpy as np

import torch
from torch.optim import SGD
from torchvision import models
from slowfast.models import build_model

from slowfast.utils.vis_misc import preprocess_video, recreate_video, save_gif
from PIL import Image
from PIL import ImageFilter
from scipy.ndimage import filters

from slowfast.utils.parser import load_config, parse_args
from slowfast.utils.checkpoint import load_checkpoint

checkpoint_path = '/ws/external/checkpoint/SLOWFAST_8x8_R50.pkl'


class CNNLayerVisualization():
    """
        Produces an image that maximizes a convolution
        operation for a specific layer and filter
        cnn_layer = "layer1"
        block_pos = 0
        sub_layer = "conv1"
        filter_pos = 5
    """

    def __init__(self, model, layer, filter, device):

        self.device = device
        self.model = model
        self.model.eval()
        self.model_name = model.__class__.__name__

        self.layer = layer
        # self.block = block
        # self.sublayer = sublayer
        self.filter = filter
        self.conv_output = 0

        self.H = 224
        self.W = 224
        self.T = 32

        self.min = -0.5
        self.max = 0.5
        self.initial_learning_rate = 150
        self.upscaling_factor = 1.2
        self.upscaling_steps = 12
        self.iteration_steps = 1000
        self.weight_decay = 0

        # Create the folder to export images if not exists
        if not os.path.exists('../generated/'):
            os.makedirs('../generated/')
        self.dir_path = "../generated/{}/".format(str(self.model_name))
        if not os.path.exists(self.dir_path):
            os.mkdir(self.dir_path)

    def hook_layer(self):
        def hook_function(module, grad_in, grad_out):
            # Gets the conv output of the selected filter (from selected layer)
            self.conv_output = grad_out[0, self.filter]

        # Hook the selected layer
        for n, m in self.model.named_modules():
            if n == str(self.layer):
                m.register_forward_hook(hook_function)


    def visualise_layer_with_hooks(self):

        # Hook the selected layer
        self.hook_layer()

        # Generate a random image
        video1 = np.uint8(np.random.uniform(150, 180, (1, self.H, self.W, 3))) * np.ones((self.T, self.H, self.W, 3))  # [16, 224, 224, 3], 0 to 255 float
        video2 = []
        for i in range(np.shape(video1)[0]):
            if i % 4 == 0:
                video2.append(video1[i])
        video2 = np.asarray(video2)
        self.created_video = [video2, video1]
        # self.created_video = np.uint8(np.random.uniform(150, 180, (self.T, self.H, self.W, 3)))

        for i in range(self.upscaling_steps):
            self.processed_video = []
            # Process image and return variable
            processed_video2 = preprocess_video(self.created_video[0], False, self.device)  # [16, H, W, 3] (0, 255) -> [1,3,16,H,W] (-1, 1) float
            processed_video1 = preprocess_video(self.created_video[1], False, self.device)
            # processed_var = torch.cat((processed_video2, processed_video1), dim=2)
            self.processed_video = [processed_video2, processed_video1]
            # Define optimizer for the image
            optimizer = SGD([processed_video2, processed_video1], lr=self.initial_learning_rate, weight_decay=self.weight_decay)
            # optimizer1 = SGD([processed_video1], lr=self.initial_learning_rate, weight_decay=self.weight_decay)
            # optimizer2 = SGD([processed_video2], lr=self.initial_learning_rate, weight_decay=self.weight_decay)

            for j in range(self.iteration_steps):
                optimizer.zero_grad()
                output = self.model([processed_video2, processed_video1])
                # Loss function is the mean of the output of the selected layer/filter
                # We try to minimize the mean of the output of that specific filter
                loss = -torch.mean(self.conv_output)
                # Backward
                loss.backward(retain_graph=True)
                # Update image
                optimizer.step()

                print('Upscaling:', str(i + 1), 'Iteration:', str(j + 1), 'Loss:', "{0:.5f}".format(loss))

                del output, loss

            self.created_video = []

            self.H = int(self.H * self.upscaling_factor)
            self.W = int(self.W * self.upscaling_factor)
            for j in range(2):
                # Recreate video
                created_video = recreate_video(self.processed_video[j])  # [1,3,16,H,W] (-1,1) float -> [16,H,W,3] (0, 255) uint8
                torch.cuda.empty_cache()
                # Upscale and blur spatially
                new_created_video = []
                for k in range(np.shape(created_video)[0]):
                    img = Image.fromarray(created_video[k])
                    img = img.resize((self.H, self.W), resample=Image.BICUBIC)
                    img = img.filter(ImageFilter.BoxBlur(radius=1))
                    new_created_video.append(np.uint8(img))
                created_video = np.asarray(new_created_video)
                # blur temporally
                # self.created_video = filters.uniform_filter1d(self.created_video, size=3)
                created_video = filters.gaussian_filter1d(created_video, axis=0, sigma=1)
                self.created_video.append(created_video)

                # Save gif
                if (i+1) % 5 == 0:
                    layer_path = self.dir_path + 'layer/'
                    if not os.path.exists(layer_path):
                        os.mkdir(layer_path)

                    path = layer_path + str(self.model_name) + '_iter' + str(i) + '_path' + str(j) + '_' + \
                           str(self.layer) + '_f' + str(self.filter) + '_lr' + str(self.initial_learning_rate) + "_wd" \
                           + str(self.weight_decay)
                    save_gif(created_video, path, stream_type="rgb")


if __name__ == '__main__':

    args = parse_args()
    cfg = load_config(args)
    # Construct the model
    model = build_model(cfg)
    load_checkpoint(checkpoint_path, model, data_parallel=False, optimizer=None, inflation=False, convert_from_caffe2=True)
    cnn_layer = "s2.pathway0_res0.branch1"     # "conv3d_0c_1x1.conv3d"
    filter_pos = 0

    device = torch.device('cuda:0')
    model = model.to(device)

    layer_vis = CNNLayerVisualization(model, cnn_layer, filter_pos, device)

    # Layer visualization with pytorch hooks
    layer_vis.visualise_layer_with_hooks()

