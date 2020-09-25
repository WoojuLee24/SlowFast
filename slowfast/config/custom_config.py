#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from fvcore.common.config import CfgNode

"""Add custom configs and default values"""


def add_custom_config(_C):
    # Add your own customized configs.
    # Index of each stage and block to add endstopping layers.
    _C.ENDSTOP = CfgNode()
    _C.ENDSTOP.LOCATION = [[[]], [[]], [[]], [[]]]
    _C.ENDSTOP.TYPE = "EndStopping"
    # Transfer function
    _C.RESNET.TRANS_FUNC = CfgNode()
    _C.RESNET.TRANS_FUNC.DEFAULT = "bottleneck_transform"
    _C.RESNET.TRANS_FUNC.SLOW = CfgNode()
    _C.RESNET.TRANS_FUNC.SLOW.TYPE = "bottleneck_transform"
    _C.RESNET.TRANS_FUNC.SLOW.LOCATION = [[[], []], [[], []], [[], []], [[], []]]
    _C.RESNET.TRANS_FUNC.FAST = CfgNode()
    _C.RESNET.TRANS_FUNC.FAST.TYPE = "endstop_bottleneck_transform"
    _C.RESNET.TRANS_FUNC.FAST.LOCATION = [[[], []], [[], []], [[], []], [[], []]]
    _C.TRAIN.FINETUNE = False

    pass
