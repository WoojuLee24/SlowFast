#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from fvcore.common.config import CfgNode

"""Add custom configs and default values"""


def add_custom_config(_C):
    # Add your own customized configs.
    _C.V1 = CfgNode()
    _C.V1.ENABLE = False
    _C.V1.ENDSTOP = "endstopping_divide"
    pass
