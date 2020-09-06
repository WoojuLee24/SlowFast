#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from fvcore.common.config import CfgNode

"""Add custom configs and default values"""


def add_custom_config(_C):
    # Add your own customized configs.
    # Index of each stage and block to add endstopping layers.
    _C.ENDSTOP = CfgNode()
    _C.ENDSTOP.LOCATION = [[[]], [[]], [[]], [[]]]
    pass
