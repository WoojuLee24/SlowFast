#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Add custom configs and default values"""


def add_custom_config(_C):
    #
    _C.SLOWFAST.FUSION_TRANS_FUNC = ""
    _C.SLOWFAST.SLOW_FUSION_CONV_CHANNEL_RATIO = 4
    _C.SLOWFAST.SLOW_FUSION_KERNEL_SZ = 1
    pass
