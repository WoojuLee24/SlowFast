TRAIN:
  ENABLE: False
  DATASET: ssv2
  BATCH_SIZE: 24
  EVAL_PERIOD: 2
  CHECKPOINT_PERIOD: 2
  AUTO_RESUME: True
  # CHECKPOINT_FILE_PATH: /ws/external/checkpoints/slowfast_8x8_r50_e196_miniKinetics200_endstop_dilation_fast_every/checkpoint_epoch_00196.pyth
  FINETUNE: False
  CHECKPOINT_TYPE: torch
DATA:
  NUM_FRAMES: 32
  SAMPLING_RATE: 2
  TRAIN_JITTER_SCALES: [128, 160]
  TRAIN_CROP_SIZE: 112
  TEST_CROP_SIZE: 128
  INPUT_CHANNEL_NUM: [3, 3]
  INV_UNIFORM_SAMPLE: True
  RANDOM_FLIP: False
  REVERSE_INPUT_CHANNEL: True
SLOWFAST:
  ALPHA: 4
  BETA_INV: 8
  FUSION_CONV_CHANNEL_RATIO: 2
  FUSION_KERNEL_SZ: 7
RESNET:
  ZERO_INIT_FINAL_BN: True
  WIDTH_PER_GROUP: 64
  NUM_GROUPS: 1
  DEPTH: 50
  TRANS_FUNC: endstop_bottleneck_transform_dilation
  STRIDE_1X1: False
  NUM_BLOCK_TEMP_KERNEL: [[3, 3], [4, 4], [6, 6], [3, 3]]
  SPATIAL_STRIDES: [[1, 1], [2, 2], [2, 2], [2, 2]]
  SPATIAL_DILATIONS: [[1, 1], [1, 1], [1, 1], [1, 1]]
NONLOCAL:
  LOCATION: [[[], []], [[], []], [[], []], [[], []]]
  GROUP: [[1, 1], [1, 1], [1, 1], [1, 1]]
  INSTANTIATION: dot_product
ENDSTOP:
  LOCATION: [[[], [0, 1, 2]], [[], [0, 1, 2, 3]], [[], [0, 1, 2, 3, 4, 5]], [[], [0, 1, 2]]]
  TYPE: "EndStopping2"
BN:
  USE_PRECISE_STATS: True
  NUM_BATCHES_PRECISE: 200
  NORM_TYPE: sync_batchnorm
  NUM_SYNC_DEVICES: 1
SOLVER:
  BASE_LR: 0.01
  LR_POLICY: steps_with_relative_lrs
  LRS: [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
  STEPS: [0, 30, 40, 45, 60]
  MAX_EPOCH: 70
  MOMENTUM: 0.9
  WEIGHT_DECAY: 1e-6
  WARMUP_EPOCHS: 0.19
  WARMUP_START_LR: 0.000025
  OPTIMIZING_METHOD: sgd
MODEL:
  NUM_CLASSES: 174
  ARCH: slowfast2
  LOSS_FUNC: cross_entropy
  DROPOUT_RATE: 0.5
TEST:
  ENABLE: False
  DATASET: ssv2
  BATCH_SIZE: 16
  NUM_ENSEMBLE_VIEWS: 1
  NUM_SPATIAL_CROPS: 1
DATA_LOADER:
  NUM_WORKERS: 8
  PIN_MEMORY: True
NUM_GPUS: 2
NUM_SHARDS: 1
RNG_SEED: 0
OUTPUT_DIR: .
LOG_MODEL_INFO: False
TENSORBOARD:
  ENABLE: True
  MODEL_VIS:
    ENABLE: True
    MODEL_WEIGHTS: True
    ACTIVATIONS: True
    INPUT_VIDEO: True
    LAYER_LIST: [s2/pathway0_res0/branch2/b, s2/pathway1_res_endstop0/branch2/b]
    GRAD_CAM:
      ENABLE: True
      LAYER_LIST: [s2/pathway0_res0/branch2/b, s2/pathway1_res_endstop0/branch2/b]