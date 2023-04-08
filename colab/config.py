import torch
# ------------ Model config --------------------- #
DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Hugging face models:
HF_MODELS = {
    "ViT": "google/vit-base-patch16-224",
    "DeiT": "facebook/deit-base-distilled-patch16-224",
    "BiT": "google/bit-50",
    "ConvNeXt": "facebook/convnext-tiny-224"
    }

# Chosen model:
MODEL_NAME =  HF_MODELS['ViT']
NUM_CLASSES = 10 #10 or 100 for CIFAR10 or CIFAR100 respectively
PRE_TRAINED = True

# ------------ Training config ------------------ #
TRAIN_SIZE = 100
VAL_PORTION = 0.1 # 10% of the training set is for validation
TEST_SIZE = 50

BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 1 # EPOCHS PER CLIENT in each round

# ------------ FL config ------------------------ #
NUM_CLIENTS = 2
NUM_ROUNDS = 10

FRAC_FIT = 1.0    # Sample X% of available clients for training
FRAC_EVAL = 1.0   # Sample X% of available clients for evaluation
MIN_FIT = 0       # Never sample less than this for training
MIN_EVAL = 0      # Never sample less than this for evaluation
MIN_AVAIL = 0     # Wait until all these # of clients are available

FIT_CONFIG_FN = lambda srvr_rnd: {
        "server_round": srvr_rnd,
        "local_epochs": EPOCHS
        }

# CPU and GPU resources for a single client. 
# Supported keys are num_cpus and num_gpus.
#   SEE Ray documentation for more details. https://docs.ray.io/en/latest/ray-core/tasks/using-ray-with-gpus.html)
CLIENT_RESOURCES = None
if DEVICE.type == "cuda":
    CLIENT_RESOURCES = {"num_gpus": 1}