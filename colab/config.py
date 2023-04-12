import torch
# ------------ Model config --------------------- #
DEVICE: str = torch.device("cpu")
#print("Device:", DEVICE, DEVICE.type)
# Hugging face models:
HF_MODELS = {
    "ViT": "google/vit-base-patch16-224",
    "DeiT": "facebook/deit-base-distilled-patch16-224",
    "BiT": "google/bit-50",
    "ConvNeXt": "facebook/convnext-tiny-224"
    }

# Chosen model:
MODEL_NAME =  HF_MODELS['ViT']
NUM_CLASSES = 100 #10 or 100 for CIFAR10 or CIFAR100 respectively
PRE_TRAINED = True

# ------------ Training config ------------------ #
TRAIN_SIZE = 1000 # for non-IID this doesnt do anything since all clients by default are given 100 curated points
VAL_PORTION = 0.1 # 10% of the training set is for validation
TEST_SIZE = 100

BATCH_SIZE = 32
LEARNING_RATE = 0.001 # 0.00001 for all others except ConVNeXt (0.0001)
EPOCHS = 1 # EPOCHS PER CLIENT in each round

# ------------ FL config ------------------------ #
NUM_CLIENTS = 5
NUM_ROUNDS = 50
DOUBLE_TRAIN = False # Double the training size for each client in each round (for non-IID only)

FRAC_FIT = 0.5    # Sample X% of available clients for training
FRAC_EVAL = 0.5   # Sample X% of available clients for evaluation
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
    
import json
RAY_ARGS = dict(
    _system_config={
        "object_spilling_config": json.dumps(
            {
              "type": "filesystem",
              "params": {
                # Multiple directories can be specified to distribute
                # IO across multiple mounted physical devices.
                "directory_path": [
                    "C:\\Users\\Jean\\OneDrive - University of Toronto\\Courses\\2-Winter 2023\\CSC2231_visual_mobile_comp\\FL-ViT\\colab\\media\\ray_spill"
                    ]
              },
            }
        )
    },
)
