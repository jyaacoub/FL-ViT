import torch, json, os
# ------------ Model config --------------------- #
NUM_CLASSES = 100 #10 or 100 for CIFAR10 or CIFAR100 respectively
NON_IID = False # True to load non-IID data from TFF, False to load IID data from torchvision
assert not(NUM_CLASSES != 10 and NON_IID), "Non-IID is only supported for CIFAR100"

DEVICE: str = torch.device("cpu")

TFF_DATA_DIR = lambda x: f'data/tff_dataloaders_10clients/{x}.pth'

# Hugging face models:
HF_MODELS = {
    "ViT": "google/vit-base-patch16-224",
    "DeiT": "facebook/deit-base-distilled-patch16-224",
    "BiT": "google/bit-50",
    "ConvNeXt": "facebook/convnext-tiny-224"
    }

# Chosen model:
MODEL_NAME =  HF_MODELS['ConvNeXt']
PRE_TRAINED = True

# ------------ Training config ------------------ #
TRAIN_SIZE = 1000 # for non-IID this doesnt do anything since all clients by default are given 100 curated points
VAL_PORTION = 0.1 # 10% of the training set is for validation
TEST_SIZE = 100

BATCH_SIZE = 32
LEARNING_RATE = 0.0001 # 0.00001 for all others except ConVNeXt (0.0001)
EPOCHS = 1 # EPOCHS PER CLIENT in each round

# ------------ FL config ------------------------ #
SERVER_ADDRESS = "JCY-PC" # LAN setup for actual FL env
NUM_CLIENTS = 5
NUM_ROUNDS = 50
DOUBLE_TRAIN = True # Double the training size for each client in each round (for non-IID only)

FRAC_FIT = 0.5    # Sample X% of available clients for training
FRAC_EVAL = 0.5   # Sample X% of available clients for evaluation
MIN_FIT = 0       # Never sample less than this for training
MIN_EVAL = 0      # Never sample less than this for evaluation
MIN_AVAIL = 0     # Wait until all these # of clients are available

FIT_CONFIG_FN = lambda srvr_rnd: {
        "server_round": srvr_rnd,
        "local_epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        }

# CPU and GPU resources for a single client. 
# Supported keys are num_cpus and num_gpus.
#   SEE Ray documentation for more details. https://docs.ray.io/en/latest/ray-core/tasks/using-ray-with-gpus.html)
CLIENT_RESOURCES = None
if DEVICE.type == "cuda":
    CLIENT_RESOURCES = {"num_gpus": 1}
    
curr_dir_path = os.path.dirname(os.path.realpath(__file__)) + "\\data\\ray_spill"
if not os.path.exists(curr_dir_path):
    os.makedirs(curr_dir_path)

RAY_ARGS = dict(
    _system_config={
        "object_spilling_config": json.dumps(
            {
              "type": "filesystem",
              "params": {
                # Multiple directories can be specified to distribute
                # IO across multiple mounted physical devices.
                "directory_path": [curr_dir_path]
              },
            }
        )
    },
)
