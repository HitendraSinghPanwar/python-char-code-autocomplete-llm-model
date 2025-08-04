import torch

FILE_PATH = "conala-paired-train.json"
MODEL_SAVE_PATH = "char_autocomplete_model.pth"
MAPPING_SAVE_PATH = "char_mappings.json"

SEQ_LENGTH = 100
HIDDEN_SIZE = 256
NUM_LAYERS = 2
LEARNING_RATE = 0.002
NUM_EPOCHS = 10
BATCH_SIZE = 128
TRAINING_SIZE_LIMIT = 500000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
