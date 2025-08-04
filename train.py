import json
import torch
import torch.nn as nn
import torch.optim as optim

import config
from data_handler import load_data_from_json, create_char_mappings, prepare_sequences
from model import CharRNN


def start_training(model, X, y):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    print("Starting model training...")
    for epoch in range(config.NUM_EPOCHS):
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True)

        total_loss = 0

        for i, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(config.DEVICE), targets.to(config.DEVICE)

            h = model.init_hidden(inputs.size(0))

            outputs, h = model(inputs, h)
            h = h.detach()

            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{config.NUM_EPOCHS}, Loss: {total_loss / len(loader):.4f}")

    print("Training finished.")


def save_model_assets(model, char_to_int_map):
    torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
    with open(config.MAPPING_SAVE_PATH, 'w', encoding='utf-8') as f:
        json.dump(char_to_int_map, f)
    print(f"Model saved to {config.MODEL_SAVE_PATH} and mappings to {config.MAPPING_SAVE_PATH}")


if __name__ == "__main__":
    print(f"Using device: {config.DEVICE}")

    print("Loading data...")
    text_data = load_data_from_json(config.FILE_PATH)
    text_data = text_data[:config.TRAINING_SIZE_LIMIT]

    char_to_int, int_to_char, n_vocab = create_char_mappings(text_data)

    print("Preparing dataset sequences...")
    X, y = prepare_sequences(text_data, char_to_int, config.SEQ_LENGTH)

    model = CharRNN(n_vocab, config.HIDDEN_SIZE, n_vocab, config.NUM_LAYERS).to(config.DEVICE)

    start_training(model, X, y)

    save_model_assets(model, char_to_int)