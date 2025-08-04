import json
import torch
from model import CharRNN
import config


def load_model_for_inference():
    with open(config.MAPPING_SAVE_PATH, 'r', encoding='utf-8') as f:
        char_to_int = json.load(f)

    int_to_char = {int(i): ch for ch, i in char_to_int.items()}
    n_vocab = len(char_to_int)

    model = CharRNN(n_vocab, config.HIDDEN_SIZE, n_vocab, config.NUM_LAYERS)
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE))
    model.to(config.DEVICE)
    model.eval()

    return model, char_to_int, int_to_char


def predict_next_chars(model, char_to_int, int_to_char, prompt, n_chars=100):
    model.eval()
    result = prompt

    with torch.no_grad():
        h = model.init_hidden(1)

        try:
            input_seq = torch.tensor([char_to_int[c] for c in prompt], dtype=torch.long).unsqueeze(0).to(config.DEVICE)
        except KeyError as e:
            return f"Error: The character '{e.args[0]}' is not in the model's vocabulary."

        _, h = model(input_seq, h)

        last_char_input = input_seq[:, -1].unsqueeze(1)

        for _ in range(n_chars):
            output, h = model(last_char_input, h)

            _, top_i = output.topk(1)
            predicted_char = int_to_char.get(top_i.item(), "?")

            result += predicted_char

            last_char_input = top_i.to(config.DEVICE)

    return result