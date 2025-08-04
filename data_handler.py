import json
import torch


def load_data_from_json(filepath):
    code_snippets = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data_item = json.loads(line)
                    if 'snippet' in data_item:
                        code_snippets.append(data_item['snippet'])

        if not code_snippets:
            print("Warning: No 'snippet' keys found in the JSONL file. Using dummy data.")
            return get_dummy_data()

        return "\n".join(code_snippets)

    except FileNotFoundError:
        print(f"Error: File not found at {filepath}. Using dummy data instead.")
        return get_dummy_data()
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from a line in {filepath}: {e}")
        print("The file might be corrupted or not in valid JSONL format. Using dummy data.")
        return get_dummy_data()


def get_dummy_data():
    return """
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b
"""


def create_char_mappings(text):
    chars = sorted(list(set(text)))
    n_chars = len(chars)
    char_to_int = {ch: i for i, ch in enumerate(chars)}
    int_to_char = {i: ch for i, ch in enumerate(chars)}
    return char_to_int, int_to_char, n_chars


def prepare_sequences(text, char_to_int, seq_length):
    dataX = []
    dataY = []
    for i in range(0, len(text) - seq_length, 1):
        seq_in = text[i:i + seq_length]
        seq_out = text[i + seq_length]
        dataX.append([char_to_int[char] for char in seq_in])
        dataY.append(char_to_int[seq_out])

    X = torch.tensor(dataX, dtype=torch.long)
    y = torch.tensor(dataY, dtype=torch.long)
    return X, y