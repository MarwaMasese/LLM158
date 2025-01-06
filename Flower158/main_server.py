import flwr as fl
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import bitsandbytes as bnb

# Hyperparameters
MODEL_NAME = "gpt2"  # Small model for demo purposes
BATCH_SIZE = 2
EPOCHS = 1
QUANT_BITS = 1.58  # Custom quantization logic

# Prepare data (dummy dataset for example)
class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer):
        self.data = ["Hello, world!", "Flower.ai is great!"]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.tokenizer(self.data[idx], return_tensors="pt", padding="max_length", max_length=50, truncation=True)

def train(model, dataloader, device):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        for batch in dataloader:
            input_ids = batch["input_ids"].squeeze(1).to(device)
            labels = input_ids.clone()
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def get_model_and_data():
    # Load quantized model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb.QuantizationConfig(bits=2),  # Simulating 1.58 bits
    )
    dataset = DummyDataset(tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
    return model, dataloader, tokenizer

class FlowerClient(fl.client.NumPyClient):
    def __init__(self):
        self.model, self.dataloader, self.tokenizer = get_model_and_data()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.model.named_parameters()]

    def set_parameters(self, parameters):
        for param, new_val in zip(self.model.parameters(), parameters):
            param.data = torch.tensor(new_val).to(self.device)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.model, self.dataloader, self.device)
        return self.get_parameters(), len(self.dataloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        return float("inf"), 0, {}  # Simplified evaluation logic

def main():
    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=FlowerClient())

if __name__ == "__main__":
    main()
