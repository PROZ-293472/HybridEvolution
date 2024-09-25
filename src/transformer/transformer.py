import pickle
import uuid

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm


class VectorSequenceDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        return torch.tensor(sequence, dtype=torch.float32)


class TransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim):
        super(TransformerModel, self).__init__()
        self.model_dim = model_dim

        self.embedding = nn.Linear(input_dim, model_dim)
        self.positional_encoding = self._generate_positional_encoding(1000, model_dim)

        self.transformer = nn.Transformer(
            d_model=model_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers
        )

        self.fc_out = nn.Linear(model_dim, output_dim)

    def forward(self, src, tgt):
        src = self.embedding(src) + self.positional_encoding[:src.size(0), :]
        tgt = self.embedding(tgt) + self.positional_encoding[:tgt.size(0), :]

        output = self.transformer(src, tgt)
        output = self.fc_out(output)
        return output

    def _generate_positional_encoding(self, max_len, model_dim):
        pos_enc = torch.zeros(max_len, model_dim)
        positions = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / model_dim))

        pos_enc[:, 0::2] = torch.sin(positions * div_term)
        pos_enc[:, 1::2] = torch.cos(positions * div_term)

        pos_enc = pos_enc.unsqueeze(1)
        return pos_enc

    def continue_sequence(self, input_sequence, continuation_len):
        input_sequence = torch.Tensor(input_sequence)
        src = input_sequence.unsqueeze(0).permute(1, 0, 2)  # [N, 1, input_dim]
        
        self.eval()  

        tgt = torch.zeros((continuation_len, 1, input_sequence.size(-1)))
        full_tgt_sequence = torch.cat((input_sequence.unsqueeze(0).permute(1, 0, 2), tgt), dim=0)
        
        with torch.no_grad():
            src_emb = self.embedding(src) + self.positional_encoding[:src.size(0), :]
            tgt_emb = self.embedding(full_tgt_sequence) + self.positional_encoding[:full_tgt_sequence.size(0), :]

            output = self.transformer(src_emb, tgt_emb)

            full_output_sequence = self.fc_out(output)

        return full_output_sequence.permute(1, 0, 2).squeeze(0).numpy()



if __name__ == '__main__':
    function = 'egg'
    sequence_len = 50

    with open(f'../../data/sequences_{function}_{sequence_len}_means_tokenized_dim_40.pkl', 'rb') as file:
        sequences = pickle.load(file)
    dataset = VectorSequenceDataset(sequences)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=10, shuffle=False)

    input_dim = 2
    model_dim = 100
    num_heads = 10
    num_layers = 6
    output_dim = input_dim

    model = TransformerModel(input_dim, model_dim, num_heads, num_layers, output_dim)

    exp_id = uuid.uuid4()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    epochs = 20
    train_losses = []
    val_losses = []

    model.train()

    for epoch in range(epochs):
        # Training
        running_train_loss = 0.0
        for batch in tqdm(train_dataloader):
            optimizer.zero_grad()
            batch = batch.permute(1, 0, 2)
            src = batch[:-1, :, :]
            tgt = batch[1:, :, :]

            output = model(src, tgt)
            loss = criterion(output, tgt)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                batch = batch.permute(1, 0, 2)
                src = batch[:-1, :, :]
                tgt = batch[1:, :, :]

                output = model(src, tgt)
                loss = criterion(output, tgt)

                running_val_loss += loss.item()

        avg_val_loss = running_val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)

        torch.save(model, f'../../models/model_{exp_id}_{function}_{sequence_len}_dim_40.pth')
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}")
        print(train_losses, val_losses)

        model.train()

    print(train_losses, val_losses)



