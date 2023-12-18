import torch
from torch.utils.data import random_split, DataLoader
from data.maestro_dataset import MaestroDataset
from miditok import REMIPlus
from palm_rlhf_pytorch import PaLM
from accelerate import Accelerator
import tqdm
from sample import generate_sample

# hyperparameters
BATCH_SIZE = 8
DIM = 512
SEQUENCE_LENGTH = 2048
DEPTH = 12
FLASH_ATTN = True
NUM_EPOCHS = 100
LR = 1e-4
VALIDATE_EVERY = 20

# create the tokenizer
tokenizer = REMIPlus()

# create the dataset
dataset = MaestroDataset(
    root_dir='data/maestro_tokenized',
    tokenizer=tokenizer,
    sequence_length=SEQUENCE_LENGTH
)

print("Created dataset.")

# split the dataset into training and validation sets
train_set, val_set = random_split(dataset, [0.8, 0.2])

print("Created training and val sets.")

# create the training and validation dataloaders
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader_iter = iter(val_loader)

print("Created training and val loaders.")

# check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# create the model
model = PaLM(
    num_tokens=tokenizer.len,
    dim = DIM,
    depth = DEPTH,
    flash_attn=FLASH_ATTN
).to(device)

num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'{num_parameters} parameters')

# create the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# create the accelerator
accelerator = Accelerator()

# train the model
for epoch in range(NUM_EPOCHS):

    print()
    print(f'Epoch {epoch + 1}/{NUM_EPOCHS}')
    print()

    for i, batch in enumerate(tqdm.tqdm(train_loader, desc='Training')):
        model.train()
        loss = model(batch.long().to(device), return_loss=True)
        accelerator.backward(loss)

        accelerator.print(f"training loss: {loss.item()}")
        accelerator.clip_grad_norm_(model.parameters(), 0.5)

        optimizer.step()
        optimizer.zero_grad()

        if i % VALIDATE_EVERY == 0:
            model.eval()
            with torch.no_grad():
                loss = model(next(val_loader_iter).long().to(device), return_loss=True)
                accelerator.print(f"validation loss: {loss.item()}")
    
    # save the model weights after each epoch
    torch.save(model.state_dict(), f'model/model_weights_epoch_{epoch+1}.pth')

    # generate a sample after each epoch
    generate_sample(model, tokenizer, f'samples/sample_epoch_{epoch+1}.mid')