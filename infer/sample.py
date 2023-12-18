import argparse
import torch
from miditok import REMIPlus
from palm_rlhf_pytorch import PaLM
from tqdm import tqdm

def generate_sample(model, tokenizer, output_path, sequence_length=2048, temperature=1.0):
    """
    Generate a sample from the model.

    Arguments:
        model (PaLM): the model to generate the sample from
        tokenizer (REMIPlus): the tokenizer to use to convert the generated tokens to MIDI
        sequence_length (int): the length of the sequence to generate
        temperature (float): the temperature for sampling
    """
    # generate the token sequence
    tokens = model.generate(100, temperature=temperature, use_tqdm=True)

    print(tokens.shape)
    # convert the tokens to MIDI and save it
    midi_data = tokenizer(tokens[0])
    midi_data.dump(output_path)

if __name__ == '__main__':
    # create an ArgumentParser object
    parser = argparse.ArgumentParser(description='Generate a sample from the model.')

    # add arguments to the parser
    parser.add_argument('--output_path', type=str, help='output path for the generated MIDI file')
    parser.add_argument('--temperature', type=float, default=1.0, help='temperature for sampling')
    parser.add_argument('--model_path', type=str, help='path to model_checkpoint.pt')

    # Parse the command-line arguments
    args = parser.parse_args()

    # define hyperparameters
    DIM = 512
    SEQUENCE_LENGTH = 2048
    DEPTH = 12
    FLASH_ATTN = True

    # check if GPU is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # create the tokenizer
    tokenizer = REMIPlus()

    # create the model
    model = PaLM(
        num_tokens=tokenizer.len,
        dim = DIM,
        depth = DEPTH,
        flash_attn=FLASH_ATTN
    ).to(device)

    # load the model checkpoint
    model.load(args.model_path)

    # Call the generate_sample function with the parsed arguments
    generate_sample(model, tokenizer, args.output_path, SEQUENCE_LENGTH, args.temperature)
