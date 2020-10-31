# todo: add greedy and beam decoding mechanisms and use the pretrained models to translate

import argparse


import torch
from torchtext.data import Example


from models.definitions.transformer_model import Transformer
from utils.data_utils import build_datasets_and_vocabs
from utils.constants import *


def greedy_decode(model, sentence):
    print('todo')


def translate(translation_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # checking whether you have a GPU

    # todo: redesign so that I can import only things I need
    _, _, src_field_processor, trg_field_processor = build_datasets_and_vocabs(use_caching_mechanism=True)
    assert src_field_processor.vocab.stoi[PAD_TOKEN] == trg_field_processor.vocab.stoi[PAD_TOKEN]

    baseline_transformer = Transformer(
        model_dimension=BASELINE_MODEL_DIMENSION,
        src_vocab_size=len(src_field_processor.vocab),
        trg_vocab_size=len(trg_field_processor.vocab),
        number_of_heads=BASELINE_MODEL_NUMBER_OF_HEADS,
        number_of_layers=BASELINE_MODEL_NUMBER_OF_LAYERS,
        dropout_probability=BASELINE_MODEL_DROPOUT_PROB
    ).to(device)

    model_path = os.path.join(CHECKPOINTS_PATH, translation_config['model_name'])
    assert os.path.exists(model_path), f'Could not find the model {model_path}. You first need to train the transformer.'

    model_state = torch.load(model_path)
    baseline_transformer.load_state_dict(model_state["state_dict"], strict=True)
    baseline_transformer.eval()

    german_sentence = translation_config['german_sentence']
    data = [Example.fromlist([german_sentence], fields=('src', src_field_processor))]
    batch = [getattr(x, 'src') for x in data]

    src_field_processor.process(batch, device)


if __name__ == "__main__":
    #
    # modifiable args - feel free to play with these (only small subset is exposed by design to avoid cluttering)
    #
    parser = argparse.ArgumentParser()
    parser.add_argument("--german_sentence", type=str, help="German sentence to translate into English", default="Ich bin.")
    parser.add_argument("--model_name", type=str, help="Transformer model name", default=r'transformer_ckpt_epoch_1.pth')
    args = parser.parse_args()

    # Wrapping training configuration into a dictionary
    translation_config = dict()
    for arg in vars(args):
        translation_config[arg] = getattr(args, arg)

    # translate the given german sentence
    translate(translation_config)
