import argparse


import torch
from torchtext.data import Example


from models.definitions.transformer_model import Transformer
from utils.data_utils import build_datasets_and_vocabs, build_masks_and_count_tokens
from utils.constants import *


def translate_a_single_sentence(translation_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # checking whether you have a GPU

    # Step 1: Prepare the field processor (tokenizer, numericalizer)
    _, _, src_field_processor, trg_field_processor = build_datasets_and_vocabs(translation_config['dataset_path'])
    assert src_field_processor.vocab.stoi[PAD_TOKEN] == trg_field_processor.vocab.stoi[PAD_TOKEN]
    pad_token_id = src_field_processor.vocab.stoi[PAD_TOKEN]  # needed for constructing masks

    # Step 2: Prepare the model
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

    # Step 3: Prepare the input sentence and output prompt
    german_sentence = translation_config['german_sentence']
    ex = Example.fromlist([german_sentence], fields=[('src', src_field_processor)])  # tokenize the sentence
    src_token_ids_batch = src_field_processor.process([ex.src], device)  # numericalize and convert to cuda tensor

    english_sentence_tokens = [BOS_TOKEN]  # initial prompt - beginning/start of the sentence token
    trg_token_ids_batch = torch.tensor([[trg_field_processor.vocab.stoi[token] for token in english_sentence_tokens]], device=device)

    # Decoding could be further optimized to cache old token activations because they can't look ahead and so
    # adding a newly predicted token won't change old token's activations.
    #
    # Example: we input <s> and do a forward pass. We get intermediate activations for <s> and at the output at position
    # 0, after the doing linear layer we get e.g. token <I>. Now we input <s>,<I> but <s>'s activations will remain
    # the same. Similarly say we now got <am> at output position 1, in the next step we input <s>,<I>,<am> and so <I>'s
    # activations will remain the same as it only looks at/attends to itself and to <s> and so forth.
    with torch.no_grad():
        while True:
            src_mask, trg_mask, _, _ = build_masks_and_count_tokens(src_token_ids_batch, trg_token_ids_batch, pad_token_id, device)
            predicted_log_distributions = baseline_transformer(src_token_ids_batch, trg_token_ids_batch, src_mask, trg_mask)

            # todo: add beam decoding mechanism
            # todo: optimize: don't run the whole model run the encoding part only once
            # Greedy decoding
            most_probable_word_index = torch.argmax(predicted_log_distributions[-1]).cpu().numpy()
            predicted_word = trg_field_processor.vocab.itos[most_probable_word_index]
            print(predicted_word)

            if predicted_word == EOS_TOKEN:
                english_sentence_tokens.append(EOS_TOKEN)
                break

            english_sentence_tokens.append(predicted_word)
            trg_token_ids_batch = torch.tensor([[trg_field_processor.vocab.stoi[token] for token in english_sentence_tokens]], device=device)

    print(f'translation = {english_sentence_tokens}')


if __name__ == "__main__":
    #
    # modifiable args - feel free to play with these (only small subset is exposed by design to avoid cluttering)
    #
    parser = argparse.ArgumentParser()
    parser.add_argument("--german_sentence", type=str, help="German sentence to translate into English", default="Ich bin.")
    parser.add_argument("--model_name", type=str, help="Transformer model name", default=r'transformer_ckpt_epoch_1.pth')
    parser.add_argument("--dataset_path", type=str, help='save dataset to this path', default=os.path.join(os.path.dirname(__file__), '.data'))
    args = parser.parse_args()

    # Wrapping training configuration into a dictionary
    translation_config = dict()
    for arg in vars(args):
        translation_config[arg] = getattr(args, arg)

    # Translate the given german sentence
    translate_a_single_sentence(translation_config)
