import argparse


import torch
from torchtext.data import Example


from models.definitions.transformer_model import Transformer
from utils.data_utils import get_datasets_and_vocabs, get_masks_and_count_tokens_src, DatasetType, LanguageDirection
from utils.constants import *
from utils.visualization_utils import visualize_attention
from utils.decoding_utils import greedy_decoding, get_beam_decoder, DecodingMethod
from utils.utils import print_model_metadata
from utils.resource_downloader import download_models


# Super easy to add translation for a batch of sentences passed as a .txt file for example
def translate_a_single_sentence(translation_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # checking whether you have a GPU

    # Step 1: Prepare the field processor (tokenizer, numericalizer)
    _, _, src_field_processor, trg_field_processor = get_datasets_and_vocabs(
        translation_config['dataset_path'],
        translation_config['language_direction'],
        translation_config['dataset_name'] == DatasetType.IWSLT.name
    )
    assert src_field_processor.vocab.stoi[PAD_TOKEN] == trg_field_processor.vocab.stoi[PAD_TOKEN]
    pad_token_id = src_field_processor.vocab.stoi[PAD_TOKEN]  # needed for constructing masks

    # Step 2: Prepare the model
    baseline_transformer = Transformer(
        model_dimension=BASELINE_MODEL_DIMENSION,
        src_vocab_size=len(src_field_processor.vocab),
        trg_vocab_size=len(trg_field_processor.vocab),
        number_of_heads=BASELINE_MODEL_NUMBER_OF_HEADS,
        number_of_layers=BASELINE_MODEL_NUMBER_OF_LAYERS,
        dropout_probability=BASELINE_MODEL_DROPOUT_PROB,
        log_attention_weights=True
    ).to(device)

    model_path = os.path.join(BINARIES_PATH, translation_config['model_name'])
    if not os.path.exists(model_path):
        print(f'Model {model_path} does not exist, attempting to download.')
        model_path = download_models(translation_config)

    model_state = torch.load(model_path)
    print_model_metadata(model_state)
    baseline_transformer.load_state_dict(model_state["state_dict"], strict=True)
    baseline_transformer.eval()

    # Step 3: Prepare the input sentence
    source_sentence = translation_config['source_sentence']
    ex = Example.fromlist([source_sentence], fields=[('src', src_field_processor)])  # tokenize the sentence

    source_sentence_tokens = ex.src
    print(f'Source sentence tokens = {source_sentence_tokens}')

    # Numericalize and convert to cuda tensor
    src_token_ids_batch = src_field_processor.process([source_sentence_tokens], device)

    with torch.no_grad():
        # Step 4: Optimization - compute the source token representations only once
        src_mask, _ = get_masks_and_count_tokens_src(src_token_ids_batch, pad_token_id)
        src_representations_batch = baseline_transformer.encode(src_token_ids_batch, src_mask)

        # Step 5: Decoding process
        if translation_config['decoding_method'] == DecodingMethod.GREEDY:
            target_sentence_tokens = greedy_decoding(baseline_transformer, src_representations_batch, src_mask, trg_field_processor)
        else:
            beam_decoding = get_beam_decoder(translation_config)
            target_sentence_tokens = beam_decoding(baseline_transformer, src_representations_batch, src_mask, trg_field_processor)
        print(f'Translation | Target sentence tokens = {target_sentence_tokens}')

        # Step 6: Potentially visualize the encoder/decoder attention weights
        if translation_config['visualize_attention']:
            visualize_attention(baseline_transformer, source_sentence_tokens, target_sentence_tokens)


if __name__ == "__main__":
    #
    # modifiable args - feel free to play with these (only small subset is exposed by design to avoid cluttering)
    #
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_sentence", type=str, help="source sentence to translate into target", default="How are you doing today?")
    parser.add_argument("--model_name", type=str, help="transformer model name", default=r'iwslt_e2g.pth')

    # Keep these 2 in sync with the model you pick via model_name
    parser.add_argument("--dataset_name", type=str, choices=['IWSLT', 'WMT14'], help='which dataset to use for training', default=DatasetType.IWSLT.name)
    parser.add_argument("--language_direction", type=str, choices=[el.name for el in LanguageDirection], help='which direction to translate', default=LanguageDirection.E2G.name)

    # Cache files and datasets are downloaded here during training, keep them in sync for speed
    parser.add_argument("--dataset_path", type=str, help='download dataset to this path', default=DATA_DIR_PATH)

    # Decoding related args
    parser.add_argument("--decoding_method", type=str, help="pick between different decoding methods", default=DecodingMethod.GREEDY)
    parser.add_argument("--beam_size", type=int, help="used only in case decoding method is chosen", default=4)
    parser.add_argument("--length_penalty_coefficient", type=int, help="length penalty for the beam search", default=0.6)

    parser.add_argument("--visualize_attention", type=bool, help="should visualize encoder/decoder attention", default=False)
    args = parser.parse_args()

    # Wrapping training configuration into a dictionary
    translation_config = dict()
    for arg in vars(args):
        translation_config[arg] = getattr(args, arg)

    # Translate the given source sentence
    translate_a_single_sentence(translation_config)
