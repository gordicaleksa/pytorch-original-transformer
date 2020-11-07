import zipfile
import os


from torch.hub import download_url_to_file


from .constants import BINARIES_PATH


IWSLT_ENGLISH_TO_GERMAN_MODEL_URL = r'https://www.dropbox.com/s/a6pfo6t9m2dh1jq/iwslt_e2g.pth?dl=1'
IWSLT_GERMAN_TO_ENGLISH_MODEL_URL = r'https://www.dropbox.com/s/dgcd4xhwig7ygqd/iwslt_g2e.pth?dl=1'


# Not yet trained
WMT14_ENGLISH_TO_GERMAN_MODEL_URL = None
WMT14_GERMAN_TO_ENGLISH_MODEL_URL = None


DOWNLOAD_DICT = {
    'iwslt_e2g': IWSLT_ENGLISH_TO_GERMAN_MODEL_URL,
    'iwslt_g2e': IWSLT_GERMAN_TO_ENGLISH_MODEL_URL,
    'wmt14_e2g': WMT14_ENGLISH_TO_GERMAN_MODEL_URL,
    'wmt14_g2e': WMT14_GERMAN_TO_ENGLISH_MODEL_URL
}


download_choices = list(DOWNLOAD_DICT.keys())


def download_models(translation_config):
    # Step 1: Form the key
    language_direction = translation_config['language_direction'].lower()
    dataset_name = translation_config['dataset_name'].lower()
    key = f'{dataset_name}_{language_direction}'

    # Step 2: Check whether this model already exists
    model_name = f'{key}.pth'
    model_path = os.path.join(BINARIES_PATH, model_name)
    if os.path.exists(model_path):
        print(f'No need to download, found model {model_path} that was trained on {dataset_name} for language direction {language_direction}.')
        return model_path

    # Step 3: Download the resource to local filesystem
    remote_resource_path = DOWNLOAD_DICT[key]
    if remote_resource_path is None:  # handle models which I've not provided URLs for yet
        print(f'No model found that was trained on {dataset_name} for language direction {language_direction}.')
        exit(0)

    print(f'Downloading from {remote_resource_path}. This may take a while.')
    download_url_to_file(remote_resource_path, model_path)

    return model_path

