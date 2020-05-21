#!/usr/bin/env python3

import os
import io
import subprocess
from pathlib import Path
from pydub import AudioSegment
import requests

DATASET_BIRDS_PATH = 'birds'

def get_bird_recording_info(genus, species, page = 1):
    base_url = 'https://www.xeno-canto.org/api/2/recordings?query='
    query = {
        'gen': genus,
        'q_gt': 'C', # Quality, from A-E
        'type': 'song'
    }

    query_pairs = []
    for key, value in query.items():
        query_pairs.append(key + ':' + str(value))

    query_string = '+'.join(query_pairs)

    full_url = base_url + query_string + '&page=' + str(page)
    result = requests.get(full_url)
    print(full_url)

    json = result.json()
    num_pages = json['numPages']

    def is_acceptable(recording):
        # Only the species we want
        if recording['sp'].lower() != species:
            return False
        # No other birds in the background
        if len(recording['also']) != 1 and recording['also'][0] != '':
            return False
        return True

    recordings = [recording for recording in json['recordings'] if is_acceptable(recording)]

    if page < num_pages:
        recordings.extend(get_bird_recording_info(genus, species, page + 1))

    return recordings


def download_bird_songs(genus, species, label):
    label_path = os.path.join(DATASET_BIRDS_PATH, label)
    if os.path.exists(label_path):
        print(f'Already got {label}')
        return
    else:
        print(f'Downloading {label}')
        Path(label_path).mkdir(parents=True, exist_ok=False)

    recordings = get_bird_recording_info('baeolophus', 'inornatus')

    for recording in recordings:
        recording_url = 'https:' + recording['file']
        r = requests.get(recording_url, allow_redirects=True)

        file_name = recording['file-name']
        new_name = label + '.' + os.path.splitext(file_name)[0] + '.wav'

        s = io.BytesIO(r.content)
        AudioSegment.from_file(s, format="mp3").export(os.path.join(label_path, new_name), format="wav")

# Download background noise
DATASET_SNSD_BACKGROUND_NOISE_TRAIN_SVN_URL = "https://github.com/microsoft/MS-SNSD/trunk/noise_train"
DATASET_SNSD_BACKGROUND_NOISE_TEST_SVN_URL = "https://github.com/microsoft/MS-SNSD/trunk/noise_test"
DATASET_SNSD_PATH = 'noise'
DATASET_SNSD_TRAIN_PATH = os.path.join(DATASET_SNSD_PATH, 'train')
DATASET_SNSD_TEST_PATH = os.path.join(DATASET_SNSD_PATH, 'test')

# Downloads extra noise files from this dataset https://github.com/microsoft/MS-SNSD
def download_and_extract_ms_snsd():
    # Check if the folder exists
    if os.path.exists(DATASET_SNSD_PATH):
        print("SNSD noise already downloaded")
    else:
        print("Downloading SNSD noise")
        Path(DATASET_SNSD_PATH).mkdir(parents=True, exist_ok=False)
        subprocess.call(["/usr/bin/svn", "checkout", DATASET_SNSD_BACKGROUND_NOISE_TRAIN_SVN_URL, DATASET_SNSD_TRAIN_PATH])
        subprocess.call(["/usr/bin/svn", "checkout", DATASET_SNSD_BACKGROUND_NOISE_TEST_SVN_URL, DATASET_SNSD_TEST_PATH])

download_and_extract_ms_snsd()
download_bird_songs('baeolophus', 'inornatus', 'titmouse')
