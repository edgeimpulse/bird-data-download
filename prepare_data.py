#!/usr/bin/env python3

import os
import io
import numpy
import random
import re
import hashlib
import math
import array
import wget
import shutil
import statistics
import subprocess
import requests
from functools import reduce
from shutil import copyfile
from os.path import expanduser
from pathlib import Path
from pydub import AudioSegment

# Set random seed
RANDOM_SEED = 1337
random.seed(RANDOM_SEED)
numpy.random.seed(RANDOM_SEED)

DATASET_SPEECH_ARCHIVE_NAME = "speech_commands_v0.02.tar.gz"
DATASET_SPEECH_URL = "http://download.tensorflow.org/data/" + DATASET_SPEECH_ARCHIVE_NAME

OUTPUT_PATH = os.path.join(os.getcwd(), "data")
SPLIT_OUTPUT_PATH = os.path.join(OUTPUT_PATH, "splits")

DATASET_SPEECH_ARCHIVE_PATH = os.path.join(OUTPUT_PATH, DATASET_SPEECH_ARCHIVE_NAME)
DATASET_SPEECH_PATH = os.path.join(OUTPUT_PATH, "speech_commands_v0.02")

DATASET_BIRDS_PATH = os.path.join(OUTPUT_PATH, 'birds')

DATASET_SPEECH_BACKGROUND_NOISE_DIR_NAME = "_background_noise_"
DATASET_SPEECH_NOISE_PATH = os.path.join(DATASET_SPEECH_PATH, DATASET_SPEECH_BACKGROUND_NOISE_DIR_NAME)

DATASET_SNSD_BACKGROUND_NOISE_TRAIN_SVN_URL = "https://github.com/microsoft/MS-SNSD/trunk/noise_train"
DATASET_SNSD_BACKGROUND_NOISE_TEST_SVN_URL = "https://github.com/microsoft/MS-SNSD/trunk/noise_test"

DATASET_SNSD_PATH = os.path.join(OUTPUT_PATH, "MS-SNSD")
DATASET_SNSD_TRAIN_PATH = os.path.join(DATASET_SNSD_PATH, "train")
DATASET_SNSD_TEST_PATH = os.path.join(DATASET_SNSD_PATH, "test")

OUTPUT_NOISE_PATH = os.path.join(OUTPUT_PATH, "noise")

# The number of frames per second of audio
EXPECTED_FRAMES = 16000
# The percentage of data reserved for testing
TEST_SPLIT = 10
# Used by `which_set()` from speech commands dataset
MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M
# How long each file can be max
AUDIO_SPLIT_MS = 30000

# Hold paths to data
global_data_noise = {}
global_data_test = {}
global_data_classes = {}

def create_output_directories():
    if not os.path.exists(OUTPUT_PATH):
        print("Creating output directory")
        Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=False)

    if not os.path.exists(SPLIT_OUTPUT_PATH):
        print("Creating split directory")
        Path(SPLIT_OUTPUT_PATH).mkdir(parents=True, exist_ok=False)

def download_and_extract_speech_commands_dataset():
    # Check if the dataset has been downloaded
    if os.path.exists(DATASET_SPEECH_ARCHIVE_PATH):
        print("Dataset archive already downloaded")
    else:
        print("Downloading dataset, will take a moment")
        wget.download(DATASET_SPEECH_URL, DATASET_SPEECH_ARCHIVE_PATH)
        print("\n")
        print("Finished downloading:", DATASET_SPEECH_ARCHIVE_PATH)

    if os.path.exists(DATASET_SPEECH_PATH):
        print("Dataset archive already extracted")
    else:
        print("Extracting dataset")
        shutil.unpack_archive(DATASET_SPEECH_ARCHIVE_PATH, DATASET_SPEECH_PATH)
        print("Finished extracting dataset")

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

    recordings = get_bird_recording_info(genus, species)

    for recording in recordings:
        recording_url = 'https:' + recording['file']
        r = requests.get(recording_url, allow_redirects=True)

        file_name = recording['file-name']

        s = io.BytesIO(r.content)
        audio = AudioSegment.from_file(s, format="mp3")
        # Split audio
        MAX_MS = 30000
        if len(audio) <= MAX_MS:
            new_name = label + '.' + os.path.splitext(file_name)[0] + '.wav'
            audio.export(os.path.join(label_path, new_name), format="wav")
        else:
            for pos in range(0, len(audio), MAX_MS):
                section = audio[pos:pos + MAX_MS]
                new_name = label + '.' + pos + '.' + os.path.splitext(file_name)[0] + '.wav'
                section.export(os.path.join(label_path, new_name), format="wav")

# Move files from one dir to another, deleting the original dir and
# prefixing the files with a string
def bulk_move_and_rename_files(original_dir, target_dir, prefix):
    files = os.listdir(original_dir)

    for file_name in files:
        if not file_name.endswith(".wav"):
            continue
        original_path = os.path.join(original_dir, file_name)
        new_name = prefix + "." + file_name
        new_path = os.path.join(target_dir, new_name)
        os.rename(original_path, new_path)

    shutil.rmtree(original_dir, ignore_errors=True)

# The SNSD files are split into train and test. Since we're combining them with
# our other dataset and creating our own splits, we'll combine them (which requires
# renaming since the file names collide)
def combine_snsd_files():
    bulk_move_and_rename_files(DATASET_SNSD_TRAIN_PATH, DATASET_SNSD_PATH, "orig_train")
    bulk_move_and_rename_files(DATASET_SNSD_TEST_PATH, DATASET_SNSD_PATH, "orig_test")

# Downloads extra noise files from this dataset https://github.com/microsoft/MS-SNSD
def download_and_extract_ms_snsd():
    # Check if the folder exists
    if os.path.exists(DATASET_SNSD_PATH):
        print("SNSD noise already downloaded")
    else:
        print("Downloading SNSD noise, ignoring proscribed split")
        Path(DATASET_SNSD_PATH).mkdir(parents=True, exist_ok=False)
        subprocess.call(["/usr/bin/svn", "checkout", DATASET_SNSD_BACKGROUND_NOISE_TRAIN_SVN_URL, DATASET_SNSD_TRAIN_PATH])
        subprocess.call(["/usr/bin/svn", "checkout", DATASET_SNSD_BACKGROUND_NOISE_TEST_SVN_URL, DATASET_SNSD_TEST_PATH])
        print("Combining SNSD files and deleting originals")
        combine_snsd_files()

# Splits audio into as many 1 second windows as possible
def split_audio(file_name, source_directory):
    print("Splitting " + file_name)
    audio = AudioSegment.from_wav(os.path.join(source_directory, file_name))
    output_dir = os.path.join(OUTPUT_NOISE_PATH, file_name)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    count = 0
    for start in range(0, len(audio), AUDIO_SPLIT_MS):
        end = start + AUDIO_SPLIT_MS
        window = audio[start:end]
        window.export(os.path.join(output_dir, file_name + "." + str(start) + ".wav"), format="wav")
        count += 1

    print("Got", count, " ", AUDIO_SPLIT_MS, "ms files from", file_name)

def split_noise_files(source_directory):
    # Chop all noise files into windows if necessary
    for file_name in os.listdir(source_directory):
        if file_name.endswith(".wav"):
            split_audio(file_name, source_directory)

def process_noise_files():
    # Chop all noise files into windows if necessary
    if not os.path.exists(OUTPUT_NOISE_PATH):
        split_noise_files(DATASET_SPEECH_NOISE_PATH)
        split_noise_files(DATASET_SNSD_PATH)
    else:
        print("Noise files already split")

# Count how many samples we have for a given class
def count_samples(path, class_name):
    class_path = os.path.join(path, class_name)
    if not os.path.isdir(class_path):
        raise Exception("Class name '" + class_name + "' is not in the dataset")

    return len(os.listdir(class_path))

# Count how many total samples of noise we have
def count_noise():
    sample_counts = [len(global_data_noise[style]) for style in global_data_noise]
    return sum(sample_counts)

def copy_files_to_output(file_paths, output_directory, split_name, class_name, check_exists=True):
    output_path = os.path.join(SPLIT_OUTPUT_PATH, output_directory, split_name, class_name)
    if check_exists and os.path.exists(output_path):
        raise Exception("Directory for " + class_name + " already exists")
    # Create the directory
    Path(output_path).mkdir(parents=True, exist_ok=not check_exists)

    # Copy the selected files, renaming on the way
    for file_path in file_paths:
        # Come up with new file name
        file_name = os.path.basename(file_path)
        new_name = class_name + "." + file_name
        new_path = os.path.join(output_path, new_name)
        copyfile(file_path, new_path)

# Copies the correct number of noise samples to the output
def select_and_copy_noise(number_to_select, output_directory):
    print("Attempting to select {} noise samples; actual number may be lower due to rounding in calculation".format(number_to_select))
    # Copy test set
    copy_files_to_output(global_data_test['noise'], output_directory, "testing", "noise", check_exists=False)

    # Get list of tuples showing total number of samples for each noise type
    noise_styles = [(style, len(global_data_noise[style])) for style in global_data_noise]

    total_noise_samples = count_noise()

    # Determine percentage of total represented by each style
    noise_style_percentages = [(style[0], (100 / total_noise_samples) * style[1])
                                for style in noise_styles]

    # Determine number of samples we should take for each style
    noise_style_sample_counts = [(style[0], math.floor((number_to_select / 100) * style[1]))
                            for style in noise_style_percentages]

    total_selected = 0
    # Select the correct number for each style and copy the files
    for style in noise_style_sample_counts:
        total_selected += style[1]
        sample = random.sample(global_data_noise[style[0]], style[1])
        copy_files_to_output(sample, output_directory, "training", "noise", check_exists=False)
    print("Selected {} samples".format(total_selected))

def list_class_directories():
    return [name for name in os.listdir(DATASET_BIRDS_PATH)
                    if os.path.isdir(os.path.join(DATASET_BIRDS_PATH, name))]

# Creates a set of files
def create_new_split(class_names, output_directory=None, noise=True, sample_percentage=100):

    # Build a directory name based on the params
    if output_directory == None:
        output_directory = ("-".join(class_names)
                            + ("-noise" if noise else "")
                            + "-" + str(sample_percentage))

    if os.path.exists(os.path.join(SPLIT_OUTPUT_PATH, output_directory)):
        print("Directory already exists for '{}', skipping".format(output_directory))
        return output_directory

    # Find out the smallest class
    lowest_class_count = min([len(global_data_classes[class_name]) for class_name in global_data_classes if class_name in class_names])
    # In case noise is the smallest class
    if noise:
        lowest_class_count = min(lowest_class_count, count_noise())

    print("Smallest class has", lowest_class_count, "samples")

    # Calculate the number of samples to use
    if sample_percentage != 100:
        lowest_class_count = math.floor((lowest_class_count / 100) * sample_percentage)

    print("Using {}%, will select {} samples".format(sample_percentage, lowest_class_count))

    if noise:
        select_and_copy_noise(lowest_class_count, output_directory)

    for name in class_names:
        # Select and copy training samples
        files = global_data_classes[name]
        print("Class {}: Copying {} of {} training samples".format(name, lowest_class_count, len(files)))
        selection = random.sample(files, lowest_class_count)
        copy_files_to_output(selection, output_directory, "training", name)
        # Copy all test files
        test_files = global_data_test[name]
        print("Class {}: Copying {} test samples".format(name, len(test_files)))
        copy_files_to_output(test_files, output_directory, "testing", name)

    return output_directory


def create_noise_splits(data_noise, data_test):
    noise_test_data = []
    training_sample_count = 0
    testing_sample_count = 0
    # For each noise style, take random TEST_SPLIT% and assign to test set
    for style in os.listdir(OUTPUT_NOISE_PATH):
        # List all the noise samples
        style_path = os.path.join(OUTPUT_NOISE_PATH, style)
        # Get the full file paths
        files = [os.path.join(style_path, file_name) for file_name in os.listdir(style_path)]
        random.shuffle(files)
        # Determine how many should be in the training split
        training_count = math.floor((len(files) / 100) * (100 - TEST_SPLIT))
        # Split the data
        training_files = files[:training_count]
        test_files = files[training_count:]
        # Update the totals
        training_sample_count += len(training_files)
        testing_sample_count += len(test_files)
        # Store the splits
        data_noise[style] = training_files
        noise_test_data.extend(test_files)

    # Save in a global noise list
    data_test['noise'] = noise_test_data

    print("Total {} noise files split into {} training and {} test".format(
        training_sample_count + testing_sample_count, training_sample_count, testing_sample_count))

def create_class_splits(data_classes, data_test):
    # For each class, take random TEST_SPLIT% and assign to test set
    for class_name in list_class_directories():
        # Set up places to put the data
        data_classes[class_name] = []
        data_test[class_name] = []
        # Assign each file to the right place
        files = [os.path.join(DATASET_SPEECH_PATH, class_name, file_name) for file_name in os.listdir(os.path.join(DATASET_SPEECH_PATH, class_name))]
        random.shuffle(files)
        split_point = math.ceil(len(files) / (100 / TEST_SPLIT))
        data_classes[class_name].extend(files[:split_point])
        data_test[class_name].extend(files[split_point:])

def init():
    # Set up dataset and output directories
    create_output_directories()
    download_and_extract_speech_commands_dataset()
    download_and_extract_ms_snsd()
    download_bird_songs('baeolophus', 'inornatus', 'titmouse')
    process_noise_files()

    # # Set up training/test splits
    create_noise_splits(global_data_noise, global_data_test)
    create_class_splits(global_data_classes, global_data_test)

if __name__ == "__main__":
    init()
    # Create datasets with varying amounts of data
    # Note: All datasets will be balanced, so their size depends on the size of their smallest class (including noise).
    #       The same test set is used for each dataset, independent of size (for unknown, the test set's content depends
    #       on the selected classes, since unknown is made up of unselected classes).
    percentages = [100]

    for percentage in percentages:
        create_new_split(["yes", "no"], sample_percentage=percentage)
        create_new_split(["yes", "no"], sample_percentage=percentage)
        create_new_split(["yes", "no"], sample_percentage=percentage)