## Bird audio dataset creation script

This script will download bird audio recordings from https://www.xeno-canto.org, split them into 8 second files, and create training and test datasets of the desired size.

A noise class will be added using background noise downloaded from the following sources:

- https://github.com/microsoft/MS-SNSD
- https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html

Training sets will be balanced, so each class will have the same number of samples. Test sets include all available remaining samples.

### Usage

To download the default audio dataset (oak titmouse and mourning dove), run the script `prepare_data.py` using python 3. You will need to install any dependencies that are missing.

Audio for different birds can be downloaded by editing the `init` and `if __name__ == "__main__":` sections of the file to specify the desired species.

### Uploading to Edge Impulse

The script in `upload_all.sh` provides an example of how to upload the data to Edge Impulse for the default dataset. You will need to modify this if you add different birds.

### Note

This is a rough script provided for experimentation only.
