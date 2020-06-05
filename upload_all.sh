set -e
edge-impulse-uploader --label noise --category training data/splits/**/training/noise/*.wav
edge-impulse-uploader --label titmouse --category training data/splits/**/training/titmouse/*.wav
edge-impulse-uploader --label mourningdove --category training data/splits/**/training/mourningdove/*.wav

edge-impulse-uploader --label noise --category testing data/splits/**/testing/noise/*.wav
edge-impulse-uploader --label titmouse --category testing data/splits/**/testing/titmouse/*.wav
edge-impulse-uploader --label mourningdove --category testing data/splits/**/testing/mourningdove/*.wav