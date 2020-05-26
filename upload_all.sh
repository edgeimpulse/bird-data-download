edge-impulse-uploader --label noise --category training data/splits/titmouse-noise-100/training/noise/*.wav
edge-impulse-uploader --label titmouse --category training data/splits/titmouse-noise-100/training/titmouse/*.wav

edge-impulse-uploader --label noise --category testing data/splits/titmouse-noise-100/testing/noise/*.wav
edge-impulse-uploader --label titmouse --category testing data/splits/titmouse-noise-100/testing/titmouse/*.wav