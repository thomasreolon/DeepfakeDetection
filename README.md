# Fake detention in videos

:construction::construction::construction: currently under development :construction::construction::construction:

This project uses [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace.git) to analyze faces in videos.
... future ...

---

**TODO**s:
:heavy_check_mark: interface to OpenFace
:heavy_multiplication_x: detect fake videos from real ones

## Project structure:

    src
    ├── videoanalyzer         [simplify the extraction of the 190 correlation features]
    |   |
    |   ├── openface          [package to interact with OpenFace]
    |   │   ├── installer     [script that can install OpenFace automatically]
    |   │   └── openfaceapi   [class that can process images/videos]
    |
    |
    ├── train.py              [trains a 2classes svm classifier]
    ├── predict.py            [predicts if video is fake]

---

## Usage

very brief snippets on how to start up

### openface package

```python
from openface.openfaceAPI import OpenFaceAPI
import openface.parts as parts


"""pt.1 Analyze some images"""


# instantiate the API
# if you haven't installed OpenFace it will ask you to. (takes a while)
api = OpenFaceAPI()

# files that we want to process
files = ['a.jpg', '../folder/imgs/obama.png']

# get the result as a map: (filename) --> DataExtractor for that file
results = api.process_img(files=files)


"""pt.2 Extract the features that we want"""

# select the file we want
extractor = results['obama.png']


# declare what we wanted
parts_of_face = [
    parts.FACE_CHIN,
    'AU01_c',
    parts.EYES_LEFT_PUPIL
]

# an extractor esposes many methods to select the features
#   eg. get_frame, get_face, ...
features = extractor.get_features(parts_of_face).to_array()

print(features)  # {0:np.array(features_for_face_0), 1:np.array(features_for_face_1), ...}
```
