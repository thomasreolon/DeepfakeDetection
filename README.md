# Fake detention in videos

:construction::construction::construction: currently under development :construction::construction::construction:

This project uses [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace.git) to analyze faces in videos.
Then uses the features suggested by [Agarwal et al.](https://openaccess.thecvf.com/content_CVPRW_2019/papers/Media%20Forensics/Agarwal_Protecting_World_Leaders_Against_Deep_Fakes_CVPRW_2019_paper.pdf) to differentiate real videos of famous people from fakes.

---

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

### videoanalizer module

This module can extract the 190 feature array described in "Protecting World Leaders Against Deep Fakes" and train your classifier.
The following snippet shows how to do this.

```python
from videoanalizer import VideoAnalizer

# Videos To Process
files = ['a.jpg', '../folder/imgs/obama.png']

# Analysis Using Openface + Correlation Extraction
vd = VideoAnalizer()
arrays, _ = vd.process_video(files=files)


print(arrays)  # [[0.42 ,0.39, ..., -0.17], [0.6 ,0.343, ..., -0.3443]]
```

This module is an interface to process videos with OpenFace easily.
The following snippet shows how pass a video to OpenFace and extract the features we are interested in.

### openface module

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
