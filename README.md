# Fake detention in videos

This project uses [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace.git) to analyze faces in videos and the ideas from [Agarwal et al.](https://openaccess.thecvf.com/content_CVPRW_2019/papers/Media%20Forensics/Agarwal_Protecting_World_Leaders_Against_Deep_Fakes_CVPRW_2019_paper.pdf) to train robust classifiers of a specific person.
The project contains a wrapper to openface and a module that can train models to detect deepfakes.

This notebook shows how to install videoanalizer and how to train a simple model to detect fake videos of Obama: [Notebook](https://colab.research.google.com/drive/1S2W_YrA8aGveSOdao07hveSkOzuBIRBk?usp=sharing)
(this is only an example... Do NOT run the notebook, it takes more than 1 hour to install OpenFace and another whole hour to process Obama's videos)

---

#### Obama (real - fake)

<img src="https://media.giphy.com/media/K9YzMLteKq6sg5VJMh/giphy.gif" width="400" height="300" /> <img src="https://media.giphy.com/media/h9kD101j2VEXIjw9eY/giphy.gif" width="400" height="300" />

## Project structure:

    SIV_project
    ├── src
    |    ├── videoanalizer                [gives functions to process videos]
    |    |    ├── openface                [package that interact with OpenFace]
    |    |    |    ├── errors             [library which contains errors]
    |    |    |    ├── face_comparator    [script to compare the similarity between two faces]
    |    |    │    ├── installer          [script that can install OpenFace automatically]
    |    |    │    ├── openfaceapi        [class that can process images/videos]
    |    |    |    ├── output_extractor   [script to extract features from the openface's csv]
    |    |    |    ├── parts              [library which contains information about face's parts]
    |    |    |    └── (OpenFace)         [where OpenFace will be installed]
    |    |    |
    |    |    ├── videoanalizer           [A class that implements functions to process videos as described by Agarwal et al.]
    |    |    ├── classifier              [Implementing OneClassSVM with the best parameters]
    |    |    ├── samplesextractor        [splits the output of a processed video in samples]
    |    |    ├── video_edits             [generate a video with landmarks]
    |    |    ├── plots                   [visualize the videos in a 2D plane]
    |    |    └── covariance              [extract the 190 features & the additional 60]
    |    |
    |    |
    |    ├── download_dataset       [download our dataset]
    |    ├── create_landmarks       [create the video with the landmark real or fake based on the predictions]
    |    ├── create_plot            [some useful plots for data analysis (PCA, LDA)]
    |    └── train_a_model          [train and test the final classification system]
    |
    |
    └── (test_data)                 [dataset of video for training and testing (run download_dataset.py)]

---

## Installation

```
    git clone https://github.com/Moreno98/SIV_project.git
    cd SIV_project
    python3 download_dataset.py
```

This script will download our custom dataset. Moreover, during the first run the program will ask you to install automatically OpenFace in the project folder, otherwise you can add your current path to OpenFace manually in the code.

## Usage

### Use landmarks

This project also offer a script to attach to a video the predicted landmarks (using our system) using the landmark function offered by openface.
In orther to run this script:

```
    cd src
    python3 create_landmarks.py
```

The videos will be saved under `output/`.  
This is the result predicting videos about Elon Musk:

#### Elon Musk (real - fake)

<img src="https://media.giphy.com/media/h31mo3j1UgSc8XE5Cx/giphy.gif" width="400" height="300" /> <img src="https://media.giphy.com/media/keuDEb10tk9Jnkpwi0/giphy.gif" width="400" height="300" />

### Train and test the classification system

After downloading the dataset (download_dataset) you may use the classification system running:

```
    cd src
    python3 train_a_model.py
```

The performance results will be saved under `output/final_results.txt`.

### General Usage 1

From /src you can use a relative import to import videoanalizer (1), you can then create an instance of VideoAnalizer (2).
VideoAnalizer offers some function to process data, for example:

- train_OneClassSVM | takes as input folder of real videos of a person, returns a trained classifier of deepfakes (3)
- process_video | takes as input some videos, returns a tuple containing a list of vectors (vector is 190 featues) and a list that says which video produced each vector (4)
- split_train_test | takes a folder process it and splits the results in taining and test (5)
- plot_features | takes as input folders of videos, outputs the result of PCA/LDA on the processed videos (6)

```python
    from videoanalizer import VideoAnalizer    # (1)

    vd = VideoAnalizer()                       # (2)

    clf = vd.train_OneClassSVM('./folder')     # (3)

    x, v_id = vd.process_video(files=['Obama.mp4', 'Elon.mp4']) # (4)

    x_train, y_train, x_test, y_test = vd.split_train_test(real_dir='./realObamas') # (5)

    vd.plot_features([['./folderObama'], ['./folderElon1', './folderElon2']])       # (6)
```

Many functions take as input **config**, this dict tells the videoanalizer how to extract videos.

```python
default_config={
    'interval':[(0,1e20)],      # intervals of frame to use, default: whole video
    'frames_per_sample': -1,    # -1 means every frame ; >0 means that a video with 600 frames will have 600/ frames_per_sample samples
    'overlap': 0,               # add overlapping samples (0->NO, 1->double, 2->triple)
    'only_success': True,       # drop frames where openface is not confident
    'vtype': 'single',          # detect one 'single' or multiple 'multi' faces in the videos
}
# NOTE: frames_per_sample is set to 300 when training classifiers
```

### General Usage 2

Every instance of VideoAnalizer contains an instance of OpenFaceAPI, you can use that instance to have a more direct interface to OpenFace (OpenFaceAPI is a python wrapper of OpenFace, which is compiled in C)

```python
    from videoanalizer import VideoAnalizer

    vd = VideoAnalizer()

    vd.api.process_images(fdir='./folderimg')  # api is the instance of OpenFaceAPI
    vd.api.process_video(fdir='./foldervid')
```

## Dataset structure:

    test_data
    └── videos
        ├── fake                  [Fake videos]
        |   ├── ElonMusk          [About Elon Musk]
        |   ├── Obama             [About Obama]
        │   ├── Renzi             [About Renzi]
        │   ├── thomas1           [About Thomas]
        |   ├── thomas2           [About Second thomas]
        |   └── morez             [About Moreno]
        |
        └── real                  [Real videos]
            ├── ElonMusk          [About Elon Musk]
            |   ├── train         [real videos for training]
            |   └── test          [fake videos for testing]
            ├── Obama             [About Obama]
            |   ├── train         [real videos for training]
            |   └── test          [fake videos for testing]
            ├── Renzi             [About Renzi]
            ├── thomas1           [About Thomas]
            ├── thomas2           [About Second thomas]
            └── morez             [About Moreno]
