# Fake detention in videos

This project uses [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace.git) to analyze faces in videos.  
The project contains a wrapper to openface and a system of models to predict if a given video of a specific person is real or fake.

---

## Project structure:
    SIV_project
    ├── src
    |    ├── openface               [package to interact with OpenFace]
    |    |   ├── errors             [library which contains errors]
    |    |   ├── face_comparator    [script to compare the similarity between two faces]
    |    │   ├── installer          [script that can install OpenFace automatically]
    |    │   ├── openfaceapi        [class that can process images/videos]
    |    |   ├── output_extractor   [script to extract features from the openface's csv]
    |    |   └── parts              [library which contains information about face's parts]
    |    |
    |    ├── create_landmarks       [create the video with the landmark real or fake based on the predictions]
    |    ├── download_dataset       [download our dataset]
    |    ├── plots                  [some useful plots for data analysis (PCA, LDA)]
    |    ├── pretrained_model       [pretrain a model on all the data of the dataset, deprecated]
    |    └── train_a_model          [train and test the final classification system]
    |
    |
    └── test_data                   [dataset of video for training and testing (not visible)]                                              

                          
---

## Usage

You can type:
```
    cd src
    python3 analize_vid.py --vn obama2.mp4
```
During the first run the program will ask you to install automatically OpenFace in the project folder, otherwise you can add your current path to OpenFace manually on the code.
After the processing the processed video will be displayed. The outputs from OpenFace will be available under <project>/output/openfacesaves/

## Train and test the classification system
After downloading the dataset (download_dataset) you may use the classification system running:
```
    cd src
    python3 train_a_model.py
```

### Dataset structure:
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
            
### Use landmarks
This project also offer a script to attach to a video the predicted landmarks (using our system) using the landmark function offered by openface.
In orther to run this script:
```
    cd src
    python3 create_landmarks.py
```
This is the result predicting videos about Obama and Elon Musk:
#### Obama (real - fake)
<img src="https://media.giphy.com/media/K9YzMLteKq6sg5VJMh/giphy.gif" width="400" height="300" />  <img src="https://media.giphy.com/media/h9kD101j2VEXIjw9eY/giphy.gif" width="400" height="300" />
#### Elon Musk (real - fake)
<img src="https://media.giphy.com/media/h31mo3j1UgSc8XE5Cx/giphy.gif" width="400" height="300" />  <img src="https://media.giphy.com/media/keuDEb10tk9Jnkpwi0/giphy.gif" width="400" height="300" />

