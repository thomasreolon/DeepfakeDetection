# Fake detention in videos

:construction::construction::construction: currently under development :construction::construction::construction:

This project uses [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace.git) to analyze faces in videos.  
The project currently contains a wrapper to OpenFace in order to extract outputs (features) from images or videos.

---

**TODO**:  
:heavy_check_mark: interface to OpenFace  
:heavy_multiplication_x: find which are the best features to detect fake/real videos  
:heavy_multiplication_x: detect fake videos from real ones  

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
    |    ├── analize_img            [analize a specific image, to see its current available flags type analize_img.py -help]
    |    ├── analize_vid            [analize a specific video, to see its current available flags type analize_vid.py -help]
    |    └── video_analysis         [extract probability distributions for Action Units, currently under development]
    |
    └── test_data                   [contain images and videos for testing purposes]                                                

                          
---

## Usage

You can type:
```
    cd src
    python3 analize_vid.py --vn obama2.mp4
```
During the first run the program will ask you to install automatically OpenFace in the project folder, otherwise you can add your current path to OpenFace manually on the code.
After the processing the processed video will be displayed. The outputs from OpenFace will be available under <project>/output/openfacesaves/

## Examples
### Obama (real - fake)
<img src="https://media.giphy.com/media/K9YzMLteKq6sg5VJMh/giphy.gif" width="470" height="320" />  <img src="https://media.giphy.com/media/h9kD101j2VEXIjw9eY/giphy.gif" width="470" height="320" />
### Elon Musk (real - fake)
<img src="https://media.giphy.com/media/h31mo3j1UgSc8XE5Cx/giphy.gif" width="470" height="320" />  <img src="https://media.giphy.com/media/keuDEb10tk9Jnkpwi0/giphy.gif" width="470" height="320" />
