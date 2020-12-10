import subprocess, sys, os
#example: python3 run.py FaceLandmarkImg example.jpg
#FaceLandmarkImg: script to run from OpenFace
#example.jpg: image (or video) to process

argv = sys.argv[1:]
cwd = os.getcwd()

script = argv[0]
imageName = argv[1]

if(len(argv) == 0):
    print("Please provide a valid image")
    quit()

command = f'{cwd}/OpenFace/build/bin/{script} -out_dir {cwd}/output -f {cwd}/{imageName}'

os.system(command)
