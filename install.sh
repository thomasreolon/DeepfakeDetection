git clone https://github.com/TadasBaltrusaitis/OpenFace.git
# sudo apt-get install ffmpeg # Don't know if needed
# sudo apt install libgstreamer1.0-dev gstreamer1.0-plugins-base gstreamer1.0-plugins-good # Don't know if needed
cd OpenFace
bash ./download_models.sh
sudo bash ./install.sh
cd ..
cd mkdir output/
