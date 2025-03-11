mkdir -p data/datasets/places2
cd data/datasets/places2

# Places365-Standard small images (256x256)
# Validation set only (~36K images)
wget http://data.csail.mit.edu/places/places365/val_256.tar
tar -xvf datasets/places2/val_256.tar