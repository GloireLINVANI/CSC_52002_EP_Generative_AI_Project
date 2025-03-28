# TEXT-CONDITIONED INPAINTING WITH REPAINT SAMPLING STRATEGY
## Data Folder
This folder contains the data and datasets used, as well as the scripts used for downloading, sampling, and processing the data. It also contains the final evaluation quantitative data.

## Directory Structure
* `data_sources/`: Contains scripts used for downloading and reorganizing data. 

    * `download_places_coco.py`: Downloads and unzips the sources of the original datasets used.
    * `reorganize_images.py`: Moves images from a batch based directory structure to a non-batch based structure.
    * `Small-ImageNet...`: Copied from [this repository](https://github.com/ndb796/Small-ImageNet-Validation-Dataset-1000-Classes/) - example of how to load their subset of imagenet data.
    * `extract_few.sh`: Extracts five samples from each class directory in a standard dataset structure. Necessary for enormous datasets like COCO and Places.
* `datasets/`: Contains the actual datasets used. Due to potential copyright issues, the datasets were uploaded to private kaggle datasets, and can be downloaded once access is requested using the `download_kaggle.sh` script provided.
* `eval_csv/`: Contains the CSV files for our quantitative results.

    * `fid_scores_large_nip.csv`: FID scores reported on samples upscaled using the non-finetuned upscaling model from GLIDE. The same model was used for all sampled 64x64 images, which were generated by the listed models.
    * `fid_scores_large.csv`: FID scores on 256x256 used in our final report. The corresponding upscaling model to the one used to generate each 64x64 sample was used.
    * `fid_scores.csv`: FID scores on images generated for each mask, in 64x64.
    * `lpips_scores_full_nip.csv`: LPIPS scores reported on samples upscaled using the non-finetuned upscaling model from GLIDE. The same model was used for all sampled 64x64 images, which were generated by the listed models.
    * `lpips_scores_full.csv`: LPIPS scores on 256x256 used in our final report. The corresponding upscaling model to the one used to generate each 64x64 sample was used.
    * `lpips_scores.csv`: LPIPS scores reported only Places 365 Validation set, used in our poster.

* `large_masked_coco/`: Masked versions of images from the COCO dataset, similar to the ones used in our human evaluation survey. Good image source for trying out the model without having to download a full dataset.
* `masks/`: Masks used in our evaluation. Based on the masks provided in the RePaint repository. 