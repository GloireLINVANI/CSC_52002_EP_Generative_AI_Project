# TEXT-CONDITIONED INPAINTING WITH REPAINT SAMPLING STRATEGY
## Evaluation Scripts Folder
Contains scripts used for evaluation, including score calculation and the actual image generation / upscaling. Each of these scripts needs to be run from the root directory of this repository, i.e. the command for running one of the scripts would be 

```sh
python eval_scripts/process_datasets.py
```

## Scripts Included
* `fid_scores_large_samples.sh`: Assumes large samples are stored in a folder called `large_samples` in the directory being run from. With pytorch_fid installed (`python -m pip install pytorch_fid`), will calculate FID scores for every model compared with the originals, stored in the directory 'original' for each dataset.
* `fid_scores.sh`: Assumes samples are stored in a folder called `samples` in the directory being run from. With pytorch_fid installed (`python -m pip install pytorch_fid`), will calculate FID scores for every model compared with the originals, stored in the directory 'original' for each mask, for each dataset.
* `lpips_scores.py`: Generates an LPIPS score for each matching image name for each model, in a directory structure matching the one for `fid_scores_large_samples.sh`.
*`process_datasets.py`: With the correct datasets in the correct folders (under `data/datasets/{dataset_name}`), will process the datasets.
* `upscale_full.py`: With the samples to upscale, as well as their corresponding originals, stored in the correct directory structure under `samples_to_upscale`, will do inpainting and upscaling on generated samples with the same model (i.e. GLIDE-Inpaint (`base`), GLIDE-Base w/ Repaint (`repaint`), GLIDE-Inpaint w/ RePaint (`rpip`)) as used on the generated samples.
* `upscale_simple.py`: With the samples to upscale, as well as their corresponding originals, stored in the correct directory structure under `samples_to_upscale`, will do inpainting and upscaling on generated samples with one non-finetuned GLIDE upscaling model.