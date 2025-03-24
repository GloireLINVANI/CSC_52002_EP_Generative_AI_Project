# TEXT-CONDITIONED INPAINTING WITH REPAINT SAMPLING STRATEGY
## Test Notebooks Folder
Contains notebooks used for testing and development. Do not expect them to work without some required files. 

⚠️ NOTEBOOKS USED FOR TESTING! EXPECT SPAGHETTI 🍝 !!! ⚠️

## Notebooks:
* `batch_process_masks.ipynb`: Notebook that the eval script `process_datasets.py` was based on.
* `inpaint_face_repaint.ipynb`: First test patching the GLIDE model with the RePaint sampling strategy. Based on the face image given with repaint, but the notebook is loosely based on the one provided with GLIDE.
* `inpaint_face.ipynb`: Test trying the GLIDE inpainting model on the inpainting task from RePaint paper (with a literal mask). It fails catastrophically due to OpenAI's filtering. See the appendix on the GLIDE paper about Safety.
* `sample_coco_for_human_eval.ipynb`: The notebook used to generate the images for the human survey.
* `test_repaint.ipynb`: Using the repaint patching on batches of images based on the grass png provided with GLIDE. Comparing the various models, used to determine comparable parameters for each model.
* `upscale_images.ipynb`: Notebook used for the basis of `upscale_simple.py` and `upscale_full.py`. Note that it also has a bunch of code used for sampling from the generated 64x64 images for the upsampling stage.