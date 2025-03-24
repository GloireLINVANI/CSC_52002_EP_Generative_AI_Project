# TEXT-CONDITIONED INPAINTING WITH REPAINT SAMPLING STRATEGY (CSC_52002_EP GenAI Project)
Semantic Control in Diffusion Inpainting: Merging RePaint Sampling with Text-Driven Generation.

## Structure of this Repository

**Files**:
 * `Inpainting_GLIDE_and_RePaint.ipynb`: The demo notebook. It can be run through google colab, and will create a Gradio UI. Run using [this link](https://githubtocolab.com/pandanautinspace/CSC_52002_EP_Generative_AI_Project/blob/refactor/Inpainting_GLIDE_and_RePaint.ipynb), or import the file using the Colab UI. The resampling steps used are greatly reduced so that similar time-performance is achieved by all models, but it is easy to change with the jump_params variable. 
 * `Report Evaluation Data.xlsx`: The excel file containing the tables used for the report. They are based on CSV files also contained in this repository.

**Directories**:
 * `data/`: All of the data used for evaluation of the model, as well as the CSV files containing the evaluation scores. Described in more detail in directory.
 * `eval_scripts/`: Scripts used during the evaluation step. Their default arguments assume a directory structure that is no longer valid for this repository, but they can be easily modified to work if needed. Described in more detail in directory.
 * `glide_patching/`: Our main contribution in terms of code. Assuming that the `glide-text2im` library is installed, the scripts here will fetch a GLIDE model in the format expected and patch it to use RePaint sampling. Also provided are sampling convenience classes. Described in more detail in directory.
 * `Source Papers/`: PDF files of the papers from which our method is inspired and derived. 
 * `test_notebooks/`: Python notebooks used during testing and construction of our process. May need slight changes in order to work with the structure of the data. Described in more detail in directory.

**Submodules**:
 * `glide-text2im`: Version of GLIDE that our model is based on. If needed, one can `cd` into this directory and `pip install .` in order to get the version of GLIDE used in our paper.
 * `RePaint`: Version of RePaint that our model is based on. The methods in `repaint_patcher.py` are based on source from this repository.