# RISE Assignment: Research Engineer in Natural Language Processing
Fine-tune the `roberta-large` model on the NLP token classification task using the [MultiNERD (Named Entity Recognition Dataset)](https://huggingface.co/datasets/Babelscape/multinerd).

## Installation 
1. Clone the repository. 
2. The repository is based on [poetry](https://python-poetry.org/). If you have poetry installed, you must type `poetry install`, or there is also `requirements.txt` in the root directory for making a Python environment using the venv method.

Note: If you are using poetry, once the module is installed, to activate the environment, type `poetry shell`


## Train & Testing 
Once the module `assignment-rise` is installed, you may train/fine-tune the model for the assignment.
1. Turn on the Python environment installed from the [above step](#installation)
2. Go to [assignment_rise](/assignment_rise/) where you will find two files: `train_A.py` and `train_B.py`. 
3. To train the system A, type `python train_A.py &` and to train the system B, type `python train_B.py`.
4. Once the training finishes, it results in two directories in the same folder as these files. These directories are the best model from the fine-tuning process, namely `lora_system_A` and `lora_system_B`.

## Examples
If the training is done, you may experiment with the model. In [examples](/assignment_rise/examples/), there are a couple of notebooks on how to use the fine-tuned model. 


## Contact
Ahmad Zainul Ihsan
a.z.ihsan@icloud.com
