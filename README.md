# RISE Assignment: Research Engineer in Natural Language Processing
Fine-tune the `roberta-large` model on the NLP token classification task using the [MultiNERD (Named Entity Recognition Dataset)](https://huggingface.co/datasets/Babelscape/multinerd).

## Installation 
There are two possibilities to install the package and make sure the Python version > 3.10
### Poetry
1. Clone the repository and enter the downloaded repository.
2. Install the poetry project

   ```poetry install```

4. Activate the the environment

   ```poetry shell```
   
### VENV
1. Clone the repository and enter the downloaded repository.
2. Make a python environment
   
   ```python3 -m venv ~/.virtualenvs/assignment-rise```
3. Activate the virtual environment
  
   ```source ~/.virtualenvs/assignment-rise/bin/activate```

4. If your command line prompt now has (assignment-rise) as prefix you have successfully activated your newly created virtual environment.
5. Then, install the package
   
   ```pip install .```


## Train & Testing 
Once the package`assignment-rise` is installed, you may train/fine-tune the model for the assignment.
1. Turn on the Python environment installed from the [above step](#installation)
2. Go to [assignment_rise](/assignment_rise/) where you will find two files: `train_A.py` and `train_B.py`. 
3. To train the system A, type `python train_A.py &` and to train the system B, type `python train_B.py`.
4. Once the training finishes, it results in two directories in the same folder as these files. These directories are the best model from the fine-tuning process, namely `lora_system_A` and `lora_system_B`.

## Examples
If the training is done, you may experiment with the model. In [examples](/assignment_rise/examples/), there are a couple of notebooks on how to use the fine-tuned model. 


## Contact

Ahmad Zainul Ihsan

a.z.ihsan@icloud.com
