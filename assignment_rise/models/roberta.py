import torch
import pandas as pd
from typing import Union
from transformers import (AutoModelForTokenClassification, 
                          AutoTokenizer, )
from peft import (PeftConfig, 
                  PeftModel)
import warnings
warnings.filterwarnings("ignore")

class ROBERTA(object):
    '''
    ROBERTA is a wrapper ROBERTA huggingface model which is an encoder-decoder model. 
    The class will generate an object of Falcon model according to the chosen model check point.
    '''
    def __init__(self, chkpt:str="roberta-large", labels_id=None):
        '''
        Initializes the ROBERTA object
            Arguments:
                chkpt: A model checkpoint path/huggingface model name from the fine-tuned ROBERTA-large
                load_in_4bit: Boolean whether we want to load the model in 4 bit
            Returns:
                ROBERTA object
        '''
        self.peft_model_id = chkpt
        self.config = PeftConfig.from_pretrained(self.peft_model_id)
        self.label2id = labels_id
        self.num_labels = len(labels_id)
        self.id2label = {value:key for key, value in labels_id.items()}   
        self.roberta_model = AutoModelForTokenClassification.from_pretrained(
            self.config.base_model_name_or_path,
            return_dict=True,
            device_map="auto",
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_name_or_path, add_prefix_space=True)
        self.lora_roberta_model = PeftModel.from_pretrained(self.roberta_model, self.peft_model_id)

    def predict_tags(self, text:Union[str, list[str]]):
        if isinstance(text, str):
            tokens = self.tokenizer(text).tokens()
            input_ids = self.tokenizer(text, is_split_into_words=False, return_tensors='pt').input_ids
        elif isinstance(text, list):
            tokens = self.tokenizer(text, is_split_into_words=True).tokens()
            input_ids = self.tokenizer(text, is_split_into_words=True, return_tensors='pt').input_ids
        else:
            raise ValueError("Please enter a text as string or a list of tokenized string")
        
        outputs = self.lora_roberta_model(input_ids).logits
        predictions = torch.argmax(outputs, dim=2)

        preds = [self.id2label[p] for p in predictions[0].numpy()]

        return pd.DataFrame([tokens, preds], index=['tokens', 'tags'])