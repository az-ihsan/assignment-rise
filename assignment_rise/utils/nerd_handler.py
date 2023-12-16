from datasets import Dataset, load_dataset
from assignment_rise.utils.labels import SYSTEM_A_LABELS_ID
from transformers import AutoTokenizer

TOKENIZER = AutoTokenizer.from_pretrained('roberta-large', add_prefix_space=True)

class NERDData():
    '''
    An object to handle Multi-NERD data
    '''
    
    def __init__(self, lang:None, target_labels_id:dict=SYSTEM_A_LABELS_ID, tokenizer=TOKENIZER):
        '''
        Initialize the MultiNERDData object
            Arguments:
                lang: Specify this if we want to have a specific language, e.g., English (en), German (de), etc in the data
                target_labels: A dictionary of labels that we are interested in the datast 

            Return:
                MultiNERDData object
        '''
        self.lang_list = ['zh', 'nl', 'en', 'fr', 'de', 'it', 'pl', 'pt', 'ru', 'es']
        self.tokenizer = tokenizer
        
        # language should be constrained according to self.lang_list
        if lang in self.lang_list:
            self.lang = lang
        else: 
            raise ValueError('the language that you choose is not available.')
       
        
        # A set of def labels keys
        DEF_LABELS_keys = {key for key in SYSTEM_A_LABELS_ID.keys()} 
        # A set of target labels keys
        target_labels_keys = {key for key in target_labels_id.keys()} 

        # constraint for inserting label, should follow to SYSTEM_A_LABELS 
        for key in target_labels_keys: 
            if key in DEF_LABELS_keys:
                continue
            else: 
                raise ValueError("You mistyped the classes")
            
        self.labels = target_labels_id
        self.labels_val_list = {val for _, val in self.labels.items()} 
        
        # work around to make labels and ids contigous
        self.old_target_labels_id = target_labels_id
        self.old_target_id_labels = {value:key for key, value in self.old_target_labels_id.items()}
        
        self.target_id_labels = {i:key for i, (key, value) in enumerate(target_labels_id.items())}
        self.target_labels_id = {value:key for key, value in self.target_id_labels.items()}
        self.target_labels_list =[key for key in self.target_labels_id.keys()]
        
        # filtered data
        self.filtered_data = self._preprocess_dataset()
        
    def _preprocess_dataset(self):
        mnerd_dataset = load_dataset('Babelscape/multinerd') 
        filtered_dataset = self._filtered_out(mnerd_dataset)

        return filtered_dataset
    

    def _preprocess_ner_tags(self, rows_ner_tags):
        new_ner_tags = []
        for tag in rows_ner_tags:
            if tag in self.labels_val_list:
                label = self.old_target_id_labels[tag]
                new_ner_tags.append(self.target_labels_id[label])
            else:
                new_ner_tags.append(0)
        return new_ner_tags
    
    def _filtered_out(self, dataset:Dataset):
        # filter the dataset according to the language 
        dataset_en = dataset.filter(lambda x: x["lang"] == self.lang)
        
        # filter the dataset according to the chosen tags
        filtered_dataset_en = dataset_en.map(lambda x: {'ner_tags': self._preprocess_ner_tags(x['ner_tags'])})

        return filtered_dataset_en
    
    def tokenize_and_align(self):
        def _tokenize(row):
            tokenized_inputs = self.tokenizer(row['tokens'], truncation=True,
                                        is_split_into_words=True)
        
            labels = []
            for idx, label in enumerate(row['ner_tags']):
                word_ids = tokenized_inputs.word_ids(batch_index=idx)
                previous_word_idx = None
                label_ids = []
                for word_idx in word_ids:
                    if word_idx is None or word_idx == previous_word_idx:
                        label_ids.append(-100)
                    else:
                        label_ids.append(label[word_idx])
                    previous_word_idx = word_idx
                labels.append(label_ids)
            tokenized_inputs['labels'] = labels

            return tokenized_inputs
        
        tokenized_aligned_dataset = self.filtered_data.map(_tokenize, 
                                                            batched=True, 
                                                            remove_columns=['lang', 'tokens', 'ner_tags'])
        
        return tokenized_aligned_dataset