import pprint
import evaluate
from transformers import (AutoTokenizer,
                          AutoModelForTokenClassification,
                          TrainingArguments, 
                          Trainer, 
                          DataCollatorForTokenClassification)
from peft import (LoraConfig, 
                  get_peft_model, 
                  prepare_model_for_int8_training, 
                  TaskType)

from assignment_rise.utils.nerd_handler import NERDData
from assignment_rise.utils.utils import prepare_compute_metrics
from assignment_rise.utils.labels import SYSTEM_A_LABELS_ID

def main():
    print(">>>>>>>>>>>>>>>>>>>>>>If you haven't a MNERD dataset, it will take a while for downloading the dataset. Be Patience<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    model_name = 'roberta-large'
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
    data = NERDData('en', target_labels_id=SYSTEM_A_LABELS_ID, tokenizer=tokenizer)
    tokenized_data = data.tokenize_and_align()
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        device_map='auto', 
        num_labels=len(data.target_id_labels),
        id2label=data.target_id_labels,
        label2id=data.target_labels_id)
    # Define LoRA Config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.TOKEN_CLS)
    
    # prepare int-8 model for training
    model = prepare_model_for_int8_training(model)

    # add LoRA adaptor
    model = get_peft_model(model, lora_config)
    print(">>>>>>>>>>>>>>>>>>>>>>MODEL INITIALIZED<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    num_epochs = 3
    batch_size = 32
    logging_steps = 200
    model_name = 'system-A'
    gradient_accumulation_steps = 2  # increase gradient accumulation steps by 2x if batch size is reduced
    optim = 'paged_adamw_32bit' # activates the paging for better memory management
    save_strategy='epoch' # checkpoint save strategy to adopt during training
    evaluation_strategy='epoch'
    learning_rate = 3e-4  # learning rate for AdamW optimizer
    max_grad_norm = 0.3 # maximum gradient norm (for gradient clipping)
    warmup_ratio = 0.03 # number of steps used for a linear warmup from 0 to learning_rate
    lr_scheduler_type = 'cosine'  # learning rate scheduler

    training_args = TrainingArguments(
        output_dir=model_name,  
        optim=optim,
        learning_rate = learning_rate,
        warmup_ratio=warmup_ratio,
        max_grad_norm=max_grad_norm,
        lr_scheduler_type=lr_scheduler_type,
        num_train_epochs=num_epochs, 
        per_device_train_batch_size=batch_size, 
        per_device_eval_batch_size=batch_size, 
        gradient_accumulation_steps=gradient_accumulation_steps,
        evaluation_strategy=evaluation_strategy,
        save_strategy=save_strategy,
        load_best_model_at_end=True,
        disable_tqdm=False, 
        logging_steps=logging_steps, 
        push_to_hub=False)
    
    seqeval = evaluate.load('seqeval')
    compute_metrics = prepare_compute_metrics(labels_list=data.target_labels_list, seqeval=seqeval)
    data_collator = DataCollatorForTokenClassification(tokenizer)
    trainer = Trainer(model=model, args=training_args, 
                  data_collator=data_collator, compute_metrics=compute_metrics, 
                  train_dataset=tokenized_data['train'], eval_dataset=tokenized_data['validation'],
                  tokenizer=tokenizer)
    # train
    print(">>>>>>>>>>>>>>>>>>>>>>FINE-TUNING<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    trainer.train()
    save_best_model = 'lora_system_A'
    trainer.model.save_pretrained(save_best_model)

    # testing 
    print(">>>>>>>>>>>>>>>>>>>>>>TESTING<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    predictions = trainer.predict(tokenized_data['test'])
    test_metrics = predictions[2]
    print('Test Result')
    pprint.pprint(test_metrics)

    return None

if __name__ == "__main__":
    main()

