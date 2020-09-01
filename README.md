# Multi-task Learning based Pre-trained Language Model for Code Completion

Recent studies have shown that statistical language modeling techniques can improve the performance of code completion tools through learning from large-scale software repositories. However, these models suffer from two major drawbacks: a) Existing research uses static embeddings, which map a word to the same vector regardless of its context. The differences in the meaning of a token in varying contexts are lost when each token is associated with a single representation; b) Existing LM-based code completion models perform poor on completing identifiers, and the type information of the identifiers is ignored in most of these models. To address these challenges, in this paper, we develop a multi-task learning based pre-trained language model for code understanding and code generation with a Transformer-based neural architecture. We pre-train it with hybrid objective functions that incorporate both code understanding and code generation tasks. Then we fine-tune the pre-trained model on code completion. During the completion, our model does not directly predict the next token. Instead, we adopt multi-task learning to predict the token and its type jointly and utilize the predicted type to assist the token prediction. 

## Datasets
The project information used in our paper are listed in the `data/JAVA_repos.txt` and `data/TS_repos.txt` files. For JAVA projetcs, each line represents a project which includes the GitHub username and project name connected by "_". 

> You can use your own datasets of any languages by formating the programs into the following format:

**Data Preparation**

+ Create data corpus

    data corpus format:

    (1) One line of code tokens (corresponding types) per line.
    
    (2) Blank lines between code files. File boundaries are needed so that the "next code segment prediction" task doesn't span between code files.

    Here, we need token corpus and corresponding type corpus as input. 

    **Type annotation:**
    For Java programs, we extract the identifiers' type information through static analysis. For TypeScript programs, we apply the approach in Hellendoorn et al. (Deep Learning Type Inference) to extract type annotations of the identifiers. 
    
    a Java input example: 
   
    **token corpus:**
    
    ```
    ["public", "class", "Note", "{"]
    ["private", "static", "Toast", "mToast", ";"]
    ["public", "static", "void", "show", "(", "String", "msg", ")", "{"]
    ["if", "(", "mToast", "==", "null", ")"]
    ["mToast", "=", "Toast", ".", "makeText", "(", "AndroidApplication", ".", "getInstance", "(", ")", ",", "null", ",", "Toast", ".", "LENGTH_SHORT", ")", ";"]
    ["mToast", ".", "setText", "(", "msg", ")", ";"]
    ["mToast", ".", "setDuration", "(", "Toast", ".", "LENGTH_SHORT", ")", ";"]
    ["mToast", ".", "show", "(", ")", ";", "}"]
    
    ```
    
    **type corpus:**
    
    ```
    ["_", "_", "_", "_"]
    ["_", "_", "android.widget.Toast", "_", "_"]
    ["_", "_", "_", "_", "_", "java.lang.String", "_", "_", "_"]
    ["_", "_", "tellh.com.gitclub.common.wrapper.Note.Note.mToast", "_", "_", "_"]
    ["tellh.com.gitclub.common.wrapper.Note.Note.mToast", "_", "android.widget.Toast", "_", "android.widget.Toast.makeText", "_", "tellh.com.gitclub.common.AndroidApplication", "_", "tellh.com.gitclub.common.AndroidApplication.getInstance", "_", "_", "_", "_", "_", "android.widget.Toast", "_", "android.widget.Toast.LENGTH_SHORT", "_", "_"]
    ["tellh.com.gitclub.common.wrapper.Note.Note.mToast", "_", "android.widget.Toast.setText", "_", "", "_", "_"]
    ["tellh.com.gitclub.common.wrapper.Note.Note.mToast", "_", "android.widget.Toast.setDuration", "_", "android.widget.Toast", "_", "android.widget.Toast.LENGTH_SHORT", "_", "_"]
    ["tellh.com.gitclub.common.wrapper.Note.Note.mToast", "_", "android.widget.Toast.show", "_", "_", "_", "_"]
    ```

+ Create data instances

    ```
    python create_instances.py \
      --input_file=$path_to_input_token_corpus$ \
      --input_type_file=$path_to_input_type_corpus$ \
      --output_file=$path_to_save_data_instances$ \
      --token_vocab_file=$path_to_token_vocab_file$ \
      --type_vocab_file=$path_to_type_vocab_file$ \
      --dupe_factor=5
    ```

## Model Pre-training

The model configuration is specified in `bert_config.json` file.

```
python run_pretraining.py \
  --input_file=$path_to_save_pre-training_data_instances$ \
  --eval_input_file=$path_to_save_eval_data_instances$ \
  --output_dir=$path_to_save model$ \
  --token_vocab_file=$path_to_token_vocab_file$ \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=bert_config.json \
  --train_batch_size=16 \
  --max_seq_length=128 \
  --max_predictions_per_seq=30 \
  --num_train_steps=600000 \
  --learning_rate=5e-5 \
  --n_gpus=3 \
  --gpu=0,1,2
    
```


## Model Fine-tuning
```
python run_finetuning.py \
  --input_file$path_to_save_fine-tuning_data_instances$ \
  --eval_input_file=$path_to_save_eval_data_instances$ \
  --test_input_file=$path_to_save_test_data_instances$ \
  --small_input_file=$path_to_save_small_test_data_instances$ \
  --output_dir=$path_to_save_model$ \
  --token_vocab_file=$path_to_token_vocab_file$ \
  --do_train=True \
  --do_eval=True \
  --LM True
  --bert_config_file=bert_config.json \
  --train_batch_size=16 \
  --max_seq_length=128 \
  --max_predictions_per_seq=30 \
  --num_train_steps=300000 \
  --learning_rate=5e-5 \
  --n_gpus=3 \
  --gpu=0,1,2
```
    
My implementation is mainly based on https://github.com/google-research/bert.