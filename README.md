# Multi-task Learning based Pre-trained Language Model for Code Completion

## Datasets
The project information are listed in the `data/JAVA_repos.txt` and `data/TS_repos.txt` files. For JAVA projetcs, each line represents a project which includes the GitHub username and project name connected by "_".

+ data preparation

1. create data corpus

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

2. create data instances

```
python create_instances.py \
  --input_file=$path to input token corpus$ \
  --input_type_file=$path to input type corpus$
  --output_file=$path to save data instances$
  --token_vocab_file=$path to token vocab file$ \
  --type_vocab_file=$path to type vocab file$ \
  --dupe_factor=5
```

## Model Pre-training
```
python run_pretraining.py \
  --input_file=$path to save pre-training data instances$ \
  --eval_input_file=$path to save eval data instances$ \
  --output_dir=$path to save model$ \
  --token_vocab_file=$path to token vocab file$ \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=bert_config.json \
  --train_batch_size=16 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=600000 \
  --learning_rate=5e-5 \
  --n_gpus=3 \
  --gpu =0,1,2

```


## Model Fine-tuning
```
python run_finetuning.py \
  --input_file=$path to save pre-training data instances$ \
  --eval_input_file=$path to save eval data instances$ \
  --test_input_file=$path to save test data instances$ \
  --small_input_file=$path to save small test data instances$ \
  --output_dir=$path to save model$ \
  --token_vocab_file=$path to token vocab file$ \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=bert_config.json \
  --train_batch_size=16 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=300000 \
  --learning_rate=5e-5 \
  --n_gpus=3 \
  --gpu =0,1,2

```


