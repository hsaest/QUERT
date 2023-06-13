import json
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

from datasets import Dataset, load_dataset

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoTokenizer,
    BertTokenizer,
    Trainer,
    DataCollatorForWholeWordMask,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
# from trainer import QuertTrainer as Trainer
from ernie import ErnieConfig
from model import QueryBertForPreTraining, bert_add_tokens,QueryErnieForPreTraining
from data_collator import DataCollatorForGeoQueryModeling
from transformers.trainer_utils import get_last_checkpoint, is_main_process

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    
    tokens_order_size: int = field(
        default=32,
        metadata={
            "help": (
                "According to your corpus, adjust the tokens order size."
            )
        },
    )
    
    phrases_order_size: int = field(
        default=40,
        metadata={
            "help": (
                "According to your corpus, adjust the phrases order size."
            )
        },
    )
    
    special_tokens_file: str = field(
        default=None,
        metadata={"help": (
                    "add special tokens for tokenizer and embedding."
              )
         },
    )    
    
    
    cl_open: bool = field(
        default=True,
        metadata={
            "help": (
                "Will training with contrastive learning. Please make sure that the corpus containing augmented data."
            )
        },
    )
    
    geohash_open: bool = field(
        default=True,
        metadata={
            "help": (
                "Will training with geo hash prediction. Please make sure that the corpus containing augmented data."
            )
        },
    )
    
    mlm_open: bool = field(
        default=True,
        metadata={
            "help": (
                "Will training with geo hash prediction. Please make sure that the corpus containing augmented data."
            )
        },
    )
    
    hash_size: int = field(
        default=33,
        metadata={
            "help": (
                "the size of hash code number."
            )
        },
    )
    
    hash_bits: int = field(
        default=6,
        metadata={
            "help": (
                "the bits number of hash code."
            )
        },
    )
    
    triplet_open: bool = field(
        default=False,
        metadata={
            "help": (
                "Will training with triplet loss learning. Please make sure that the corpus containing augmented data."
            )
        },
    )
    
    hard_negative: bool = field(
        default=True,
        metadata={
            "help": (
                "Will training with triplet loss learning. Please make sure that the corpus containing augmented data."
            )
        },
    )
    
    pooler_type: str = field(
        default='cls',
        metadata={"help": (
                    "Pooler type in ['cls', 'mean']. "
              )
         },
    ) 
    
    temp: float = field(
        default=0.1,
        metadata={
            "help": "Temperature for softmax."
        }
    )
    
    margin: float = field(
        default=1,
        metadata={
            "help": "Temperature for softmax."
        }
    )
    
    
    order_open: bool = field(
        default=True,
        metadata={
            "help": (
                "Will training with order prediction. Please make sure that the corpus containing order data."
            )
        },
    )
    
    nsp_open: bool = field(
        default=False,
        metadata={
            "help": (
                "Will training with next sentence prediction."
            )
        },
    )
    
    geo_aware_open: bool = field(
        default=True,
        metadata={
            "help": (
                "Will training with geo aware mask prediction. Please make sure that the corpus containing geo data."
            )
        },
    )
    
    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    
    train_single_geo_ref_file: Optional[str] = field(
        default=None,
        metadata={"help": "An input train ref data file for single geo masking in Chinese."},
    )
    train_cover_geo_ref_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input train ref data file for cover geo masking in Chinese."},
    )
    train_order_ref_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input train ref data file for order shuffle."},
    )
    train_geohash_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input train data file containing geohash information."},
    )
    
    val_single_geo_ref_file: Optional[str] = field(
        default=None,
        metadata={"help": "An input val ref data file for single geo masking in Chinese."},
    )
    val_cover_geo_ref_file: Optional[str] = field(
        default=None,
        metadata={"help": "An input val ref data file for cover geo masking in Chinese."},
    )
    val_order_ref_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input val ref data file for order shuffle in Chinese."},
    )
    val_geohash_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input val data file containing geohash information."},
    )
    
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    max_seq_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated. Default to the max input length of the model."
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )

        
    pos_shuffle_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )

    
    def __post_init__(self):
        if self.train_file is not None:
            extension = self.train_file.split(".")[-1]
            assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
        if self.validation_file is not None:
            extension = self.validation_file.split(".")[-1]
            assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


def add_references(dataset, single_ref_file, cover_ref_file, order_ref_file, geohash_label_file, cl_open=True):
    import json
    with open(single_ref_file, "r", encoding="utf-8") as f:
        single_refs = [json.loads(line) for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
        
    with open(cover_ref_file, "r", encoding="utf-8") as f:
        cover_refs = [json.loads(line) for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
         
    with open(order_ref_file, "r", encoding="utf-8") as f:
        order_refs = [json.loads(line) for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]    
    
    with open(geohash_label_file, "r", encoding="utf-8") as f:
        geohash_labels = [line.split(' ') for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]  

    assert len(dataset) == len(single_refs) == len(cover_refs) == len(order_refs) == len(geohash_labels)
        
    dataset_dict = {c: dataset[c] for c in dataset.column_names}
    
    dataset_dict["single_geo_ref"] = single_refs
    dataset_dict["cover_geo_ref"] = cover_refs
    dataset_dict["order_ref"] = order_refs
    dataset_dict["geohash_label"] = geohash_labels
    
    return Dataset.from_dict(dataset_dict)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name)
        if "validation" not in datasets.keys():
            datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
            )
            datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
            )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = data_args.train_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
        datasets = load_dataset(extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        if 'ernie' in model_args.model_name_or_path.lower():
            config = ErnieConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
        else:
            config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")
            
    config.phrases_order_size = model_args.phrases_order_size
    config.tokens_order_size = model_args.tokens_order_size
    config.geohash_open = model_args.geohash_open
    config.hash_size = model_args.hash_size
    config.hash_bits = model_args.hash_bits
    config.cl_open = model_args.cl_open
    config.mlm_open = model_args.mlm_open
    config.order_open = model_args.order_open
    config.nsp_open = model_args.nsp_open
    config.triplet_open = model_args.triplet_open
    config.margin = model_args.margin
    config.hard_negative = model_args.hard_negative
    config.pooler_type = model_args.pooler_type 
    config.temp = model_args.temp
    
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        if 'ernie' in model_args.model_name_or_path.lower():
            tokenizer = BertTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if model_args.model_name_or_path:
        if 'ernie' in model_args.model_name_or_path.lower():
            model = QueryErnieForPreTraining.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        else:   
            model = QueryBertForPreTraining.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
            )
    else:
        logger.info("Training new model from scratch")
        model = QueryBertForPreTraining.from_config(config)
        
    if model_args.special_tokens_file:
        tokenizer = bert_add_tokens(tokenizer,model_args.special_tokens_file)
        
    model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = datasets["train"].column_names
    else:
        column_names = datasets["validation"].column_names
        
    text_column_name = "text" if "text" in column_names else column_names[0]

    padding = "max_length" if data_args.pad_to_max_length else False


    def tokenize_function(examples):
        # Remove empty lines
        features = {}
        if model_args.cl_open or model_args.triplet_open:
            sentences_mask = [line.split("\t")[0] for line in examples["text"] if len(line) > 0 and not line.isspace()]
            sentences_anchor = [line.split("\t")[0].split('[SEP]')[0] for line in examples["text"] if
                                len(line) > 0 and not line.isspace()]
            sentences_positive = [line.split("\t")[1] for line in examples["text"] if
                                  len(line) > 0 and not line.isspace()]
            if model_args.hard_negative or model_args.triplet_open:
                sentences_negative = [line.split("\t")[2] for line in examples["text"] if
                                  len(line) > 0 and not line.isspace()]
                sentences = sentences_anchor + sentences_mask + sentences_positive + sentences_negative
                assert len(sentences_anchor) == len(sentences_positive) == len(sentences_negative)
            else:
                sentences = sentences_anchor + sentences_mask + sentences_positive
                assert len(sentences_anchor) == len(sentences_positive)

            total = len(sentences_anchor)
            sent_features = tokenizer(sentences, padding=padding, truncation=True, max_length=data_args.max_seq_length)
            if model_args.hard_negative or model_args.triplet_open:
                for key in sent_features:
                    features[key] = [
                        [sent_features[key][i], sent_features[key][i + total], sent_features[key][i + total * 2],
                         sent_features[key][i + total * 3]] for i in range(total)]
            else:
                for key in sent_features:
                    features[key] = [
                        [sent_features[key][i], sent_features[key][i + total], sent_features[key][i + total * 2]] for i in range(total)]

        elif model_args.nsp_open:
            sentences_mask = [line.split("\t")[0] for line in examples["text"] if len(line) > 0 and not line.isspace()]
            sentences_anchor = [line.split("\t")[0].split('[SEP]')[0] for line in examples["text"] if
                                len(line) > 0 and not line.isspace()]
            sentences_positive = [(line.split("\t")[0], line.split("\t")[1]) for line in examples["text"] if
                                  len(line) > 0 and not line.isspace()]
            sentences_negative = [(line.split("\t")[0], line.split("\t")[2]) for line in examples["text"] if
                                  len(line) > 0 and not line.isspace()]
            assert len(sentences_anchor) == len(sentences_positive) == len(sentences_negative)
            total = len(sentences_anchor)
            sentences = sentences_anchor + sentences_mask + sentences_positive + sentences_negative
            sent_features = tokenizer(sentences, padding=padding, truncation=True, max_length=data_args.max_seq_length)
            for key in sent_features:
                features[key] = [
                    [sent_features[key][i], sent_features[key][i + total], sent_features[key][i + total * 2],
                     sent_features[key][i + total * 3]] for i in range(total)]

        else:
            sentences_mask = [line.split("\t")[0] for line in examples["text"] if len(line) > 0 and not line.isspace()]
            sentences_anchor = [line.split("\t")[0].split('[SEP]')[0] for line in examples["text"] if
                                len(line) > 0 and not line.isspace()]
            sentences = sentences_anchor + sentences_mask
            sent_features = tokenizer(sentences, padding=padding, truncation=True, max_length=data_args.max_seq_length)
            total = len(sentences_anchor)
            for key in sent_features:
                features[key] = [[sent_features[key][i], sent_features[key][i + total]] for i in range(total)]

        return features
    
    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=[text_column_name],
        load_from_cache_file=not data_args.overwrite_cache,
    )

    # Add the chinese references if provided
    if data_args.train_single_geo_ref_file is not None and data_args.train_cover_geo_ref_file is not None and data_args.train_order_ref_file is not None and data_args.train_geohash_file is not None:
        tokenized_datasets["train"] = add_references(tokenized_datasets["train"], data_args.train_single_geo_ref_file, data_args.train_cover_geo_ref_file, data_args.train_order_ref_file, data_args.train_geohash_file, model_args.cl_open)
    else:
        raise ValueError(
            "Empty geo train ref is not allowed"
        )
        
        
    if data_args.val_single_geo_ref_file is not None and data_args.val_cover_geo_ref_file is not None and data_args.val_order_ref_file is not None and data_args.val_geohash_file is not None:
        tokenized_datasets["validation"] = add_references(tokenized_datasets["validation"], data_args.val_single_geo_ref_file, data_args.val_cover_geo_ref_file, data_args.val_order_ref_file, data_args.val_geohash_file, model_args.cl_open)
    else:
        raise ValueError(
            "Empty geo validation ref is not allowed"
        )
    # If we have ref files, need to avoid it removed by trainer
    has_ref = True 
    if has_ref:
        training_args.remove_unused_columns = False
    
    # Data collator
    # This one will take care of randomly masking the tokens.
    data_collator = DataCollatorForGeoQueryModeling(tokenizer=tokenizer,cl_open=model_args.cl_open, triplet_open=model_args.triplet_open, order_open=model_args.order_open, nsp_open=model_args.nsp_open, hard_negative = model_args.hard_negative,geohash_open=model_args.geohash_open,mlm_open=model_args.mlm_open, geo_aware_open=model_args.geo_aware_open, pos_shuffle_probability=data_args.pos_shuffle_probability)

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"] if training_args.do_train else None,
        eval_dataset=tokenized_datasets["validation"] if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
        if trainer.is_world_process_zero():
            with open(output_train_file, "w") as writer:
                logger.info("***** Train results *****")
                for key, value in sorted(train_result.metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        eval_output = trainer.evaluate()

        perplexity = math.exp(eval_output["eval_loss"])
        results["perplexity"] = perplexity

        output_eval_file = os.path.join(training_args.output_dir, "eval_results_mlm_wwm.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in sorted(results.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()