Resources for our KDD 2023 paper: "[QUERT: Continual Pre-training of Language Model for Query Understanding in Travel Domain Search](https://arxiv.org/abs/2306.06707)".

#### TODO

- [ ] Release downstream tasks

## Dependencies

- `pytorch==1.8.0`
- `transformers==4.18.0`
- `datasets`

For convenience, you can also run `pip3 install -r requirements.txt`

## Code Structure Explanation


- `config.json` Configuration for the model
- `data_collator.py` Training data batch generation
- `model.py` Model Structure
- `data_preprocess.py` Corpus preprocessing code
- `run.py` Model training code
- `train.sh` Launched by shell script

## Parameter Explaination
- `geohash_open` - Whether to enable the geohash pre-training task, default is true
- `nsp_open` - Whether to enable the NSP task (query[SEP]item), default is false
- `cl_open` - Whether to enable contrastive learning (i.e., UCBL task), default is true
- `order_open` - Whether to enable order prediction task, default is true
- `geo_aware_open` - Whether to enable the Geo-MP task, default is true, if closed, it degrades to the original MLM
- `mlm_open` - Whether to enable the MLM task, default is true
- `hard_negative` - Whether to enable hard negative sample learning for contrastive learning (data needs to be prepared in advance, default is false)
- `triplet_open` - Whether to use triplet loss (under this setting, contrastive learning needs to be turned off), default is false

## Data Processing
First, run the `data_preprocess.py` file. 

The input file format is `[query, item, geo_in_query, geo_in_item, query_tag (tokenized query), query_json, geo_hash, similar_query]`, separated by `\t` with each line being a data column.

Example: 

`318川藏线全包 定制旅行西藏川藏线318自驾稻城亚丁包车拼车定制旅游 none 稻城亚丁 318川藏线;全包 [{"term": "318川藏线", "first_cate": "division2poi", "second_cate": "division2poi", "codes": ""}, {"term": "全包", "first_cate": "abstract", "second_cate": "abstract", "codes": ""}] w j p 3 f h 西藏`

## Run

```bash
export TRAIN_FILE=path/to/your/training_file.txt
export VALIDATION_FILE=path/to/your/validation_file.txt
export TRAIN_SINGLE_REF_FILE=path/to/your/training_single_reference_file.txt
export TRAIN_COVER_REF_FILE=path/to/your/training_cover_reference_file.txt
export TRAIN_ORDER_REF_FILE=path/to/your/training_order_reference_file.txt
export TRAIN_GEOHASH_FILE=path/to/your/training_geohash_reference_file.txt
export VALIDATION_SINGLE_REF_FILE=path/to/your/validation_single_reference_file.txt
export VALIDATION_COVER_REF_FILE=path/to/your/validation_cover_reference_file.txt
export VALIDATION_ORDER_REF_FILE=path/to/your/validation_order_reference_file.txt
export VALIDATION_GEOHASH_FILE=path/to/your/validation_geohash_reference_file.txt
export OUTPUT_DIR=path/to/your/output_directory

python run.py \
    --model_name_or_path bert-base-chinese \
    --tokens_order_size 64 \
    --phrases_order_size 64 \
    --cl_open True \
    --geohash_open True \
    --order_open True \
    --mlm_open True  \
    --nsp_open False  \
    --geo_aware_open True \
    --hard_negative False \
    --triplet_open False \
    --pooler_type "cls" \
    --temp 0.1 \
    --overwrite_output_dir \
    --pos_shuffle_probability 0.15 \
    --train_file $TRAIN_FILE \
    --validation_file $VALIDATION_FILE \
    --train_single_geo_ref_file $TRAIN_SINGLE_REF_FILE \
    --train_cover_geo_ref_file $TRAIN_COVER_REF_FILE \
    --train_order_ref_file $TRAIN_ORDER_REF_FILE \
    --train_geohash_file $TRAIN_GEOHASH_FILE \
    --val_single_geo_ref_file $VALIDATION_SINGLE_REF_FILE \
    --val_cover_geo_ref_file $VALIDATION_COVER_REF_FILE \
    --val_order_ref_file $VALIDATION_ORDER_REF_FILE \
    --val_geohash_file $VALIDATION_GEOHASH_FILE \
    --do_train \
    --do_eval \
    --output_dir $OUTPUT_DIR \
    --eval_steps 5000\
    --save_total_limit 2 \
    --evaluation_strategy steps \
    --load_best_model_at_end True \
    --num_train_epochs 10 \
    --save_steps 5000 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 32
```

## Citation

If our paper or related resources prove valuable to your research, we kindly ask for citation. Please feel free to contact us with any inquiries.

```
@article{xie2023quert,
  title={QUERT: Continual Pre-training of Language Model for Query Understanding in Travel Domain Search},
  author={Jian Xie, Yidan Liang, Jingping Liu, Yanghua Xiao, Baohua Wu, Shenghua Ni},
  journal={arXiv preprint arXiv:2306.06707},
  year={2023}
}
```

## Question

If you find any questions, please feel free to contact Jian Xie `jianx0321@gmail.com` .  

You can also create new issue directly.
