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
    --eval_steps 10000\
    --save_total_limit 2 \
    --evaluation_strategy steps \
    --load_best_model_at_end True \
    --num_train_epochs 10 \
    --save_steps 10000 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 32