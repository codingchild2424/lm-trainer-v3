    # p.add_argument('--model_name_or_path', type=str, default=None)
    # p.add_argument('--train_file_path', type=str, default=None)
    # p.add_argument('--test_file_path', type=str, default=None)
    # p.add_argument('--output_dir', type=str, default=None)
    # p.add_argument('--text_column_name', type=str, default='text')
    # p.add_argument('--label_column_name', type=str, default='label')
    # p.add_argument('--num_labels', type=int, default=2)
    # p.add_argument('--id2label', type=dict, default={ 0: "CLEAN", 1: "UNCLEAN" })
    # p.add_argument('--per_device_train_batch_size', type=int, default=8)
    # p.add_argument('--per_device_eval_batch_size', type=int, default=8)
    # p.add_argument('--epochs', type=int, default=3)

torchrun --nproc_per_node=8 --master_port=34321 ../trainers/run_mlm.py \
    --model_name_or_path='monologg/koelectra-base-v3-discriminator' \
    --train_file_path='/workspace/Coding/lm-trainer/datasets/pre_datasets/hate_speech_data/hate_speech_total_70000.csv' \
    --output_dir='/workspace/Coding/lm-trainer/model_records/hate_speech_koelectra_v2' \
    --text_column_name='text' \
    --label_column_name='label' \
    --num_labels=2 \
    --per_device_train_batch_size=512 \
    --per_device_eval_batch_size=1024 \
    --epochs=5 \

    