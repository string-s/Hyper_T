# THCL model
## 文件说明
- data: 数据集格式示例
- re_label.py：原始数据处理
- data_preprocessing.py：数据预处理（建图）
- model.py
- train.py
## 模型训练
python train.py   --data_dir /root/autodl-tmp/MIMI-CIII   --save_dir ./mimic_processed   --batch_size 128   --emb_dim 256   --tgat_heads 8   --epochs 150   --lr 2e-4   --weight_decay 1e-5 
