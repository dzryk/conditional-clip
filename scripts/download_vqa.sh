#!/bin/bash

download_dir='/data'

wget -P ${download_dir} https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip
wget -P ${download_dir} https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip
wget -P ${download_dir} https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip
wget -P ${download_dir} https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip
wget -P ${download_dir} http://images.cocodataset.org/zips/train2014.zip
wget -P ${download_dir} http://images.cocodataset.org/zips/val2014.zip

unzip ${download_dir}/v2_Annotations_Train_mscoco.zip -d ${download_dir}
unzip ${download_dir}/v2_Questions_Train_mscoco.zip -d ${download_dir}
unzip ${download_dir}/v2_Annotations_Val_mscoco.zip -d ${download_dir}
unzip ${download_dir}/v2_Questions_Val_mscoco.zip -d ${download_dir}
unzip ${download_dir}/train2014.zip -d ${download_dir}
unzip ${download_dir}/val2014.zip -d ${download_dir}

rm ${download_dir}/v2_Annotations_Train_mscoco.zip
rm ${download_dir}/v2_Questions_Train_mscoco.zip
rm ${download_dir}/v2_Annotations_Val_mscoco.zip
rm ${download_dir}/v2_Questions_Val_mscoco.zip
rm ${download_dir}/train2014.zip
rm ${download_dir}/val2014.zip

python3 vqa.py --datadir=${download_dir}