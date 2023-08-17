# MFS-Seg
Multi-scale Full Spike Pattern for Semantic Segmentation

This is the official repository for our work;
The _ddd17 folder is used to process the ddd17 dataset. 
(1) Setup this code  with pytorch=1.10.1,py3.8, cuda11.3, cudnn8.2.0_0 
pip install apex,tensorBoard

(2) Usage
Training/Evaluation
Download the Camvid dataset http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/
bash multiscale_fullspike_seg/MIMO-UNet-Inverse.sh
--model_name "choose in the models dir" \
--mode "train,val,test" \
--exp_name "create new model save-file here" \
--data_dir "dataset dir" \
--resume "model.pkl" \
--attention_map_dir "spike map dir"

(3)Visualize
It can be downloaded from https://github.com/luo3300612/Visualizer 
Thanks for their nice contribution.
