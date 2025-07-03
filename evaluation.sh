## DSRNet
CUDA_VISIBLE_DEVICES=6 python3 new_eval.py --experiment_name DSRNet --strategy "S1" --pre_model "./DSRNetModels/weights/dsrnet_l_4000_epoch33.pt"    
CUDA_VISIBLE_DEVICES=6 python3 new_eval.py --experiment_name DSRNet --strategy "S2" --pre_model "./DSRNetModels/weights/dsrnet_l_finetuning_on_clean_RDP_083_00332000.pt"     

## ERRNet
CUDA_VISIBLE_DEVICES=6 python3 new_eval.py --experiment_name ERRNet --strategy "S1" --pre_model "./ERRNetModels/weights/errnet_060_00463920_ori.pt"  
CUDA_VISIBLE_DEVICES=6 python3 new_eval.py --experiment_name ERRNet --strategy "S2" --pre_model "./ERRNetModels/weights/errnet_150_01231800_fin_clean.pt"  

## RAGNet
CUDA_VISIBLE_DEVICES=6 python3 new_eval.py --experiment_name RAGNet --strategy "S1" --pre_model "./RAGNetModels/weights/pretrain.pth"  
CUDA_VISIBLE_DEVICES=6 python3 new_eval.py --experiment_name RAGNet --strategy "S2" --pre_model "./RAGNetModels/weights/epoch_090_G_1.296_P_24.598.pth"  

## RDNetRRNet
CUDA_VISIBLE_DEVICES=6 python3 new_eval.py --experiment_name RDNetRRNet --strategy "S1" --pre_model "./RDNetRRNetModels/weights/RR.pth" --pre_model_Det "./RDNetRRNetModels/weights/RD.pth"  
CUDA_VISIBLE_DEVICES=6 python3 new_eval.py --experiment_name RDNetRRNet --strategy "S2" --pre_model "./RDNetRRNetModels/weights/RR_ft195-clean.pth" --pre_model_Det "./RDNetRRNetModels/weights/RD_ft195-clean.pth"  

## BaselineModel
CUDA_VISIBLE_DEVICES=6 python3 new_eval.py --experiment_name BaselineModel --strategy "S1" --pre_model "./BaselineModels/weights/errnet_stg1.pt"  
CUDA_VISIBLE_DEVICES=6 python3 new_eval.py --experiment_name BaselineModel --strategy "S2" --pre_model "./BaselineModels/weights/errnet_stg2_140_00240820.pt"  