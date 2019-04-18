
#V1
#python3.5 dnn_model.py --batch_size 400 --train_steps 100000 --hidden_units 400x400x20  --nr_epochs 0 --choosen_label repaired --label_path Data2/V1/ --data_path Data2/V1/ --compressed_data True --max_nr_nan 0 --fixed_selection False --suffix V1_01_1

#V1 half repaired
#python3.5 dnn_model.py --batch_size 400 --train_steps 100000 --hidden_units 400x400x20  --nr_epochs 0 --choosen_label repaired --label_path Data2/V1/Half_repaired/ --data_path Data2/V1/Half_repaired/ --compressed_data True --max_nr_nan 0 --fixed_selection False --suffix V1_half_01_1






#V1 valid half repaired
#python3.5 dnn_model.py --batch_size 100 --train_steps 50000 --hidden_units 400x400x20  --nr_epochs 0 --choosen_label repaired --data_path Data2/V1/Half_repaired/ --max_nr_nan 0 --fixed_selection False --suffix DNN_Kfold5_V1_V3_Ex_4

#V3
#python3.5 dnn_model.py --batch_size 100 --train_steps 25000 --hidden_units 250x250x20  --nr_epochs 0 --choosen_label repaired --data_path Data2/V3/ --max_nr_nan 0 --fixed_selection True --suffix V3_Zero_250_4




# Validate program
#python3.5 dnn_model_validate.py --batch_size 100 --train_steps 100000 --hidden_units 400x400x200x20 --choosen_label repaired --data_path Data2/V3/ --suffix V3_Validate



# Cnn
#python3.5 cnn_model.py --batch_size 100 --train_steps 10000  --nr_epochs 0 --choosen_label repaired --data_path Data2/V1/Half_repaired/ --fixed_selection False --suffix CNN_Kfold5_Normal_800_CL2_ADAGR_01_LL20SM_NP_DR02_1234_DIL2_dnn5_4


# Validate program CNN
python3.5 cnn_model_validate.py --choosen_label repaired --data_path Data2/V3/V32_Validate/ --suffix V32_Validate_Pred
