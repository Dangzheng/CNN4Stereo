# Efficient Deep Learning for Stereo Matching


## Using pretrained model
1. We include pretrained model of KITTI2015 and KITTI2012 in **pretrain**. To use pretrained model of KITTI2015(similar for KITTI2012), run:

	`th inference_match_subimg.lua -g 0 --model split_win37_dep9 --data_version kitti2015 --data_root pretrain/kitti2015/sample_img --model_param pretrain/kitti2015/param.t7 --bn_meanstd pretrain/kitti2015/bn_meanstd.t7 --saveDir outImg --start_id 1 --n 1`
2. Results of unary images will be save in **outImg**


## Training

### Prepare data
1. Modify variable **data_root** in **preprocess/kitti2015_gene_loc_1.m** with the path to corresponding training folder.
2. Go to preprocess folder and use matlab to run: `kitti2015_gene_loc_1(160,40,18,100,'debug_15',123)` to generate three binary files(~300MB total), corresponding to pixel locations you want to train and validate on. 
	
	Parameters: 160 is number of images to train on, 40 is number of image to validate on, 18 represents size of image patch with (2x18+1) by (2x18+1), 100 represents searching range(disparity range to train on, corresponding to 2x100+1), 'debug_15' is the folder to save results, 123 is the random seed.
	
### Running training script
1. Install/update torch
2. A running script example:
	
	`th train_match.lua -g 0 --tr_num 160 --val_num 40 --data_version kitti2015 -s logs/debug --model dot_win37_dep9 --psz 18 --util_root preprocess/debug_15 --data_root /ais/gobi3/datasets/kitti/scene_flow/training`
	
	remember to change **util_root**(the one specified in preprocess step) and **data_root** to proper directory. Notice this is trainng on only 160 images and will do validation on the remaining images.
	
	use `th train_match.lua -h` for more detailed explanation, and use corresponding parameters to train longer for better performance.
	
3. A training will be saved in **logs/debug** by default as well as model parameter and batch normalization statistics from training.

## Testing
1. Sample running script is:

	`th inference_match_subimg.lua -g 0 --model split_win37_dep9 --data_version kitti2015 --data_root /ais/gobi3/datasets/kitti/scene_flow/training --perm_fn preprocess/debug_15/myPerm.bin --model_param logs/debug_15/param_epoch_10.t7 --bn_meanstd logs/debug_15/param_epoch_10.t7 --saveDir outImg --start_id 161 --n 1`

	remember to change **data_root** as proper directory for images, **perm_fn** for permutation on file list(generated automatically from preprcess script), **model_param** for parameters, **bn_meanstd** for batch normalization statistics and **start_id** for image id for validation(since we are training on 160 images, we validate on images from 161th). It should take less than a second for one image.
	
	use `th inference_match_subimg.lua -h` for more detailed explanation.

2. Results will be saved in **outImg** by default. You should see image from unary for both left and right image.
3. No postprocess by default. Additional libs required.


## Misc
1. Apply same steps for running on KITTI 2012 stereo dataset.
2. For postprocessing, you need to setup corresponding code from [MC-CNN](https://github.com/jzbontar/mc-cnn) or [SPS](http://ttic.uchicago.edu/~dmcallester/SPS/index.html). By default, the inference code will output unary images for both left and right image.

## License
This code is licensed under GPL-3.0. If you use our code in your research, please cite our paper as:


	@inproceedings{luo16a,
  		title = {Efficient Deep Learning for Stereo Matching},
		author = {Luo, W. and Schwing, A. and Urtasun, R.},
  		booktitle = {International Conference on Computer Vision and Pattern Recognition (CVPR)},
  		year = {2016},
	}
