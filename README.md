# Quantization
A simple demo for the illustration of the quantization of NNs.  
Top file: train_mnist_nas.py  

## Training step

### Step 1: Training
Run the “model_train()” function in debug mode and set a breakpoint at “hook = 0”.  
After the finish of training, the code will be stuck at the breakpoint.  
Then, please use the “dump_file()” function to dump the quantized INT8 model.
### Step 2: Verification
If you need to test the quantized INT8 model, please set correct model file and run the “model_eval()” function. 
### Step 3: Extraction
If you want to analyze the dumped file, please use the “parse_hand_dumped_file()” function to decode the dumped file.  
Some relevant .txt files could be generated.






































This quantization method has already been used in the following publications:

@ARTICLE{9793397,  author={Huang, Mingqiang and Liu, Yucen and Man, Changhai and Li, Kai and Cheng, Quan and Mao, Wei and Yu, Hao},  journal={IEEE Transactions on Circuits and Systems I: Regular Papers},   title={A High Performance Multi-Bit-Width Booth Vector Systolic Accelerator for NAS Optimized Deep Learning Neural Networks},   year={2022},  volume={},  number={},  pages={1-13},  doi={10.1109/TCSI.2022.3178474}}

@ARTICLE{9997088,  author={Cheng, Quan and Dai, Liuyao and Huang, Mingqiang and Shen, Ao and Mao, Wei and Hashimoto, Masanori and Yu, Hao},  journal={IEEE Transactions on Circuits and Systems II: Express Briefs},   title={A Low-Power Sparse Convolutional Neural Network Accelerator with Pre-Encoding Radix-4 Booth Multiplier},   year={2022},  volume={},  number={},  pages={1-1},  doi={10.1109/TCSII.2022.3231361}}
