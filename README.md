# EA-ASP-evaluation-dataset

This is a simulated dataset based on Voxceleb1, MUSAN, RIR. It is used for evaluation dataset of our paper **EA-ASP: Enroll-Aware Attentive Statistics Pooling for Target Speaker Verification**

You can directly use the configure file *mix_configure_sort_uniq*. Here are 3 datasets available:

**Vox1_Aug**: simulation of single-speaker speech in a noisy environment

**Vox1_Mix**: simulation of multi-speaker speech in a normal environment

**Vox1_MixAug**: simulation of multi-speaker speech in a noisy environment

For example, you can run the mix_utt code by `python mix_two_utt.py --trial_path mix_configure_sort_uniq --store_dir Mix_True_Aug_False --mix True --aug False`  to create the Vox1_Mix .


## Reference
```
@article{zhang2022enroll,
  title={Enroll-Aware Attentive Statistics Pooling for Target Speaker Verification$\}$$\}$},
  author={Zhang, Leying and Chen, Zhengyang and Qian, Yanmin},
  journal={Proc. Interspeech 2022},
  pages={311--315},
  year={2022}
}
```
