# VisDrone21
Crowd counting through drones using VisDrone Dataset / Neural Networks<br />

The VisDrone Dataset could be found [here](https://github.com/VisDrone/VisDrone-Dataset).

# Prepare the dataset
```
python gt_generate_visdrone20.py
```

# Generate image list
```
python TransCrowd\make_npydata.py
```

# Training
```
python TransCrowd/train.py --dataset VisDrone --save_path ./save_file/VisDrone --batch_size X --model_type 'token' --epoch X --print_freq 5
```

- Gianmarco Scarano
- Pasquale De Marinis (Test annotations converter)
- Credits for TransCrowd implementation:
```
@article{liang2021transcrowd,
  title={TransCrowd: Weakly-Supervised Crowd Counting with Transformer},
  author={Liang, Dingkang and Chen, Xiwu and Xu, Wei and Zhou, Yu and Bai, Xiang},
  journal={arXiv preprint arXiv:2104.09116},
  year={2021}
}
```

// *More credits to be written soon*
__________________________________________________________________

![MicrosoftTeams-image](https://user-images.githubusercontent.com/6324754/141682229-290c7fbc-14d3-4f0e-ab76-5eed982b76b0.png)
