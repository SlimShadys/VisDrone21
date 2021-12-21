# VisDrone21
## Crowd counting through drones using VisDrone Dataset / Neural Networks via TransCrowd transformer

The VisDrone Dataset could be found [here](https://github.com/VisDrone/VisDrone-Dataset).

- Task 5: Crowd Counting

The TransCrowd implementation can be found under the directory "*/TransCrowd*".

For official implementation, have a look [here](https://github.com/dk-liang/TransCrowd).

# Run the notebook
You can run the whole process via the *Google Colab Notebook* that can be found [here](https://colab.research.google.com/drive/1GICGGQyNmUErkBJ2iHDBGvVVg6AiJ37_).

It is highly suggested to run the notebook on a *Pro* account, since it requires at least **16GB** of RAM (Colab Free gives **12GB**).

# Credits
- Gianmarco Scarano (*Owner*)
- Pasquale De Marinis (*Test annotations converter*)
- Credits for TransCrowd implementation:
```
@article{liang2021transcrowd,
  title={TransCrowd: Weakly-Supervised Crowd Counting with Transformer},
  author={Liang, Dingkang and Chen, Xiwu and Xu, Wei and Zhou, Yu and Bai, Xiang},
  journal={arXiv preprint arXiv:2104.09116},
  year={2021}
}
```
__________________________________________________________________

### Normal images vs Density maps
<p align="center">
  <img alt="Normal images" src="https://github.com/SlimShadys/VisDrone21/blob/GPU/GIFs/Normal.gif" width="45%"> 
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="Density maps" src="https://github.com/SlimShadys/VisDrone21/blob/GPU/GIFs/Density.gif" width="45%">
</p>