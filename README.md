# Joint Bilateral Learning

This repository is an unofficial implementation in PyTorch for the paper:

[Joint Bilateral Learning for Real-time Universal Photorealistic Style Transfer](https://arxiv.org/abs/2004.10955)



### Dependencies

- Python 3.7.2
- PyTorch 1.2
- CUDA10.0 and cuDNN



### Train

```
$ python main.py --cont_img_path <path/to/cont_img> --style_img_path <path/to/style_img> --batch_size 8
```



### Test

```
$ python test.py --cont_img_path <path/to/single_cont_img> --style_img_path <path/to/single_style_img> --model_checkpoint <path/to/model checkpoint>
```


