from dinov3.data.datasets import ImageNet

for split in ImageNet.Split:
    dataset = ImageNet(split=split, root="/home/fryderyk/.cache/huggingface/datasets/mlx-vision___imagenet-1k", extra="<EXTRA>")
    dataset.dump_extra()