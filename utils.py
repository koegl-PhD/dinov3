from PIL import Image
import torch
import torchvision.transforms.functional as TF

PATCH_SIZE = 16
IMAGE_SIZE = 768


def load_model() -> torch.nn.Module:
    DINOV3_REPO_LOCATION = r"/home/koeglf/Documents/code/dinov3"
    DINOV3_MODEL_LOCATION = r"/home/koeglf/Downloads/dinov3_models/dinov3_vits16.pth"
    MODEL_NAME = "dinov3_vits16"

    model = torch.hub.load(
        repo_or_dir=DINOV3_REPO_LOCATION,   # local repo folder with hubconf.py
        model=MODEL_NAME,              # e.g. "dinov3_vits16"
        source="local",                 # never GitHub
        pretrained=False,               # don't fetch weights
        force_reload=False,             # don't refresh from remote
        skip_validation=True,           # avoid network checks
        trust_repo=True,                # suppress extra validation that can hit network
    )

    state = torch.load(DINOV3_MODEL_LOCATION, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.cuda()

    return model


def resize_transform(
    mask_image: Image,
    image_size: int = IMAGE_SIZE,
    patch_size: int = PATCH_SIZE,
) -> torch.Tensor:
    w, h = mask_image.size
    h_patches = int(image_size / patch_size)
    w_patches = int((w * image_size) / (h * patch_size))
    return TF.to_tensor(TF.resize(mask_image, (h_patches * patch_size, w_patches * patch_size)))
