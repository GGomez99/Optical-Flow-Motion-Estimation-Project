import torch
from torchvision.models.optical_flow import raft_large, raft_small, Raft_Large_Weights, Raft_Small_Weights
import torchvision.transforms.functional as F

from tqdm import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#DEVICE = torch.device('cpu')
WEIGHTS = Raft_Large_Weights.DEFAULT
#WEIGHTS = Raft_Small_Weights.DEFAULT # Uncomment to use small model
TRANSFORMS = WEIGHTS.transforms()

def preprocess(images):
    """
    Preprocesses a batch of images.

    Args:
        images (torch.Tensor) : (n_images, h, w)
    """

    _, _, h, w = images.shape
    images = F.resize(images, size=[(h//8)*8, (w//8)*8], antialias=False) # w,h should be divisible by 8, RAFT model was trained without antialiasing
    return images

def postprocess(flows, h, w):
    """
    Postprocesses a batch of flows to have them back to their original sizes.

    Args:
        flows (torch.Tensor) : (batch_size, h, w, 2)
        h : original height
        w : original width
    """

    return F.resize(flows, size=[h, w], antialias=False)

def cleanup(flows):
    for f in flows[:-1]:
        f.detach()
        del f

    return flows[-1].detach().cpu()

def compute_flow_seq(images, batch_size=1):
    """
    Computes the flow sequentially on the given image sequence.
    RAFT model only accepts RGB images.

    Args:
        images (torch.Tensor) : (n_images, 3, h, w)
        batch_size (int) : batch size for the RAFT inference model.
    """

    n_images, _, h, w = images.shape

    # Preprocessing
    preprocessed = preprocess(images)

    img1 = preprocessed[:-1].to(DEVICE)
    img2 = preprocessed[1:].to(DEVICE)
    img1, img2 = TRANSFORMS(img1, img2) # Computes RAFT preprocessing transformations

    # Pushing model on device
    model = raft_large(weights=WEIGHTS, progress=True).to(DEVICE)
    #model = raft_small(weights=WEIGHTS, progress=True).to(DEVICE)  # Uncomment to use small model
    model = model.eval()

    print("Processing", n_images, "images")
    print("Batch size : ", batch_size)
    flows = []
    for i in tqdm(range(n_images // batch_size - 1), desc='RAFT'):
        # Pushes results on CPU to have VRAM available for the rest of the inferences.
        flows.append(model(img1[i*batch_size:(i+1)*batch_size], img2[i*batch_size:(i+1)*batch_size], num_flow_updates=12)[-1].detach().cpu())
    
    # Last batch
    # flows.append(model(img1[(n_images // batch_size - 1) * batch_size:], img2[(n_images // batch_size - 1) * batch_size:], num_flow_updates=12)[-1].detach().cpu())

    # Post-processing to retrieve original image sizes
    flows = postprocess(torch.cat(flows), h, w).to(DEVICE)
    return flows


def get_optimal_batch_size(h, w):
    """
    Computes the optimal batch_size using the available VRAM to maximize GPU usage.
    """

    if torch.cuda.is_available():
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = (r-a)/(1000*8)  # free inside reserved in Mo
    else:
        return 1 # CPU is sequential anyway

    needed = 2096 * (h * w / 102480)
    return max(1, int(f // needed))

    
def compute_flow_direct(images, batch_size=None):
    """
    Computes the flow directly on the given image sequence.
    RAFT model only accepts RGB images.

    Args:
        images (torch.Tensor) : (n_images, 3, h, w)
        batch_size (int) : batch size for the RAFT inference model.
    """

    n_images, _, h, w = images.shape

    if batch_size is None:
        if torch.cuda.is_available():
            print("Automatically computed batch size based on available VRAM")
            
        batch_size = get_optimal_batch_size(h, w)

    # Preprocessing
    preprocessed = preprocess(images)

    # first image is always the same
    img1 = torch.stack([preprocessed[0]] * (n_images-1)).to(DEVICE)
    img2 = preprocessed[1:].to(DEVICE)
    img1, img2 = TRANSFORMS(img1, img2)  # Computes RAFT preprocessing transformations
    n_images = n_images - 1 # We are considering pairs of images

    # Pushing model on device
    model = raft_large(weights=WEIGHTS, progress=True).to(DEVICE)
    #model = raft_small(weights=WEIGHTS, progress=True).to(DEVICE)  # Uncomment to use small model
    model = model.eval()

    print("Processing", n_images, "images")
    print("Batch size : ", batch_size)
    flows = []
    for i in tqdm(range(n_images // batch_size - 1), desc='RAFT'):
        # Pushes results on CPU to have VRAM available for the rest of the inferences.
        flows.append(model(img1[i*batch_size:(i+1)*batch_size], img2[i * batch_size:(i+1) * batch_size],
                           num_flow_updates=12)[-1].detach().cpu())

    # Last batch
    flows.append(model(img1[(n_images // batch_size - 1) * batch_size:], img2[(n_images // batch_size - 1) * batch_size:], num_flow_updates=12)[-1].detach().cpu())

    # Post-processing to retrieve original image sizes
    flows = postprocess(torch.cat(flows), h, w).to(DEVICE)
    return flows