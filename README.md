# Optical-Flow-Motion-Estimation-Project

## Setup

We recommend you to setup a python (conda) environment for this repository. Once in the environment, this repository requirements are installed using :

```
pip install -r requirements.txt
```

## Content of the repository

In this repository, we used the Horn and Schunk, Farneback, Lucas-Kanade and Raft models in order to compute the optical flow of an image sequence and then propagate a mask to perform object detection.
You can find these models in the `models` folder.

## Folder structure
Put every images (including original images, and masks) inside `data/sequences-train`. <br />
Only the first mask is required to perform the mask integration, however, all the masks should be included when computing the metrics.

## Flow computation
Using the `flow.py`, one can compute optical flow for any sequence of images.

```
python flow.py ./data <sequence name> <method name>
```

`<sequence name>` referes to the name of the sequence, examples are `swan`, `octopus`, ... . <br />
`<method name>` referes to the name of the method, available methods are as follows :
| Method name  | Optical flow method description |
| ------------- | ------------- |
| `direct-raft`  | Direct integration with RAFT method  |
| `direct-HS`  | Direct integration with Horn-Schunck method  |
| `direct-Fa`  | Direct integration with Farneback method  |
| `direct-LK`  | Direct integration with Lucas-Kanade method  |
| `seq-raft`  | Sequential integration with RAFT method  |
| `seq-HS`  | Sequential integration with Horn-Schunck method  |
| `seq-Fa`  | Sequential integration with Farneback method  |
| `seq-LK`  | Sequential integration with Lucas-Kanade method  |


---
_When the flow computation is done, the flows are saved in `data/flows-output`. This is compatible with other algorithms in this repo, which will look for the flows in the latter. You shouldn't change anything when using the whole mask integration pipeline end-to-end._
<br />
__In order to visualize the flow, image representations are also saved in `data/flows-img-outputs`.__

## Mask integration
Using the `segment.py`, one can integrate the segmentation mask, effectively segmenting the whole sequence.

```
python segment.py ./data <sequence name> <method name>
```

See table in the flow computation for different method names.

__Note__ : In order to perform mask segmentation, one should first compute the optical flow of the sequence (see section above).

---
_When the segmentation masks are computed, they are saved in `data/mask-output`._

## Post-processing
The post-processing code can be found in `utils/postprocess.py`.
See project presentation (slides) for more information about the post-processing. In a few words, it consists on avoiding masks residues using image erosion, and closing cracks in the masks using image dilatation.

## Process everything at once (notebook)
A convenience notebook is provided in `main processing.ipynb` that can be used to compute the flow computation and mask integration at once. Results are computed in another notebook, see following section.

## Results & scoring (notebook)
The code for aggregating results, after having computed everything (flows, mask integration, and post-processing. See sections above), can be found in the `results scoring.ipynb` notebook file.

## Usage example
Example for computing the octopus flow computation using a sequential integration with Farneback algorithm:
```
python flow.py ./data octopus seq-Fa
```

<br />
Example for computing the swan mask propagation using a direct integration with Horn-Schunck algorithm:

```
python segment.py ./data swan direct-raft
```

General use-cases can be found in the different notebooks (see sections above)