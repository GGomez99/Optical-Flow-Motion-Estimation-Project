# Optical-Flow-Motion-Estimation-Project

## Setup

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
| `raft`  | Direct integration with RAFT method  |
| `HS`  | Direct integration with Horn-Schunck method  |
| `Fa`  | Direct integration with Farneback method  |
| `LK`  | Direct integration with Lucas-Kanade method  |
| `seq-raft`  | Sequential integration with RAFT method  |
| `seq-HS`  | Sequential integration with Horn-Schunck method  |
| `seq-Fa`  | Sequential integration with Farneback method  |
| `seq-LK`  | Sequential integration with Lucas-Kanade method  |

### Usage example
Example for computing the octopus mask propagation using a sequential integration with Farneback algorithm:
```
python flow.py ./data octopus seq-Fa
```

## Internal file structure
We have the following files:
- `flow.py` that computes the optical flows for a given image sequence.
- `segment.py` that performs the mask propagation with the direct and sequential method.
- `main processing.ipynb` which is a notebook file aggregating flow computation, mask integration, and post processing. One should probably start by this file to experiment with this repository.
- Finally, `results scoring.py` to compute the results for the sequences.