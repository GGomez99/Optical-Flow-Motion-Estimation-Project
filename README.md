# Optical-Flow-Motion-Estimation-Project

In this repository, we used the Horn and Schunk, Farneback and Raft models in order to compute the optical flow of an image sequence and then propagate a mask to perform object detection.
You can find these three models in the `models` folder.

We have the following files:
- `flow.py` that computes the optical flows for a given image sequence.
- `segment.py` that performs the mask propagation with the direct and sequential method.
- `main processing.ipynb` which is the main files and perform all the computation.
- Finally, `results scoring.py` to compute the results for the sequences.
