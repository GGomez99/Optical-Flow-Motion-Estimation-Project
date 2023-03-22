# Optical-Flow-Motion-Estimation-Project

In this repository, we used the Horn and Schunk, Farneback and Raft models in order to compute some optical flows and then propagate some masks to perform object detection.
You can fing these three models in the `models` folder.

We have the following fildes:
- `flow.py` that computes the optical flows the given image sequence.
- `segment.py` that performs the mask propagation in the direct and sequential way.
- `main processing.ipynb` which is the main files and perform all the computation.
- Finally, `results scoring.py` to compute the results for the sequences.
