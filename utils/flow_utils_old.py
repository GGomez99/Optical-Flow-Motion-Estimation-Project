import numpy as np
from scipy.interpolate import interp2d


def propagate_mask(flow, mask_begin):
    new_mask = np.zeros(shape=mask_begin.shape[:2])
    for x in range(mask_begin.shape[0]):
        for y in range(mask_begin.shape[1]):
            x_, y_ = np.rint(x + flow[x, y, 1]).astype(int), np.rint(y + flow[x, y, 0]).astype(int)
            if (x_ >= 0) and (x_ < mask_begin.shape[0]) and (y_ >= 0) and (y_ < mask_begin.shape[1]):
                if mask_begin[x, y] > 0:
                    new_mask[x_, y_] = 255
    return new_mask.astype(np.uint8)


def flow_concatenation(unary_flow, to_ref_flow):
    flow = np.zeros((unary_flow.shape[0], unary_flow.shape[1], 2), dtype=np.float64)
    x0 = np.arange(0, unary_flow.shape[0])
    y0 = np.arange(0, unary_flow.shape[1])
    xx, yy = np.meshgrid(x0, y0)
    z = unary_flow[xx, yy, 1]
    fx = interp2d(x0, y0, z, kind='cubic')
    z = unary_flow[xx, yy, 0]
    fy = interp2d(x0, y0, z, kind='cubic')
    for x in range(unary_flow.shape[0]):
        for y in range(unary_flow.shape[1]):
            flow_x = fx(x + to_ref_flow[x, y, 1], y + to_ref_flow[x, y, 0])
            flow_y = fy(x + to_ref_flow[x, y, 1], y + to_ref_flow[x, y, 0])
            flow[x, y, 1] = to_ref_flow[x, y, 1] + flow_x
            flow[x, y, 0] = to_ref_flow[x, y, 0] + flow_y
    return flow

