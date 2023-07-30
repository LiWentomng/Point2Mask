import numpy as np
import time
from pydijkstra import dijkstra2d, dijkstra_image
import torch
import torch.nn.functional as F

class AffGraph:
    def __init__(self, data):
        self.data = data

    def get(self, x1, x2):
        i1, j1 = x1
        i2, j2 = x2
        if abs(i1 - i2) > 1 or abs(j1 - j2) > 1:
            return float('inf')
        if (i1 == i2) and (j1 == j2):
            return 0
        return abs(self.data[i1, j1] - self.data[i2, j2])

    def __call__(self, *args):
        return self.get(*args)

def find_min(cands):
    value = cands[0][1]
    index = 0
    for i, (_, v) in enumerate(cands[1:], 1):
        if v < value:
            index = i
            value = v
    return index

def dijkstra_python(data, source_coords):
    h, w = data.shape
    graph = AffGraph(data)
    outs = []
    for si, sj in source_coords:
        out = np.full((h, w), float('inf'), np.float32)
        #cands = [[(i, j), graph((i, j), (si, sj))] for i in range(h) for j in range(w)]
        cands = [[(i, j), graph((i, j), (si, sj))] for j in range(w) for i in range(h)]
        while len(cands) > 0:
            min_idx = find_min(cands)
            (i, j), v = cands[min_idx]
            out[i, j] = v

            del cands[min_idx]
            for idx in range(len(cands)):
                (ci, cj), cv = cands[idx]
                cands[idx][1] = min(cv, v + graph((i, j), (ci, cj)))

        outs.append(out)
    return np.array(outs)

def diff8(x, kernel_size, dilation):
    isnumpy = isinstance(x, np.ndarray)
    if isnumpy:
        x = torch.as_tensor(x).float()
    assert x.ndim == 4, x.shape
    assert kernel_size % 2 == 1, kernel_size
    padding = (kernel_size + (dilation - 1) * (kernel_size - 1)) // 2
    unfold_x = F.unfold(x, kernel_size=kernel_size, dilation=dilation, padding=padding)

    n, c, h, w = x.shape
    unfold_x = unfold_x.reshape(n, c, -1, h, w)

    size = kernel_size**2
    out = torch.cat((unfold_x[:, :, :size//2], unfold_x[:, :, size//2 + 1:]), 2) - unfold_x[:, :, size//2:size//2+1]

    if isnumpy:
        out = out.data.cpu().numpy()
    return out

def pydijkstra_image(data, source_coords):
    if data.ndim == 2:
        data = data[..., None]
    data = diff8(data.transpose(2, 0, 1)[None], 3, 1)[0]
    data = np.abs(data).sum(0).transpose(1, 2, 0)
    t1 = time.time()
    out = dijkstra_image(data, source_coords)
    t2 = time.time()
    return out, t1, t2

if __name__ == '__main__':
    np.random.seed(1)
    size = 100
    npoints = 10
    rand_map = (np.random.rand(size, size) * 10).astype(int)
    source_coords = np.random.randint(2, size-2, (npoints, 2))

    t0 = time.time()
    #outs = dijkstra_python(rand_map, source_coords)
    t1 = time.time()
    outs2 = dijkstra2d(rand_map[..., None], np.array(source_coords))
    t2 = time.time()

    outs3, t3, t4 = pydijkstra_image(rand_map[..., None], np.array(source_coords))

    print(t1 - t0, t2 - t1, t4 - t3)
    print((outs2 != outs3).sum((1, 2)))
    print(rand_map, '\n')
    i = 1
    print(source_coords[i])

