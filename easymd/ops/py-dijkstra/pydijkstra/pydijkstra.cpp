#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <limits>
#include <queue>
#include <vector>
#include <omp.h>

namespace py = pybind11;
using namespace py::literals;

typedef float float32;
typedef int int32;

inline void ASSERT(bool cond, const std::string &msg="") {
    if (!cond)
        throw std::runtime_error(msg + "\n");
};

struct Item {
    Item(int i, int j, float32 v) : i(i), j(j), v(v) {};
    int i, j;
    float32 v;
    bool operator>(const Item &other) const {return this->v > other.v;}
    bool operator<(const Item &other) const {return this->v < other.v;}
    bool operator>=(const Item &other) const {return this->v >= other.v;}
    bool operator<=(const Item &other) const {return this->v <= other.v;}
};

class Graph {
public:
    Graph(const float32* data, int h, int w, int d, float32 nd) : data(data), h(h), w(w), d(d), nd(nd) {}
    float32 get(int i1, int j1, int i2, int j2) const {
        if ((std::abs(i1 - i2) > 1) || (std::abs(j1 - j2) > 1))
            return std::numeric_limits<float32>::max();
        if ((i1 == i2) && (j1 == j2))
            return 0;
        // if ((i1 < 0) or (j1 < 0) or (i2 < 0) or (j2 < 0) or (i1 >= h) or (i2 >= h) or (j1 >= w) or (j2 >= w))
        //     return std::numeric_limits<float32>::max();
        float32 dist = nd;
        if (std::abs(i1 - i2) + std::abs(j1 - j2) > 1)
            dist *= std::sqrt(2);
        const float32* val1 = data + (i1 * w + j1) * d;
        const float32* val2 = data + (i2 * w + j2) * d;
        for (int k = 0; k < d; ++k)
            dist += std::abs(val1[k] - val2[k]);
        return dist;
    }
private:
    const float32* data;
    float32 nd;
    int h, w, d;
};

class ImageGraph {
    public:
        ImageGraph(const float32* data, int h, int w) : data(data), h(h), w(w) {}
        bool isconnect(int i1, int j1, int i2, int j2) const {
            int di = std::abs(i1 - i2);
            int dj = std::abs(j1 - j2);
            return (di + dj) > 0 && (di + dj) < 3;
        }
        float32 get(int i1, int j1, int i2, int j2) const {
            // if (!isconnect(i1, j1, i2, j2))
            //     return std::numeric_limits<float32>::max();
            int di = i2 - i1;
            int dj = j2 - j1;
            int index = di * 3 + dj + 4;
            index = index - ((index > 4) ? 1 : 0);
            if ((index < 0) || (index > 7))
                throw std::runtime_error("Index = " + std::to_string(index));
            return data[(i1 * w + j1) * 8 + index];
        }

    private:
        const float32* data;
        int h, w;
};

class MinHeap {
    public:
        MinHeap(const int& h, const int &w) : h(h), w(w) {
            indices = new size_t[h * w];
            for (int i = 0; i < h * w; ++i)
                indices[i] = h * w;
        }
        ~MinHeap() {
            delete[] indices;
        }

        size_t size() const {return data.size();}
        size_t parent(const size_t &i) const {return (i - 1) / 2;};
        size_t child(const size_t &i) const {return i * 2 + 1;};

        void print() const {
            std::string msg = "MinHeap.size() = " + std::to_string(size()) + "\n";
            for (int i = 0; i < size(); ++i) {
                const Item &item = data[i];
                const size_t idx = this->get_index(item.i, item.j);
                msg += "(" + std::to_string(item.i) + ", " + std::to_string(item.j) + ", " + 
                    (item.v < 1e10 ? std::to_string(item.v) : "inf") + ", " +
                    std::to_string(idx) + "), ";
                if (i % 6 == 0)
                    msg += "\n";
            }
            msg += "\n";
            std::cout << msg;
        }

        void push(const Item &item) {
            data.push_back(item);
            int idx = item.i * w + item.j;
            indices[idx] = size() - 1;
            up(size() - 1);
        }

        void swap(const size_t &i, const size_t &j) {
            int idx_i = data[i].i * w + data[i].j;
            int idx_j = data[j].i * w + data[j].j;
            std::swap(indices[idx_i], indices[idx_j]);
            std::swap(data[i], data[j]);
        }

        void down(const size_t &i) {
            size_t left = child(i);
            if (left >= size())
                return;
            size_t right = left + 1;
            size_t j;
            if (right >= size())
                j = left;
            else
                j = data[left] < data[right] ? left : right;
            if (data[i] <= data[j])
                return;
            swap(i, j);
            down(j);
        }

        void up(const size_t &i) {
            size_t j = parent(i);
            if (data[i] >= data[j])
                return;
            swap(i, j);
            up(j);
        }

        void update(const size_t &i, const float32 &v) {
            if (v == data[i].v)
                return;
            data[i].v = v;
            up(i);
            down(i);
        }

        void update(const int &i, const int &j, const float32 &v) {
            size_t idx = indices[i * w + j];
            if (idx == h * w)
                throw std::runtime_error("idx = " + std::to_string(idx));
            update(idx, v);
        }

        size_t get_index(const int &i, const int &j) const {
            return indices[i * w + j];
        }

        void pop() {
            swap(0, size() - 1);
            data.pop_back();
            down(0);
        }

        const Item& top() const {
            return data[0];
        }

        Item& at(const size_t &i) {
            return data[i];
        }


    private:
        std::vector<Item> data;
        size_t* indices;
        int h, w;
};

py::array_t<float32> dijkstra_image(
        const py::array_t<float32, py::array::c_style | py::array::forcecast> &input,
        const py::array_t<int32, py::array::c_style | py::array::forcecast> &sources) {
    // input: [h, w, 8/4]
    py::buffer_info input_buf = input.request();
    float32* input_data = static_cast<float32*>(input_buf.ptr);
    ASSERT(input_buf.ndim == 3, "Input should have dim(3), given(" + std::to_string(input_buf.ndim) + ")");
    const int h = input_buf.shape[0];
    const int w = input_buf.shape[1];
    const int d = input_buf.shape[2];
    ASSERT(d == 8, "Diff dim should be (8), given (" + std::to_string(d) + ")");

    // source points coordinates, [n, 2]
    py::buffer_info sources_buf = sources.request();
    int32* sources_data = static_cast<int32*>(sources_buf.ptr);
    ASSERT(sources_buf.ndim == 2, "Sources should have dim(2), given(" + std::to_string(sources_buf.ndim) + ")");
    const int num_sources = sources_buf.shape[0];
    ASSERT(sources_buf.shape[1] == 2, "Sources should be [n, (2)], given (" + std::to_string(sources_buf.shape[1]) + ")");

    // output: [n, h, w]
    py::array_t<float32> output({num_sources, h, w});
    py::buffer_info output_buf = output.request();
    float32* output_data = static_cast<float32*>(output_buf.ptr);

    // compute
    ImageGraph graph(input_data, h, w);
    for (int n = 0; n < num_sources; ++n) {
        int si = sources_data[n * 2];
        int sj = sources_data[n * 2 + 1];

        MinHeap minheap(h, w);
        for (int i = 0; i < h; ++i) {
            for (int j = 0; j < w; ++j) {
                float32 v = ((i == si) && (j == sj)) ? 0 : std::numeric_limits<float32>::max();
                minheap.push(Item(i, j, v));
            }
        }

        while (minheap.size() > 0) {
            const Item item = minheap.top();
            output_data[n * h * w + item.i * w + item.j] = item.v;
            minheap.pop();

            for (int di : {-1, 0, 1}) {
                for (int dj : {-1, 0, 1}) {
                    if ((di == 0) && (dj == 0))
                        continue;
                    int ii = item.i + di;
                    int jj = item.j + dj;
                    if ((ii < 0) || (ii > h-1) || (jj < 0) || (jj > w - 1))
                        continue;

                    size_t idx = minheap.get_index(ii, jj);
                    Item &other = minheap.at(idx);
                    minheap.update(idx, std::min(other.v, item.v+graph.get(item.i, item.j, ii, jj)));
                }
            }
        }
    }
    return output;
}


py::array_t<float32> dijkstra2d(const py::array_t<float32> &input, const py::array_t<int32> sources, const float32 neighbour_distance=0) {
    // data of shape [h, w, d]
    py::buffer_info input_buf = input.request();
    if (input_buf.ndim != 3)
        throw std::runtime_error("Input should have dim(3), given(" + std::to_string(input_buf.ndim) + ")\n");
    const int h = input_buf.shape[0];
    const int w = input_buf.shape[1];
    const int d = input_buf.shape[2];
    float32* input_data = static_cast<float32*>(input_buf.ptr);
    
    // source points of shape [n, 2]
    py::buffer_info sources_buf = sources.request();
    if (sources_buf.ndim != 2)
        throw std::runtime_error("Souces should have dim(2), given(" + std::to_string(sources_buf.ndim) + ")\n");
    if (sources_buf.shape[1] != 2)
        throw std::runtime_error("Sources should have shape(2) at dim(1), given(" + std::to_string(sources_buf.shape[1]) + ")\n");
    int32* sources_data = static_cast<int32*>(sources_buf.ptr);
    const int num_sources = sources_buf.shape[0];

    // output of shape [n, h, w]
    py::array_t<float32> output({num_sources, h, w});
    py::buffer_info output_buf = output.request();
    float32* output_data = static_cast<float32*>(output_buf.ptr);

    Graph graph(input_data, h, w, d, neighbour_distance);

    auto find_min = [](const std::vector<Item> &cands) {
        size_t min_idx = 0;
        float32 v = cands[0].v;
        for (size_t i = 1; i < cands.size(); ++i) {
            if (cands[i].v < v) {
                min_idx = i;
                v = cands[i].v;
            }
        }
        return min_idx;
    };

    for (int n = 0; n < num_sources; ++n) {
        int si = sources_data[n * 2];
        int sj = sources_data[n * 2 + 1];
        std::vector<Item> cands;
        for (int i = 0; i < h; ++i)
            for (int j = 0; j < w; ++j)
                cands.emplace_back(i, j, graph.get(i, j, si, sj));

        while (cands.size() > 0) {
            size_t idx = find_min(cands);
            const Item &item = cands[idx];

            output_data[n * h * w + item.i * w + item.j] = item.v;
            for (Item &cand_item : cands) {
                cand_item.v = std::min(cand_item.v, item.v + graph.get(item.i, item.j, cand_item.i, cand_item.j));
            }
            cands.erase(cands.begin() + idx);
        }
    }
    return output;
}

PYBIND11_MODULE(pydijkstra, m) {
    m.def("dijkstra2d", &dijkstra2d, "input"_a, "sources"_a, "neighbour_distance"_a=0);
    m.def("dijkstra_image", &dijkstra_image, "input"_a, "sources"_a);
}

