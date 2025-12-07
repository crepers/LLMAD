# retriever.py
import heapq
import numpy as np
from fastdtw import fastdtw

def fast_dtw_distance(series1, series2, dist_div_len=False):
    series1 = np.array(series1, dtype=float)
    series2 = np.array(series2, dtype=float)

    distance, path = fastdtw(series1, series2, dist=lambda x, y: np.linalg.norm(x - y))
    if dist_div_len:
        avg_length = (len(series1) + len(series2)) / 2
        if avg_length > 0:
            distance /= avg_length
    return distance

def find_most_similar_series_fast(X, T_list, top_k=1, dist_div_len=False):
    # Check if the list/array of series is empty
    if len(T_list) == 0 or top_k <= 0:
        return [], [], []
        
    heap = []
    for idx, Y in enumerate(T_list):
        score = fast_dtw_distance(X, Y, dist_div_len)
        
        if len(heap) < top_k:
            heapq.heappush(heap, (-score, idx, Y))
        else:
            heapq.heappushpop(heap, (-score, idx, Y))
            
    top_k_series, top_k_scores, top_k_indices = [], [], []
    for score, idx, series in sorted(heap, reverse=True):
        top_k_scores.append(-score)
        top_k_indices.append(idx)
        top_k_series.append(series)
        
    return top_k_series, top_k_scores, top_k_indices
