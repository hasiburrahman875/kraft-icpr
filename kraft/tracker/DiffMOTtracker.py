# DiffMOTtracker.py (KDF + medium confidence consistency + deleted recovery)
# - KDF: KF + DiffMOT residual fusion with strict motion and scale clamps
# - Stage A: high-score association with IoU + FPC + optional appearance
# - Stage B: center rescue
# - Stage Del: deleted detection recovery (extra support source)
# - Stage M: medium association with confidence consistency only (IoU + conf)
# - Stage U: ultra-low support
# - Stage L: low-score IOU-only
# - ReID bridge preserved
# - Crash hardened with float32 contiguous buffers

import os
import time
import hashlib
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from models import *

from tracking_utils.kalman_filter import KalmanFilter
from tracking_utils.log import logger
from tracking_utils.utils import *

from tracker import matching

from .basetrack import BaseTrack, TrackState
from .cmc import CMCComputer
from .gmc import GMC
from .embedding import EmbeddingComputer

BIG = 1e6


# =========================
# Native handoff guards
# =========================
def _f32c(a):
    return np.ascontiguousarray(a, dtype=np.float32)


def _nan2big(a, big=BIG):
    return np.nan_to_num(a, nan=big, posinf=big, neginf=big)


# =========================
# Helpers: box conversions
# =========================
def _tlwh_to_xyxy(boxes_tlwh: np.ndarray) -> np.ndarray:
    boxes_tlwh = np.asarray(boxes_tlwh, dtype=np.float32)
    if boxes_tlwh.size == 0:
        return boxes_tlwh.reshape(0, 4)
    x, y, w, h = boxes_tlwh[:, 0], boxes_tlwh[:, 1], boxes_tlwh[:, 2], boxes_tlwh[:, 3]
    out = np.stack([x, y, x + w, y + h], axis=1).astype(np.float32)
    return _f32c(out)


def _xywh_to_tlbr(dets_xywh_score: np.ndarray) -> np.ndarray:
    """
    Input: Nx5 [x, y, w, h, score] absolute (top-left xywh)
    Output: Nx5 [x1, y1, x2, y2, score]
    """
    d = np.asarray(dets_xywh_score, dtype=np.float32).reshape(-1, 5)
    if d.size == 0:
        return d.reshape(0, 5)
    out = d.copy()
    out[:, 2] = out[:, 0] + out[:, 2]
    out[:, 3] = out[:, 1] + out[:, 3]
    return _f32c(out)


def compute_centers(tlwh_boxes: np.ndarray) -> np.ndarray:
    tlwh_boxes = np.asarray(tlwh_boxes, dtype=np.float32)
    if tlwh_boxes.size == 0:
        return tlwh_boxes.reshape(0, 2)
    c = tlwh_boxes.copy()
    c[:, 0] = c[:, 0] + c[:, 2] * 0.5
    c[:, 1] = c[:, 1] + c[:, 3] * 0.5
    return _f32c(c[:, :2])


def center_distance_matrix(trk_tlwh: np.ndarray, det_tlwh: np.ndarray,
                           norm: str, img_w: int, img_h: int) -> np.ndarray:
    trk_tlwh = np.asarray(trk_tlwh, dtype=np.float32)
    det_tlwh = np.asarray(det_tlwh, dtype=np.float32)
    T, D = trk_tlwh.shape[0], det_tlwh.shape[0]
    if T == 0 or D == 0:
        return np.zeros((T, D), dtype=np.float32)
    tC = compute_centers(trk_tlwh)
    dC = compute_centers(det_tlwh)
    diffs = tC[:, None, :] - dC[None, :, :]
    dist = np.sqrt((diffs ** 2).sum(-1)).astype(np.float32)
    if norm == "pairwise_box":
        t_hw = trk_tlwh[:, 2:4]
        d_hw = det_tlwh[:, 2:4]
        t_s = np.maximum(t_hw[:, 0], t_hw[:, 1])[:, None]
        d_s = np.maximum(d_hw[:, 0], d_hw[:, 1])[None, :]
        scale = np.maximum(np.maximum(t_s, d_s), 1e-6)
        return _f32c(dist / scale)
    diag = float(np.hypot(img_w, img_h))
    return _f32c(dist / max(diag, 1e-6))


def iou_xyxy_matrix(trk_xyxy: np.ndarray, det_xyxy: np.ndarray) -> np.ndarray:
    trk_xyxy = np.asarray(trk_xyxy, dtype=np.float32)
    det_xyxy = np.asarray(det_xyxy, dtype=np.float32)
    if trk_xyxy.size == 0 or det_xyxy.size == 0:
        return np.zeros((trk_xyxy.shape[0], det_xyxy.shape[0]), dtype=np.float32)
    t = trk_xyxy[:, None, :]
    d = det_xyxy[None, :, :]
    inter_x1 = np.maximum(t[..., 0], d[..., 0])
    inter_y1 = np.maximum(t[..., 1], d[..., 1])
    inter_x2 = np.minimum(t[..., 2], d[..., 2])
    inter_y2 = np.minimum(t[..., 3], d[..., 3])
    inter_w = np.maximum(0., inter_x2 - inter_x1)
    inter_h = np.maximum(0., inter_y2 - inter_y1)
    inter   = inter_w * inter_h
    area_t = (t[..., 2] - t[..., 0]) * (t[..., 3] - t[..., 1])
    area_d = (d[..., 2] - d[..., 0]) * (d[..., 3] - d[..., 1])
    union  = np.maximum(area_t + area_d - inter, 1e-6)
    return _f32c((inter / union).astype(np.float32))


# =========================
# Deleted detection recovery
# =========================
def _bbox_overlaps_xyxy(a_xyxy: np.ndarray, b_xyxy: np.ndarray) -> np.ndarray:
    a = np.asarray(a_xyxy, dtype=np.float32).reshape(-1, 4)
    b = np.asarray(b_xyxy, dtype=np.float32).reshape(-1, 4)
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)

    A = a[:, None, :]
    B = b[None, :, :]

    inter_x1 = np.maximum(A[..., 0], B[..., 0])
    inter_y1 = np.maximum(A[..., 1], B[..., 1])
    inter_x2 = np.minimum(A[..., 2], B[..., 2])
    inter_y2 = np.minimum(A[..., 3], B[..., 3])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = (A[..., 2] - A[..., 0]) * (A[..., 3] - A[..., 1])
    area_b = (B[..., 2] - B[..., 0]) * (B[..., 3] - B[..., 1])
    union = np.maximum(area_a + area_b - inter, 1e-6)

    return _f32c(inter / union)


def find_deleted_detections(dets_tlbr: np.ndarray, dets95_tlbr: np.ndarray, thr=0.97):
    """
    dets_tlbr: Nx5
    dets95_tlbr: Mx5
    Return subset of dets95 considered suppressed by NMS relative to dets.
    """
    dets_tlbr = np.asarray(dets_tlbr, dtype=np.float32).reshape(-1, 5)
    dets95_tlbr = np.asarray(dets95_tlbr, dtype=np.float32).reshape(-1, 5)
    if dets_tlbr.size == 0 or dets95_tlbr.size == 0:
        return np.zeros((0, 5), dtype=np.float32)

    ious = _bbox_overlaps_xyxy(dets_tlbr[:, :4], dets95_tlbr[:, :4])
    max_iou = ious.max(axis=0) if ious.size else np.zeros((dets95_tlbr.shape[0],), np.float32)
    del_mask = max_iou < float(thr)
    return _f32c(dets95_tlbr[del_mask])


# =========================
# Formation helpers (unchanged core)
# =========================
def _knn_idx(points, k):
    points = np.asarray(points, dtype=np.float32)
    N = points.shape[0]
    if N == 0:
        return []
    d2 = ((points[:, None, :] - points[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    k_eff = min(k, max(N - 1, 1))
    idx = np.argpartition(d2, kth=k_eff - 1, axis=1)[:, :k_eff]
    out = []
    for i in range(N):
        neigh = idx[i]
        order = np.argsort(d2[i, neigh])
        out.append(neigh[order].tolist())
    return out


def _procrustes_se2(P, Q):
    if P.shape[0] == 0:
        return 1.0, np.eye(2, dtype=np.float32)
    H = P.T @ Q
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    denom = (P ** 2).sum()
    s = float(S.sum() / max(denom, 1e-12))
    return s, R.astype(np.float32)


def formation_cost_sparse(trk_tlwh, det_tlwh, img_w, img_h, Nlist,
                          iou_matrix, cd_matrix, candidate_mask,
                          k_neighbors=2, proxy_gate_frac=0.05, huber_delta=0.02,
                          topk=6):
    trk_tlwh = np.asarray(trk_tlwh, dtype=np.float32)
    det_tlwh = np.asarray(det_tlwh, dtype=np.float32)
    iou_matrix = np.asarray(iou_matrix, dtype=np.float32)
    cd_matrix = np.asarray(cd_matrix, dtype=np.float32)
    candidate_mask = candidate_mask.astype(bool) if candidate_mask.size else candidate_mask

    T, D = trk_tlwh.shape[0], det_tlwh.shape[0]
    form = np.zeros((T, D), dtype=np.float32)
    if T == 0 or D == 0:
        return form

    diag = float(np.hypot(img_w, img_h))
    tC = compute_centers(trk_tlwh)
    dC = compute_centers(det_tlwh)

    cd_all = np.sqrt(((tC[:, None, :] - dC[None, :, :]) ** 2).sum(-1))
    nearest_det = np.argmin(cd_all, axis=1) if D else np.zeros((T,), int)
    nearest_dist = cd_all[np.arange(T), nearest_det] if D else np.full((T,), np.inf, np.float32)

    proxyC = tC.copy()
    gate = proxy_gate_frac * diag
    use_det = nearest_dist <= gate
    if D:
        proxyC[use_det] = dC[nearest_det[use_det]]

    for i in range(T):
        if not candidate_mask[i].any():
            continue
        neigh = Nlist[i]
        if len(neigh) < 2:
            continue

        P = (tC[i][None, :] - tC[neigh])
        Pz = P - P.mean(0, keepdims=True)
        denom = (Pz ** 2).sum()
        if denom <= 1e-12:
            continue

        neigh_proxy = proxyC[neigh]

        cand = np.where(candidate_mask[i])[0]
        if cand.size == 0:
            continue
        K = min(topk, cand.size)

        # prefer candidates with smaller center distance
        sel = cand[np.argpartition(cd_matrix[i, cand], K - 1)[:K]]

        for j in sel:
            Q = (dC[j][None, :] - neigh_proxy)
            Qz = Q - Q.mean(0, keepdims=True)
            Hm = Pz.T @ Qz
            U, S, Vt = np.linalg.svd(Hm)
            R = Vt.T @ U.T
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T
            s = float(S.sum() / denom)
            resid = np.linalg.norm(Qz - s * (Pz @ R.T), axis=1)
            med = float(np.median(resid) / max(diag, 1e-6))
            d = huber_delta
            form[i, j] = med if med <= d else d * (med - 0.5 * d)

    return _f32c(form)


# =========================
# Matching helpers
# =========================
def hungarian_min(cost: np.ndarray) -> np.ndarray:
    if cost is None:
        return np.empty((0, 2), dtype=int)
    cost = _f32c(_nan2big(cost))
    if cost.size == 0 or cost.shape[0] == 0 or cost.shape[1] == 0:
        return np.empty((0, 2), dtype=int)
    try:
        return matching.linear_assignment2(cost)
    except AttributeError:
        matches, _, _ = matching.linear_assignment(cost, thresh=1e9)
        return np.asarray(matches, dtype=int) if len(matches) else np.empty((0, 2), dtype=int)
    except Exception:
        try:
            from scipy.optimize import linear_sum_assignment
            r, c = linear_sum_assignment(cost)
            return np.stack([r, c], axis=1).astype(int)
        except Exception:
            return np.empty((0, 2), dtype=int)


def _cosine_cost(strack_pool, detections):
    T, D = len(strack_pool), len(detections)
    if T == 0 or D == 0:
        return np.zeros((T, D), dtype=np.float32)
    idx_t = [i for i, st in enumerate(strack_pool) if getattr(st, "emb", None) is not None]
    idx_d = [j for j, dt in enumerate(detections) if getattr(dt, "emb", None) is not None]
    if not idx_t or not idx_d:
        return np.zeros((T, D), dtype=np.float32)
    U = np.asarray([strack_pool[i].emb for i in idx_t], dtype=np.float32)
    V = np.asarray([detections[j].emb for j in idx_d], dtype=np.float32)
    U /= (np.linalg.norm(U, axis=1, keepdims=True) + 1e-12)
    V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    dist = 1.0 - (U @ V.T)
    C = np.zeros((T, D), dtype=np.float32)
    C[np.ix_(idx_t, idx_d)] = dist
    return _f32c(C)


def confidence_consistency_cost(stracks, det_scores):
    """
    TrackTrack style confidence trend projection, but minimal and safe.
    Returns TxD matrix in [0,1] approximately.
    """
    T = len(stracks)
    D = len(det_scores)
    if T == 0 or D == 0:
        return np.zeros((T, D), dtype=np.float32)

    det_scores = np.asarray(det_scores, dtype=np.float32).reshape(-1)
    # predict track score using last two observations if available
    pred = []
    for st in stracks:
        mem = getattr(st, "score_memory", None)
        if mem is not None and len(mem) >= 2:
            s_now = float(mem[-1])
            s_prev = float(mem[-2])
            s_pred = s_now + (s_now - s_prev)
        else:
            s_pred = float(getattr(st, "score", 0.5))
        pred.append(np.clip(s_pred, 0.0, 1.0))
    pred = np.asarray(pred, dtype=np.float32)
    # absolute difference
    C = np.abs(pred[:, None] - det_scores[None, :])
    return _f32c(np.clip(C, 0.0, 1.0))


# =========================
# Subset cache tag helper
# =========================
def _subset_cache_tag(tag: str, frame_id: int, boxes: np.ndarray) -> str:
    b = boxes.reshape(-1, 4).astype(np.float32)
    q = np.round(b, 1).tobytes()
    h = hashlib.sha1(q).hexdigest()[:8]
    return f"{tag}__f{frame_id}__n{b.shape[0]}__{h}"


# =========================
# KDF track
# =========================
class STrack(BaseTrack):
    def __init__(self, tlwh, score, temp_feat=None, buffer_size=30):
        self.xywh_omemory = deque([], maxlen=buffer_size)
        self.xywh_pmemory = deque([], maxlen=buffer_size)
        self.xywh_amemory = deque([], maxlen=buffer_size)
        self.conds = deque([], maxlen=5)

        # score memory for medium confidence consistency
        self.score_memory = deque([], maxlen=5)

        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = float(score)
        self.tracklet_len = 0

        self.emb = temp_feat
        self.features = deque([], maxlen=buffer_size)

        self.unobserved = False
        self.miss_streak = 0

        # persistent center-KF
        self.kf_center = None
        self.kf_inited = False

    def update_features(self, feat, alpha=0.95):
        if feat is None:
            return
        if self.emb is None:
            self.emb = feat
        else:
            self.emb = alpha * self.emb + (1 - alpha) * feat
        n = np.linalg.norm(self.emb) + 1e-12
        self.emb = self.emb / n

    # persistent KF helpers
    def _kf_measure_xyah(self):
        cx, cy, w, h = map(float, self.xywh)
        a = w / max(h, 1e-12)
        return np.array([cx, cy, a, h], dtype=np.float32)

    def kf_update_with_measurement(self):
        if self.kf_center is None:
            self.kf_center = KalmanFilter()
        z = self._kf_measure_xyah()
        if (not self.kf_inited) or (self.mean is None) or (self.covariance is None):
            self.mean, self.covariance = self.kf_center.initiate(z)
            self.kf_inited = True
        else:
            self.mean, self.covariance = self.kf_center.predict(self.mean.copy(), self.covariance.copy())
            self.mean, self.covariance = self.kf_center.update(self.mean, self.covariance, z)

    def kf_predict_delta_xywh(self):
        if (not self.kf_inited) or self.kf_center is None or self.mean is None or self.covariance is None:
            return np.zeros(4, dtype=np.float32)
        mean1, cov1 = self.kf_center.predict(self.mean.copy(), self.covariance.copy())
        z_pred, _ = self.kf_center.project(mean1, cov1)
        cx_p, cy_p, a_p, h_p = map(float, z_pred[:4])
        w_p = a_p * h_p
        cx, cy, w, h = map(float, self.xywh)
        return np.array([cx_p - cx, cy_p - cy, w_p - w, h_p - h], dtype=np.float32)

    @staticmethod
    def multi_predict_diff(stracks, model, img_w, img_h,
                           lambda_kf=0.91, residual_is_normalized=False):
        """
        KDF: delta = lambda_kf * KF_delta + Diff_residual
        Strict clamps to prevent explosions.
        """
        N = len(stracks)
        if N == 0:
            return

        # unwrap DP safely
        if hasattr(model, "module"):
            model_core = model.module
        else:
            model_core = model

        W = float(max(img_w, 1e-6))
        H = float(max(img_h, 1e-6))

        cur_xywh = np.asarray([st.xywh for st in stracks], dtype=np.float32)

        prev_xywh = np.asarray([(st.xywh_amemory[-1] if len(st.xywh_amemory) >= 1 else st.xywh)
                                for st in stracks], dtype=np.float32)
        prev2_xywh = np.asarray([(st.xywh_amemory[-2] if len(st.xywh_amemory) >= 2 else prev_xywh[i])
                                 for i, st in enumerate(stracks)], dtype=np.float32)
        prev_speed = np.linalg.norm(prev_xywh[:, :2] - prev2_xywh[:, :2], axis=1)

        kf_motion = np.vstack([st.kf_predict_delta_xywh() for st in stracks]).astype(np.float32)

        # Diff residual
        residual = np.zeros_like(cur_xywh, dtype=np.float32)
        try:
            conds_list = []
            for st in stracks:
                if len(st.conds):
                    conds_list.append(np.asarray(st.conds, dtype=np.float32))
                else:
                    conds_list.append(np.zeros((1, 8), np.float32))
            gen = model_core.generate(conds_list, sample=1, bestof=True, img_w=img_w, img_h=img_h)
            gen = np.asarray(gen, dtype=np.float32)
            if gen.ndim == 3:
                residual = gen.mean(0)
            elif gen.ndim == 2:
                residual = gen
            if residual.shape != cur_xywh.shape:
                residual = np.zeros_like(cur_xywh, dtype=np.float32)
        except Exception:
            residual = np.zeros_like(cur_xywh, dtype=np.float32)

        residual = _f32c(residual)
        if residual_is_normalized:
            residual *= np.array([W, H, W, H], dtype=np.float32)

        delta = (lambda_kf * kf_motion) + residual
        fused_xywh = cur_xywh + delta

        # clamp center motion
        max_move = np.maximum(3.0, 3.5 * prev_speed)[:, None]
        dc = fused_xywh[:, :2] - cur_xywh[:, :2]
        dc = np.clip(dc, -max_move, max_move)
        fused_xywh[:, :2] = cur_xywh[:, :2] + dc

        # clamp size change per frame
        rel = 0.45
        fused_xywh[:, 2] = np.clip(fused_xywh[:, 2], (1.0 - rel) * cur_xywh[:, 2], (1.0 + rel) * cur_xywh[:, 2])
        fused_xywh[:, 3] = np.clip(fused_xywh[:, 3], (1.0 - rel) * cur_xywh[:, 3], (1.0 + rel) * cur_xywh[:, 3])

        # bounds
        fused_xywh[:, 2] = np.clip(fused_xywh[:, 2], 1e-3, W)
        fused_xywh[:, 3] = np.clip(fused_xywh[:, 3], 1e-3, H)
        fused_xywh[:, 0] = np.clip(fused_xywh[:, 0], 0.0, W)
        fused_xywh[:, 1] = np.clip(fused_xywh[:, 1], 0.0, H)

        tlwh = np.empty_like(fused_xywh, dtype=np.float32)
        tlwh[:, 0] = fused_xywh[:, 0] - 0.5 * fused_xywh[:, 2]
        tlwh[:, 1] = fused_xywh[:, 1] - 0.5 * fused_xywh[:, 3]
        tlwh[:, 2] = fused_xywh[:, 2]
        tlwh[:, 3] = fused_xywh[:, 3]

        for st, box in zip(stracks, tlwh):
            st._tlwh = box
            st.xywh_pmemory.append(st.xywh.copy())
            st.xywh_amemory.append(st.xywh.copy())
            if len(st.xywh_amemory) >= 2:
                tmp_delta = st.xywh.copy() - st.xywh_amemory[-2].copy()
            else:
                tmp_delta = np.zeros(4, dtype=np.float32)
            st.conds.append(np.concatenate((st.xywh.copy(), tmp_delta)))

    def activate(self, frame_id):
        self.track_id = self.next_id()
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

        self.xywh_omemory.append(self.xywh.copy())
        self.xywh_pmemory.append(self.xywh.copy())
        self.xywh_amemory.append(self.xywh.copy())
        self.score_memory.append(float(self.score))

        delta_bbox = np.zeros_like(self.xywh.copy())
        self.conds.append(np.concatenate((self.xywh.copy(), delta_bbox)))

        self.unobserved = False
        self.miss_streak = 0
        self.kf_update_with_measurement()

    def re_activate(self, new_track, frame_id, new_id=False):
        self._tlwh = new_track.tlwh
        self.score = float(new_track.score)
        self.score_memory.append(float(self.score))

        self.xywh_omemory.append(self.xywh.copy())
        self.xywh_amemory.append(self.xywh.copy())
        if len(self.xywh_amemory) >= 2:
            tmp_delta_bbox = self.xywh.copy() - self.xywh_amemory[-2].copy()
        else:
            tmp_delta_bbox = np.zeros_like(self.xywh.copy())
        self.conds.append(np.concatenate((self.xywh.copy(), tmp_delta_bbox)))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id

        self.unobserved = False
        self.miss_streak = 0
        if new_id:
            self.track_id = self.next_id()
        self.kf_update_with_measurement()

    def update(self, new_track, frame_id, update_feature=False):
        self.frame_id = frame_id
        self.tracklet_len += 1
        self._tlwh = new_track.tlwh
        self.score = float(new_track.score)
        self.score_memory.append(float(self.score))

        self.xywh_omemory.append(self.xywh.copy())
        self.xywh_amemory.append(self.xywh.copy())
        if len(self.xywh_amemory) >= 2:
            tmp_delta_bbox = self.xywh.copy() - self.xywh_amemory[-2].copy()
        else:
            tmp_delta_bbox = np.zeros_like(self.xywh.copy())
        self.conds.append(np.concatenate((self.xywh.copy(), tmp_delta_bbox)))

        self.state = TrackState.Tracked
        self.is_activated = True

        if update_feature and getattr(new_track, "emb", None) is not None:
            self.update_features(new_track.emb)

        self.kf_update_with_measurement()

    @property
    def tlwh(self):
        return self._tlwh.copy()

    @property
    def tlbr(self):
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def xywh(self):
        ret = self.tlwh.copy()
        ret[:2] = ret[:2] + ret[2:] / 2
        return ret

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr, dtype=np.float32).copy()
        ret[2:] -= ret[:2]
        return ret


# =========================
# Tracker
# =========================
class diffmottracker(object):
    def __init__(self, config, frame_rate=30):
        self.config = config
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []

        self.frame_id = 0
        self.det_thresh = float(self.config.high_thres)

        self.buffer_size = int(frame_rate / 30.0 * 30)
        self.max_time_lost = self.buffer_size

        self.embedder = EmbeddingComputer(self.config, 'mot17', False, True)
        self.alpha_fixed_emb = 0.95

        # Association knobs
        self.center_norm = getattr(self.config, "center_norm", "pairwise_box")
        self.max_center_frac = float(getattr(self.config, "max_center_frac", 0.06))
        self.min_iou_gate   = float(getattr(self.config, "min_iou_gate", 0.10))
        self.w_assoc_emb    = float(getattr(self.config, "w_assoc_emb", 1.0))

        # Medium-stage confidence weight
        self.w_medium_conf = float(getattr(self.config, "w_medium_conf", 0.15))
        self.medium_iou_thr = float(getattr(self.config, "medium_iou_thr", 0.70))

        # Formation
        self.fila_k           = int(getattr(self.config, "fila_k", 2))
        self.fila_proxy_frac  = float(getattr(self.config, "fila_proxy_frac", 0.05))
        self.fila_huber_delta = float(getattr(self.config, "fila_huber_delta", 0.02))

        # Ultra-low support
        self.support_zone_frac = float(getattr(self.config, "support_zone_frac", 0.06))
        self.ultra_low_thres   = float(getattr(self.config, "ultra_low_thres",
                                               max(0.02, self.config.low_thres - 0.05)))
        self.support_iou_gate  = float(getattr(self.config, "support_iou_gate", 0.10))
        self.support_bonus     = float(getattr(self.config, "support_bonus", 0.10))

        # Deleted detection recovery
        self.del_iou_dup_thr = float(getattr(self.config, "del_iou_dup_thr", 0.97))
        self.del_match_thr   = float(getattr(self.config, "del_match_thr", 0.75))

        # ReID
        self.reid_enabled     = bool(getattr(self.config, "reid_enabled", True))
        self.reid_cos_thr     = float(getattr(self.config, "reid_cos_thr", 0.35))
        self.reid_time_window = int(getattr(self.config, "reid_time_window", self.buffer_size // 2))

        # KDF fusion flags
        self.residual_is_normalized = bool(getattr(self.config, "residual_is_normalized", False))
        self.lambda_kf = float(getattr(self.config, "lambda_kf", 0.91))

        # emission options
        self.emit_unobserved_max = int(getattr(self.config, "emit_unobserved_max", 0))

    def dump_cache(self):
        self.embedder.dump_cache()

    def _safe_compute_emb(self, img, boxes, tag):
        boxes = _f32c(boxes)
        try:
            embs = self.embedder.compute_embedding(img, boxes, tag)
            return embs
        except RuntimeError as e:
            if "cached embeddings don't match" in str(e):
                logger.warning("ReID cache mismatch for tag %s; flushing and recomputing.", tag)
                try:
                    self.embedder.dump_cache()
                except Exception:
                    pass
                return self.embedder.compute_embedding(img, boxes, tag)
            raise

    def update(self, dets_norm, model, frame_id, img_w, img_h, tag,
               img=None, dets_95_norm=None):
        """
        dets_norm: Nx5 [x, y, w, h, score] absolute tlxywh
        dets_95_norm: optional backup set with weaker NMS or higher pre-NMS pool
        """
        self.model = model
        self.frame_id += 1

        activated_starcks, refind_stracks, lost_stracks, removed_stracks = [], [], [], []

        # ---- sanitize detections ----
        dets_norm = np.asarray(dets_norm, dtype=np.float32).reshape(-1, 5)
        if dets_norm.size:
            good = np.isfinite(dets_norm).all(axis=1)
            whp = (dets_norm[:, 2] > 0) & (dets_norm[:, 3] > 0)
            dets_norm = dets_norm[good & whp]
        dets_norm = _f32c(dets_norm)

        dets_all = _xywh_to_tlbr(dets_norm)

        # backup dets95
        if dets_95_norm is not None:
            dets_95_norm = np.asarray(dets_95_norm, dtype=np.float32).reshape(-1, 5)
            if dets_95_norm.size:
                good = np.isfinite(dets_95_norm).all(axis=1)
                whp = (dets_95_norm[:, 2] > 0) & (dets_95_norm[:, 3] > 0)
                dets_95_norm = dets_95_norm[good & whp]
            dets_95_norm = _f32c(dets_95_norm)
            dets95_all = _xywh_to_tlbr(dets_95_norm)
        else:
            dets95_all = np.zeros((0, 5), dtype=np.float32)

        # split by score
        remain_inds = dets_all[:, 4] > self.det_thresh if dets_all.size else np.array([], dtype=bool)
        inds_low    = dets_all[:, 4] > self.config.low_thres if dets_all.size else np.array([], dtype=bool)
        inds_high   = dets_all[:, 4] < self.det_thresh if dets_all.size else np.array([], dtype=bool)

        inds_medium = np.logical_and(inds_low, inds_high) if dets_all.size else np.array([], dtype=bool)
        inds_ultra  = np.logical_and(dets_all[:, 4] >= self.ultra_low_thres,
                                     dets_all[:, 4] <= self.config.low_thres) if dets_all.size else np.array([], dtype=bool)

        dets_high = dets_all[remain_inds] if dets_all.size else np.zeros((0, 5), np.float32)
        dets_med  = dets_all[inds_medium] if dets_all.size else np.zeros((0, 5), np.float32)

        # deleted detection pool
        dets_del = find_deleted_detections(dets_all, dets95_all, thr=self.del_iou_dup_thr) \
                   if (dets_all.size and dets95_all.size) else np.zeros((0, 5), np.float32)
        # keep only useful deleted boxes near medium-high
        dets_del_high = dets_del[dets_del[:, 4] >= self.config.low_thres] if dets_del.size else np.zeros((0, 5), np.float32)

        # embeddings if needed
        need_emb = (self.w_assoc_emb > 0.0) or (self.reid_enabled and len(self.lost_stracks) > 0)

        dets_high_embs = [None] * len(dets_high)
        if need_emb and len(dets_high) and (img is not None):
            dets_high_embs = [e for e in self._safe_compute_emb(img, dets_high[:, :4], tag)]

        trust = ((dets_high[:, 4] - self.det_thresh) / max((1 - self.det_thresh), 1e-6)) \
                if len(dets_high) else np.zeros((0,), np.float32)
        dets_alpha = self.alpha_fixed_emb + (1 - self.alpha_fixed_emb) * (1 - trust) \
                     if len(dets_high) else np.zeros((0,), np.float32)

        detections_high = [STrack(STrack.tlbr_to_tlwh(t[:4]), float(t[4]), f, 30)
                           for (t, f) in zip(dets_high[:, :5], dets_high_embs)] if len(dets_high) else []

        # prepare medium detections object list (no emb by default)
        detections_med = [STrack(STrack.tlbr_to_tlwh(t[:4]), float(t[4]), None, 30)
                          for t in dets_med[:, :5]] if len(dets_med) else []

        # deleted recovery detections list
        detections_del = [STrack(STrack.tlbr_to_tlwh(t[:4]), float(t[4]), None, 30)
                          for t in dets_del_high[:, :5]] if len(dets_del_high) else []

        # split confirmed/unconfirmed
        unconfirmed, tracked_stracks = [], []
        for tr in self.tracked_stracks:
            (unconfirmed if not tr.is_activated else tracked_stracks).append(tr)

        # KDF prediction on pool
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        STrack.multi_predict_diff(
            strack_pool, self.model, img_w, img_h,
            lambda_kf=self.lambda_kf,
            residual_is_normalized=self.residual_is_normalized
        )

        # ---------- Stage A: high association with IoU + FPC + optional emb ----------
        trk_tlwh = _f32c(np.array([st.tlwh for st in strack_pool], dtype=np.float32)) \
                   if len(strack_pool) else np.zeros((0, 4), np.float32)
        det_tlwh = _f32c(np.array([dt.tlwh for dt in detections_high], dtype=np.float32)) \
                   if len(detections_high) else np.zeros((0, 4), np.float32)

        trk_xyxy = _tlwh_to_xyxy(trk_tlwh)
        det_xyxy = _tlwh_to_xyxy(det_tlwh)

        iou_matrix = iou_xyxy_matrix(trk_xyxy, det_xyxy)
        cd_matrix  = center_distance_matrix(trk_tlwh, det_tlwh, self.center_norm, img_w, img_h)
        trkC_all = compute_centers(trk_tlwh)
        Nlist = _knn_idx(trkC_all, self.fila_k)

        T, D = iou_matrix.shape

        DYN_IOU_MARGIN = 0.08
        TOPK_PER_ROW   = 6
        CENTER_GATE    = max(self.max_center_frac * 1.5, 0.09)

        best_iou = iou_matrix.max(axis=1) if (T and D) else np.zeros((T,), np.float32)
        dyn_thr  = np.maximum(self.min_iou_gate, best_iou - DYN_IOU_MARGIN)

        cand_mask = (iou_matrix >= dyn_thr[:, None]) | (cd_matrix <= CENTER_GATE) \
                    if (T and D) else np.zeros((T, D), bool)

        if T and D:
            for i in range(T):
                if D <= TOPK_PER_ROW:
                    cand_mask[i, :] = True
                else:
                    kk = np.argpartition(iou_matrix[i], -(TOPK_PER_ROW))[-TOPK_PER_ROW:]
                    cand_mask[i, kk] = True

        iou_cost_full = 1.0 - iou_matrix
        base_cost = np.full_like(iou_cost_full, BIG, dtype=np.float32)
        base_cost[cand_mask] = iou_cost_full[cand_mask]

        form_matrix = formation_cost_sparse(
            trk_tlwh, det_tlwh, img_w, img_h, Nlist,
            iou_matrix, cd_matrix, cand_mask,
            k_neighbors=self.fila_k,
            proxy_gate_frac=self.fila_proxy_frac,
            huber_delta=self.fila_huber_delta,
            topk=TOPK_PER_ROW,
        )

        emb_cost = _cosine_cost(strack_pool, detections_high) if need_emb else np.zeros_like(base_cost, np.float32)
        emb_cost[~cand_mask] = 0.0

        fused_cost = base_cost + 1.25 * form_matrix + (self.w_assoc_emb * emb_cost)
        fused_cost = _f32c(_nan2big(fused_cost))

        # ensure each row has some finite fallback
        if T and D:
            row_has = np.isfinite(fused_cost).any(axis=1)
            for i in range(T):
                if not row_has[i]:
                    fused_cost[i, :] = iou_cost_full[i, :]

        matches_A = hungarian_min(fused_cost) if (T and D) else np.empty((0, 2), int)

        picked_trk = set(matches_A[:, 0].tolist()) if len(matches_A) else set()
        picked_det = set(matches_A[:, 1].tolist()) if len(matches_A) else set()

        # ---------- Stage B: center rescue on remaining high dets ----------
        remain_trk_idx = [i for i in range(len(strack_pool)) if i not in picked_trk]
        remain_det_idx = [j for j in range(len(detections_high)) if j not in picked_det]

        if len(remain_trk_idx) and len(remain_det_idx):
            cd_sub = cd_matrix[np.ix_(remain_trk_idx, remain_det_idx)]
            RESCUE_CENTER_FRAC = max(self.max_center_frac, 0.06)
            cd_gate_sub = (cd_sub <= RESCUE_CENTER_FRAC)
            cd_cost_sub = np.where(cd_gate_sub, cd_sub, BIG)
            cd_cost_sub = _f32c(_nan2big(cd_cost_sub))
            matches_B_sub = hungarian_min(cd_cost_sub) if cd_cost_sub.size else np.empty((0, 2), dtype=int)
            matches_B = np.array([[remain_trk_idx[i], remain_det_idx[j]] for i, j in matches_B_sub], dtype=int)
        else:
            matches_B = np.empty((0, 2), dtype=int)

        matched_indices = np.vstack([matches_A, matches_B]) \
                          if (len(matches_A) or len(matches_B)) else np.empty((0, 2), dtype=int)

        # reject very low IoU pairs
        if iou_matrix.size and matched_indices.size:
            keep = []
            for m in matched_indices:
                if iou_matrix[m[0], m[1]] < 0.10:
                    continue
                keep.append(m.reshape(1, 2))
            matched_indices = np.concatenate(keep, axis=0) if len(keep) else np.empty((0, 2), dtype=int)

        # apply matches A+B
        for itracked, idet in matched_indices:
            tr = strack_pool[itracked]
            det = detections_high[idet]
            alp = float(dets_alpha[idet]) if idet < len(dets_alpha) else 0.95

            if tr.state == TrackState.Tracked:
                tr.update(det, self.frame_id)
                tr.unobserved = False
                tr.miss_streak = 0
                if getattr(det, "emb", None) is not None:
                    tr.update_features(det.emb, alp)
                activated_starcks.append(tr)
            else:
                tr.re_activate(det, self.frame_id, new_id=False)
                tr.unobserved = False
                tr.miss_streak = 0
                if getattr(det, "emb", None) is not None:
                    tr.update_features(det.emb, alp)
                refind_stracks.append(tr)

        unmatched_detections_high = [d for d in range(len(detections_high))
                                     if d not in matched_indices[:, 1]] if matched_indices.size \
                                    else list(range(len(detections_high)))
        unmatched_trackers = [t for t in range(len(strack_pool))
                              if t not in matched_indices[:, 0]] if matched_indices.size \
                             else list(range(len(strack_pool)))

        # ---------- Stage Del: deleted detection recovery ----------
        # Use ONLY as extra support source with IoU-dominant matching
        if len(detections_del) and len(unmatched_trackers):
            cand_trk_idx = [i for i in unmatched_trackers if strack_pool[i].state == TrackState.Tracked]
            cand_trks = [strack_pool[i] for i in cand_trk_idx]

            if len(cand_trks):
                dists_del = matching.iou_distance(cand_trks, detections_del)
                matches_del, u_trk_del, u_det_del = matching.linear_assignment(dists_del, thresh=self.del_match_thr)
            else:
                matches_del = np.empty((0, 2), dtype=int)
                u_trk_del = list(range(len(cand_trks)))
                u_det_del = list(range(len(detections_del)))

            rescued = set()
            for li, lj in matches_del:
                itrk = cand_trk_idx[li]
                tr = strack_pool[itrk]
                det = detections_del[lj]
                tr.update(det, self.frame_id, update_feature=False)
                tr.unobserved = False
                tr.miss_streak = 0
                activated_starcks.append(tr)
                rescued.add(itrk)

            if rescued:
                unmatched_trackers = [i for i in unmatched_trackers if i not in rescued]

        # ---------- Stage M: medium association with confidence consistency only ----------
        # This stage is strictly IoU + confidence consistency, no angle, no extra cues
        if len(detections_med) and len(unmatched_trackers):
            cand_trk_idx = [i for i in unmatched_trackers if strack_pool[i].state == TrackState.Tracked]
            cand_trks = [strack_pool[i] for i in cand_trk_idx]

            if len(cand_trks):
                # IoU cost
                iou_cost_m = matching.iou_distance(cand_trks, detections_med)  # cost = 1 - iou
                # Confidence consistency cost
                det_scores_m = np.array([d.score for d in detections_med], dtype=np.float32)
                conf_cost_m = confidence_consistency_cost(cand_trks, det_scores_m)

                # fuse
                w_conf = np.clip(self.w_medium_conf, 0.0, 0.5)
                cost_m = (1.0 - w_conf) * iou_cost_m + w_conf * conf_cost_m
                cost_m = _f32c(_nan2big(cost_m))

                matches_m, u_trk_m, u_det_m = matching.linear_assignment(cost_m, thresh=self.medium_iou_thr)
            else:
                matches_m = np.empty((0, 2), dtype=int)
                u_trk_m = list(range(len(cand_trks)))
                u_det_m = list(range(len(detections_med)))

            rescued_m = set()
            for li, lj in matches_m:
                itrk = cand_trk_idx[li]
                tr = strack_pool[itrk]
                det = detections_med[lj]
                tr.update(det, self.frame_id, update_feature=False)
                tr.unobserved = False
                tr.miss_streak = 0
                activated_starcks.append(tr)
                rescued_m.add(itrk)

            if rescued_m:
                unmatched_trackers = [i for i in unmatched_trackers if i not in rescued_m]

        # ---------- Stage U: ultra-low support ----------
        if (dets_all.size and np.any(inds_ultra)) and len(unmatched_trackers):
            dets_ultra = dets_all[inds_ultra]
            det_ultra_objs = [STrack(STrack.tlbr_to_tlwh(t[:4]),
                                     float(min(1.0, t[4] + self.support_bonus)),
                                     None, 30)
                              for t in dets_ultra[:, :5]]

            cand_trk_idx = [i for i in unmatched_trackers if strack_pool[i].state == TrackState.Tracked]
            cand_trks = [strack_pool[i] for i in cand_trk_idx]

            if len(cand_trks) and len(det_ultra_objs):
                dists_u = matching.iou_distance(cand_trks, det_ultra_objs)
                matches_u, u_trk_u, u_det_u = matching.linear_assignment(dists_u, thresh=0.75)
            else:
                matches_u = np.empty((0, 2), dtype=int)

            rescued_u = set()
            for li, lj in matches_u:
                itrk = cand_trk_idx[li]
                tr = strack_pool[itrk]
                det = det_ultra_objs[lj]
                tr.update(det, self.frame_id, update_feature=False)
                tr.unobserved = False
                tr.miss_streak = 0
                activated_starcks.append(tr)
                rescued_u.add(itrk)

            if rescued_u:
                unmatched_trackers = [i for i in unmatched_trackers if i not in rescued_u]

        # ---------- Low-score fallback (classic ByteTrack style) ----------
        # Use dets_med as already consumed; keep a secondary low pool if you still want
        # Here we reuse "medium objects were already used"; so we do a final low-only IOU stage
        # with dets in (low_thres, high_thres) that remain unmatched is complicated.
        # For stability, we skip extra low-stage duplication here.

        # ---------- Unconfirmed vs remaining high dets ----------
        detections_u = [detections_high[i] for i in unmatched_detections_high]
        if len(unconfirmed) and len(detections_u):
            dists_u = matching.iou_distance(unconfirmed, detections_u)
            matches_u2, u_unconfirmed, u_det = matching.linear_assignment(dists_u, thresh=0.75)
        else:
            matches_u2 = np.empty((0, 2), dtype=int)
            u_unconfirmed, u_det = [], []

        for itracked, idet in matches_u2:
            unconfirmed[itracked].update(detections_u[idet], self.frame_id)
            unconfirmed[itracked].unobserved = False
            unconfirmed[itracked].miss_streak = 0
            activated_starcks.append(unconfirmed[itracked])

        for it in u_unconfirmed:
            tr = unconfirmed[it]
            tr.mark_removed()
            removed_stracks.append(tr)

        # ---------- Mark remaining unmatched as lost ----------
        for idx in unmatched_trackers:
            tr = strack_pool[idx]
            if tr.state != TrackState.Lost:
                tr.mark_lost()
                lost_stracks.append(tr)

        # ---------- ReID bridge ----------
        # unchanged logic, but only use remaining unmatched high detections
        unmatched_detections = unmatched_detections_high

        if self.reid_enabled and len(unmatched_detections) and len(self.lost_stracks):
            cand_dets = [detections_high[i] for i in unmatched_detections]
            pairs = []
            for lt_i, lt in enumerate(self.lost_stracks):
                if lt.emb is None:
                    continue
                if self.frame_id - lt.end_frame > self.reid_time_window:
                    continue
                for dj, det in enumerate(cand_dets):
                    if getattr(det, "emb", None) is None and img is not None and need_emb:
                        subtag = _subset_cache_tag(tag, self.frame_id, det.tlbr.reshape(1, 4))
                        emb = self._safe_compute_emb(img, det.tlbr.reshape(1, 4), subtag)
                        det.emb = emb[0]
                    if getattr(det, "emb", None) is None:
                        continue
                    n1 = np.linalg.norm(lt.emb) + 1e-12
                    n2 = np.linalg.norm(det.emb) + 1e-12
                    dist = float(1.0 - np.dot(lt.emb, det.emb) / (n1 * n2))
                    cd = center_distance_matrix(
                        np.array([lt.tlwh], np.float32),
                        np.array([det.tlwh], np.float32),
                        self.center_norm, img_w, img_h
                    )[0, 0]
                    if dist <= self.reid_cos_thr and cd <= max(self.max_center_frac * 2, 0.12):
                        pairs.append((lt_i, dj, dist + 0.2 * cd))

            if pairs:
                pairs.sort(key=lambda x: x[2])
                used_lost, used_det = set(), set()
                for lt_i, dj, _ in pairs:
                    if lt_i in used_lost or dj in used_det:
                        continue
                    lt = self.lost_stracks[lt_i]
                    det = cand_dets[dj]
                    lt.re_activate(det, self.frame_id, new_id=False)
                    lt.unobserved = False
                    lt.miss_streak = 0
                    activated_starcks.append(lt)
                    used_lost.add(lt_i)
                    used_det.add(dj)

                unmatched_detections = [idx for k, idx in enumerate(unmatched_detections) if k not in used_det]

        # ---------- Birth new tracks from remaining high dets ----------
        for inew in unmatched_detections:
            tr = detections_high[inew]
            if tr.score < self.det_thresh:
                continue
            tr.activate(self.frame_id)
            activated_starcks.append(tr)

        # ---------- Cleanup old lost ----------
        for tr in self.lost_stracks:
            if self.frame_id - tr.end_frame > self.max_time_lost:
                tr.mark_removed()
                removed_stracks.append(tr)

        # ---------- Merge lists ----------
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)

        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)

        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)

        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        # ---------- Emission filter ----------
        output_stracks = [
            tr for tr in self.tracked_stracks
            if tr.is_activated and (not tr.unobserved or tr.miss_streak <= self.emit_unobserved_max)
        ]
        return output_stracks


# =========================
# List utilities
# =========================
def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    if len(stracksa) == 0 or len(stracksb) == 0:
        return stracksa, stracksb
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if i not in dupa]
    resb = [t for i, t in enumerate(stracksb) if i not in dupb]
    return resa, resb
