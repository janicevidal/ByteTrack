import numpy as np
from collections import defaultdict
from collections import deque
import os
import os.path as osp
import copy
import torch
import torch.nn.functional as F

from .kalman_filter import KalmanFilter
from yolox.tracker import matching
from .basetrack import BaseTrack, TrackState

class MCSTrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score, cls_id):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.cls_id = cls_id
        self.tracklet_len = 0

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance =MCSTrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def reset_track_id(self):
        self.reset_track_count(self.cls_id)
        
    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id(self.cls_id)
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id(self.cls_id)
        self.score = new_track.score

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_({}-{})_({}-{})'.format(self.cls_id, self.track_id,
                                           self.start_frame, self.end_frame)


class MCBYTETracker(object):
    def __init__(self, args, frame_rate=30, num_classes=1):
        # self.tracked_stracks = []  # type: list[STrack]
        # self.lost_stracks = []  # type: list[STrack]
        # self.removed_stracks = []  # type: list[STrack]
        
        self.tracked_tracks_dict = defaultdict(list)  # dict(list[STrack])
        self.lost_tracks_dict = defaultdict(list)  # dict(list[STrack])
        self.removed_tracks_dict = defaultdict(list)  # dict(list[STrack])

        self.frame_id = 0
        self.args = args
        #self.det_thresh = args.track_thresh
        self.num_classes = num_classes
        self.det_thresh = args.track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

    """
        pred_dets (np.array): Detection results of the image, the shape is
            [N, 6], means 'cls_id, score, x0, y0, x1, y1'.
    """
    def update(self, pred_dets):
        self.frame_id += 1
        if self.frame_id == 1:
            MCSTrack.init_count(self.num_classes)

        activated_tracks_dict = defaultdict(list)
        refined_tracks_dict = defaultdict(list)
        lost_tracks_dict = defaultdict(list)
        removed_tracks_dict = defaultdict(list)
        output_tracks_dict = defaultdict(list)
        
        pred_dets_dict = defaultdict(list)
        
        # unify single and multi classes detection and embedding results
        for cls_id in range(self.num_classes):
            cls_idx = (pred_dets[:, 0:1] == cls_id).squeeze(-1)
            pred_dets_dict[cls_id] = pred_dets[cls_idx]
            
        for cls_id in range(self.num_classes):
            """ Step 1: Get detections by class"""
            pred_dets_cls = pred_dets_dict[cls_id]
            remain_inds = (pred_dets_cls[:, 1:2] > self.args.track_thresh).squeeze(-1)
            if remain_inds.sum() > 0:
                pred_dets_cls = pred_dets_cls[remain_inds]
                detections = [
                    MCSTrack(
                        MCSTrack.tlbr_to_tlwh(tlbrs[2:6]),
                        tlbrs[1],
                        cls_id) for tlbrs in pred_dets_cls
                ]
            else:
                detections = []
                
            ''' Add newly detected tracklets to tracked_stracks'''
            unconfirmed_dict = defaultdict(list)
            tracked_tracks_dict = defaultdict(list)
            for track in self.tracked_tracks_dict[cls_id]:
                if not track.is_activated:
                    # previous tracks which are not active in the current frame are added in unconfirmed list
                    unconfirmed_dict[cls_id].append(track)
                else:
                    # Active tracks are added to the local list 'tracked_stracks'
                    tracked_tracks_dict[cls_id].append(track)
            
            ''' Step 2: First association, with high score detection boxes'''
            # building tracking pool for the current frame
            track_pool_dict = defaultdict(list)
            track_pool_dict[cls_id] = joint_stracks(
                tracked_tracks_dict[cls_id], self.lost_tracks_dict[cls_id])

            # Predict the current location with KalmanFilter
            MCSTrack.multi_predict(track_pool_dict[cls_id])

            dists = matching.iou_distance(track_pool_dict[cls_id], detections)
            if not self.args.mot20:
                dists = matching.fuse_score(dists, detections)
            matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)
            
            for i_tracked, idet in matches:
                # i_tracked is the id of the track and idet is the detection
                track = track_pool_dict[cls_id][i_tracked]
                det = detections[idet]
                if track.state == TrackState.Tracked:
                    # If the track is active, add the detection to the track
                    track.update(detections[idet], self.frame_id)
                    activated_tracks_dict[cls_id].append(track)
                else:
                    # We have obtained a detection from a track which is not active,
                    # hence put the track in refind_stracks list
                    track.re_activate(det, self.frame_id, new_id=False)
                    refined_tracks_dict[cls_id].append(track)
                    
            ''' Step 3: Second association, with low score detection boxes'''
            # association the untrack to the low score detections
            inds_low = pred_dets_dict[cls_id][:, 1:2] > 0.1
            inds_high = pred_dets_dict[cls_id][:, 1:2] < self.args.track_thresh
            inds_second = np.logical_and(inds_low, inds_high).squeeze(-1)
            pred_dets_cls_second = pred_dets_dict[cls_id][inds_second]
            
            # association the untrack to the low score detections
            if len(pred_dets_cls_second) > 0:
                detections_second = [
                    MCSTrack(
                        MCSTrack.tlbr_to_tlwh(tlbrs[2:6]),
                        tlbrs[1],
                        cls_id)
                    for tlbrs in pred_dets_cls_second
                ]
            else:
                detections_second = []
            
            r_tracked_stracks = [
                track_pool_dict[cls_id][i] for i in u_track
                if track_pool_dict[cls_id][i].state == TrackState.Tracked
            ]
            dists = matching.iou_distance(r_tracked_stracks, detections_second)
            matches, u_track, u_detection_second = matching.linear_assignment(
                    dists, thresh=0.5)
            
            for i_tracked, idet in matches:
                track = r_tracked_stracks[i_tracked]
                det = detections_second[idet]
                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated_tracks_dict[cls_id].append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refined_tracks_dict[cls_id].append(track)

            for it in u_track:
                track = r_tracked_stracks[it]
                if not track.state == TrackState.Lost:
                    track.mark_lost()
                    lost_tracks_dict[cls_id].append(track)
            
            '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
            detections = [detections[i] for i in u_detection]
            dists = matching.iou_distance(unconfirmed_dict[cls_id], detections)
            if not self.args.mot20:
                dists = matching.fuse_score(dists, detections)
            matches, u_unconfirmed, u_detection = matching.linear_assignment(
                dists, thresh=0.7)
            for i_tracked, idet in matches:
                unconfirmed_dict[cls_id][i_tracked].update(detections[idet], self.frame_id)
                activated_tracks_dict[cls_id].append(unconfirmed_dict[cls_id][i_tracked])
            for it in u_unconfirmed:
                track = unconfirmed_dict[cls_id][it]
                track.mark_removed()
                removed_tracks_dict[cls_id].append(track)
            
            """ Step 4: Init new stracks"""
            for inew in u_detection:
                track = detections[inew]
                if track.score < self.det_thresh:
                    continue
                track.activate(self.kalman_filter, self.frame_id)
                activated_tracks_dict[cls_id].append(track)
            
            """ Step 5: Update state"""
            for track in self.lost_tracks_dict[cls_id]:
                if self.frame_id - track.end_frame > self.max_time_lost:
                    track.mark_removed()
                    removed_tracks_dict[cls_id].append(track)

            self.tracked_tracks_dict[cls_id] = [
                t for t in self.tracked_tracks_dict[cls_id]
                if t.state == TrackState.Tracked
            ]
            self.tracked_tracks_dict[cls_id] = joint_stracks(
                self.tracked_tracks_dict[cls_id], activated_tracks_dict[cls_id])
            self.tracked_tracks_dict[cls_id] = joint_stracks(
                self.tracked_tracks_dict[cls_id], refined_tracks_dict[cls_id])
            self.lost_tracks_dict[cls_id] = sub_stracks(
                self.lost_tracks_dict[cls_id], self.tracked_tracks_dict[cls_id])
            self.lost_tracks_dict[cls_id].extend(lost_tracks_dict[cls_id])
            self.lost_tracks_dict[cls_id] = sub_stracks(
                self.lost_tracks_dict[cls_id], self.removed_tracks_dict[cls_id])
            self.removed_tracks_dict[cls_id].extend(removed_tracks_dict[cls_id])
            self.tracked_tracks_dict[cls_id], self.lost_tracks_dict[
                cls_id] = remove_duplicate_stracks(
                    self.tracked_tracks_dict[cls_id],
                    self.lost_tracks_dict[cls_id])
                
            # get scores of lost tracks
            output_tracks_dict[cls_id] = [
                track for track in self.tracked_tracks_dict[cls_id]
                if track.is_activated
            ]

        return output_tracks_dict


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
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
