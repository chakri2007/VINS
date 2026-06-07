# database.py
import numpy as np

class FeatureDatabase:
    def __init__(self, max_history_length=10):
        self.max_history_length = max_history_length
        self.next_id = 0
        self.reference_frame = None
        
        self.ids = np.empty((0,), dtype=np.int64)
        self.points = np.empty((0, 2), dtype=np.float32)
        self.ages = np.empty((0,), dtype=np.int32)
        
        # Historical dictionary map to store trajectories for the ROS visualizer node
        # Structure: { feature_id: [(u_curr, v_curr), (u_prev1, v_prev1), ...] }
        self.history_map = {}

    def initialize_ledger(self, initial_points, gray_frame, timestamp):
        num_points = len(initial_points)
        if num_points == 0:
            return
            
        self.ids = np.arange(self.next_id, self.next_id + num_points, dtype=np.int64)
        self.next_id += num_points
        
        self.points = np.array(initial_points, dtype=np.float32)
        self.ages = np.ones((num_points,), dtype=np.int32)
        
        self.history_map = {
            id_val: [(pt[0], pt[1])] for id_val, pt in zip(self.ids, self.points)
        }
        
        self.reference_frame = gray_frame.copy()

    def get_active_tracks(self):
        return self.reference_frame, self.points, self.ids

    def update_active_positions(self, tracked_ids, tracked_curr_points):
        if len(tracked_ids) == 0:
            return
            
        _, indices_in_ledger, indices_in_tracked = np.intersect1d(
            self.ids, tracked_ids, return_indices=True
        )
        
        self.points[indices_in_ledger] = tracked_curr_points[indices_in_tracked]
        self.ages[indices_in_ledger] += 1
        
        for i_ledger, i_tracked in zip(indices_in_ledger, indices_in_tracked):
            feat_id = self.ids[i_ledger]
            pt = tracked_curr_points[i_tracked]
            
            if feat_id in self.history_map:
                self.history_map[feat_id].insert(0, (pt[0], pt[1]))
                if len(self.history_map[feat_id]) > self.max_history_length:
                    self.history_map[feat_id].pop()

    def purge_tracks(self, lost_ids):
        if len(lost_ids) == 0:
            return
            
        keep_mask = ~np.isin(self.ids, lost_ids)
        
        self.ids = self.ids[keep_mask]
        self.points = self.points[keep_mask]
        self.ages = self.ages[keep_mask]
        
        for fid in lost_ids:
            if fid in self.history_map:
                del self.history_map[fid]

    def add_new_tracks(self, new_points, gray_frame):
        num_new = len(new_points)
        if num_new == 0:
            return
            
        new_ids = np.arange(self.next_id, self.next_id + num_new, dtype=np.int64)
        self.next_id += num_new

        self.ids = np.concatenate([self.ids, new_ids])
        self.points = np.concatenate([self.points, np.array(new_points, dtype=np.float32)])
        self.ages = np.concatenate([self.ages, np.ones((num_new,), dtype=np.int32)])

        for nid, npt in zip(new_ids, new_points):
            self.history_map[nid] = [(npt[0], npt[1])]

    def set_reference_frame(self, gray_frame):
        self.reference_frame = gray_frame.copy()

    def get_active_feature_histories(self):
        return self.history_map.copy()