"""Conservative observation association, suspended trails and multi-frame apparent displacement."""

import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from .dynamic_tracking import center,iou


class ConservativeTracker:
    def __init__(self,config):
        self.config=config
        self.reset()

    def reset(self):
        self.tracks,self.next_id,self.last_timestamp={},1,None
        self.suspended_ids=[]

    def update(self,detections,timestamp,frame_index,image=None):
        if not math.isfinite(timestamp) or (self.last_timestamp is not None and timestamp <= self.last_timestamp):
            raise ValueError("Timestamp inválido para acompanhamento")
        self.last_timestamp=timestamp
        cfg=self.config
        self.tracks={key:t for key,t in self.tracks.items() if timestamp-t['time'] <= cfg['max_gap_seconds']+1e-9}
        keys=sorted(self.tracks)
        cost=np.full((len(keys),len(detections)),1e6)
        for row,key in enumerate(keys):
            old=self.tracks[key]
            dt=timestamp-old['time']
            velocity=np.median(old['velocities'][-3:],axis=0) if old['velocities'] else np.zeros(2)
            predicted=np.asarray(old['box'])+np.tile(velocity*dt,2)
            old_size=np.asarray(old['box'][2:])-old['box'][:2]
            for col,det in enumerate(detections):
                box=np.asarray(det['xyxy'])
                size=box[2:]-box[:2]
                diagonal=max(1.,min(np.linalg.norm(size),np.linalg.norm(old_size)))
                innovation=np.linalg.norm(center(box)-center(predicted))/diagonal
                step_rate=np.linalg.norm(center(box)-center(old['box']))/diagonal/dt
                area_ratio=float(np.prod(size)/np.prod(old_size))
                overlap=iou(box,predicted)
                # All gates apply, unlike the legacy OR gate. No location/central exclusion.
                if (overlap>=cfg['min_iou'] and innovation<=cfg['max_innovation_diagonals']
                    and step_rate<=cfg['max_center_speed_diagonals_per_second']
                    and 1/cfg['max_area_ratio']<=area_ratio<=cfg['max_area_ratio']):
                    cost[row,col]=1-overlap+innovation
        matches={}
        if cost.size:
            rr,cc=linear_sum_assignment(cost)
            for row,col in zip(rr,cc):
                value=cost[row,col]
                if value>=1e6:continue
                alternatives=np.concatenate([np.delete(cost[row],col),np.delete(cost[:,col],row)])
                margin=float(alternatives.min()-value) if len(alternatives) else 1e6
                if margin>=cfg['min_assignment_margin']:
                    matches[int(col)]=(keys[int(row)],float(value),min(margin,1e6))
        observations=[]
        observed_ids=set()
        for number,det in enumerate(detections):
            box,position=det['xyxy'],center(det['xyxy'])
            match=matches.get(number)
            assignment_cost,margin=None,None
            points,velocities,streak,total=[],[],1,1
            if match is None:
                ident=self.next_id;self.next_id+=1
            else:
                ident,assignment_cost,margin=match
                old=self.tracks[ident]
                total=old['observations']+1
                if old['frame']==frame_index-1:
                    streak=old['streak']+1
                    points=[p for p in old['points'] if timestamp-p[0] <= cfg['trail_seconds']+1e-9]
                    velocities=(old['velocities']+[(position-center(old['box']))/(timestamp-old['time'])])[-3:]
            points.append([float(timestamp),*position.tolist()])
            confirmed=streak>=cfg['confirmation_frames']
            speed=None
            recent=[p for p in points if timestamp-p[0]<=cfg['speed_window_seconds']+1e-9]
            if (confirmed and streak>=cfg['min_speed_observations'] and len(recent)>=cfg['min_speed_observations']
                and timestamp-recent[0][0]>=cfg['min_speed_history_seconds']-1e-9):
                speed=float(np.linalg.norm(position-np.asarray(recent[0][1:]))/(timestamp-recent[0][0]))
            self.tracks[ident]={'box':box,'time':timestamp,'frame':frame_index,'streak':streak,
                'observations':total,'points':points,'velocities':velocities}
            observed_ids.add(ident)
            observations.append({**det,'track_id':ident,'track_observations':total,'consecutive_observations':streak,
                'track_confirmed':confirmed,'tracking_status':'confirmed' if confirmed else 'tentative',
                'association_cost':assignment_cost,'association_margin':margin,'observed_current_frame':True,
                'position_predicted_only':False,'center_xy':position.tolist(),'speed_source_px_per_second':speed,
                'speed_basis':'net observed center displacement / elapsed time over recent consecutive history',
                'trail_txy':points if confirmed else []})
        self.suspended_ids=sorted(set(self.tracks)-observed_ids)
        return observations
