"""Bounded joint-target sequencing gated by measured task state."""
import numpy as np


class ContactController:
    def __init__(self, grip, start, case, seated=None, seated_depth=.01,release_mode='drop',seating_preload=None):
        self.grip=np.asarray(grip,float);self.start=np.asarray(start,float);self.case=case
        self.high=self.grip+[0,0,.14];self.above=self.grip+[0,0,.12]
        self.seated=np.asarray(seated,float) if seated is not None else self.high.copy()
        self.seated_depth=seated_depth
        self.waypoint=np.array([.57,.094,1.27]);self.approach=self.seated-[0,.035,0]
        self.stage='above';self.since=0.;self.origin=self.start.copy();self.dwell=None
        self.release_mode=release_mode;self.acquired=False
        self.seating_preload=seating_preload
        self.preload_peak_m=0.;self.seating_contact_since=None;self.seating_last_time=None;self.seating_observation={};self.seating_events=[]
        if seating_preload is not None:
            values=[seating_preload[k] for k in ('limit_m','speed_m_s','depth_margin_m','contact_dwell_s')]
            if not np.isfinite(values).all() or not (0<values[0]<=values[2]<.001) or not (0<values[1]<=.001) or values[3]<.1:
                raise ValueError('bounded sub-mm seating travel/depth, slow speed and >=0.1s contact dwell required')
        self.failure=None;self.failures=[];self.events=[];self.baseline=None;self.release_point=None

    def record_failure(self,failure):
        self.failures.append(failure)
        if self.failure is None:self.failure=failure

    def adapt_grasp(self,delta):
        """Update robot TCP goals from a measured held-object offset; never move the object."""
        delta=np.asarray(delta,dtype=float)
        if delta.shape!=(3,) or not np.isfinite(delta).all():raise ValueError('finite measured grasp offset required')
        self.seated+=delta;self.approach+=delta;self.waypoint+=delta

    @property
    def phase(self):
        return {'above':'approach','descend':'approach','route1':'route','done':'release'}.get(self.stage,self.stage)

    def transition(self, stage, t, state):
        self.events.append(dict(t_s=float(t),from_stage=self.stage,to_stage=stage,tcp=np.asarray(state['tcp']).tolist(),
                                plug_z_m=float(state['z']),finger_forces_N=[float(state['left']),float(state['right'])]))
        if self.stage=='insert' and self.seating_preload:
            reason='stable measured socket contact' if stage=='release' and self.failure is None else (self.failure or {}).get('reason','insertion ended')
            self.seating_observation['stop_reason']=reason
            self.seating_events.append(dict(self.seating_observation,event='stopped',t_s=float(t)))
        self.stage=stage;self.since=t;self.origin=np.asarray(state['tcp']).copy();self.dwell=None
        if stage=='lift':self.acquired=True
        if stage=='release':self.release_point=self.origin.copy()

    def command(self,t,state):
        if self.baseline is None:self.baseline=state['z']
        elapsed=t-self.since
        stages={'above':(self.above,1.3,6.,'descend'),
                'descend':(self.grip,1.5,5.,'close'),
                'lift':(self.high,1.5,4.,'hold'),
                'route1':(self.waypoint,2.5,6.,'align'),
                'align':(self.approach,2.5,6.,'insert' if self.case=='insert' else 'release'),
                'insert':(self.seated,4.,7.,'release'),
                'place':(self.grip,2.5,6.,'release')}
        bilateral=state['left']>=.05 and state['right']>=.05
        if self.stage in stages:
            target,duration,timeout,next_stage=stages[self.stage]
            if self.stage=='insert' and self.seating_preload:
                settings=self.seating_preload
                preload_dt=0. if self.seating_last_time is None else max(0.,min(t-self.seating_last_time,elapsed-duration))
                self.seating_last_time=t
                self.seating_observation=dict(t_s=float(t),target_preload_m=float(self.preload_peak_m),
                    actual_depth_m=float(state['depth']),socket_force_N=float(state['socket']),
                    actual_axial_tcp_offset_m=float(np.asarray(state['tcp'])[1]-self.seated[1]),contact_dwell_s=0.)
                if state['depth']>self.seated_depth+settings['depth_margin_m']:
                    self.record_failure(dict(stage='insert',reason='actual depth exceeds bounded seating margin',t_s=float(t),depth_m=float(state['depth'])))
                    self.transition('place',t,state);return self.command(t,state)
                safe=(elapsed>=duration and np.linalg.norm(np.asarray(state['tcp'])-self.seated)<.004
                      and state['angle']<.06 and abs(state['depth']-self.seated_depth)<.001
                      and state['lateral']<.001 and state['axis']<.08)
                if safe and state['socket']<.01:
                    previous=self.preload_peak_m
                    self.preload_peak_m=min(settings['limit_m'],previous+settings['speed_m_s']*preload_dt)
                    if previous==0. and self.preload_peak_m>0.:
                        self.seating_events.append(dict(self.seating_observation,target_preload_m=float(self.preload_peak_m),event='preload_started'))
                target=target+[0,self.preload_peak_m,0]
                self.seating_observation['target_preload_m']=float(self.preload_peak_m)
            u=np.clip(elapsed/duration,0,1);u=u*u*(3-2*u)
            goal=self.origin+(target-self.origin)*u
            reached=np.linalg.norm(np.asarray(state['tcp'])-target)<(.0025 if self.stage=='descend' else .004) and state['angle']<.06
            if self.stage=='place':reached=reached and abs(state['z']-self.baseline)<.005
            if self.stage=='lift':reached=reached and state['z']-self.baseline>=.08 and bilateral
            if self.stage=='align' and self.case=='insert':reached=reached and state['axis']<.08 and state['lateral']<.001
            if self.stage=='insert':
                reached=reached and abs(state['depth']-self.seated_depth)<.001 and state['socket']>=.01
                if self.seating_preload:
                    good=elapsed>=duration and reached and state['lateral']<.001 and state['axis']<.08
                    self.seating_contact_since=t if good and self.seating_contact_since is None else self.seating_contact_since if good else None
                    dwell=0. if self.seating_contact_since is None else t-self.seating_contact_since
                    self.seating_observation['contact_dwell_s']=float(dwell)
                    reached=good and dwell>=self.seating_preload['contact_dwell_s']
            if elapsed>=duration and reached:
                self.transition(next_stage,t,state)
                return self.command(t,state)
            if elapsed>timeout:
                self.record_failure(dict(stage=self.stage,reason='measured state did not reach bounded phase gate',t_s=float(t),position_error_m=float(np.linalg.norm(np.asarray(state['tcp'])-target))))
                self.transition('place' if (self.release_mode=='place' or self.case=='insert') and self.stage!='place' else 'release',t,state);return self.command(t,state)
            return goal,.04 if self.stage in ('above','descend') or (self.stage=='place' and not self.acquired) else .002,self.phase
        if self.stage=='close':
            # A grasp needs load on BOTH measured finger/plug pairs and a plausible 14 mm pinch.
            good=bilateral and .008<=state['width']<=.025
            self.dwell=t if good and self.dwell is None else self.dwell if good else None
            if self.dwell is not None and t-self.dwell>=.1:
                self.transition('lift',t,state);return self.command(t,state)
            if elapsed>2.5:
                self.record_failure(dict(stage='close',reason='no sustained bilateral contact at connector width',t_s=float(t)))
                self.transition('place' if (self.release_mode=='place' or self.case=='insert') and self.stage!='place' else 'release',t,state);return self.command(t,state)
            return self.grip,.04+(.002-.04)*min(elapsed/.8,1),self.phase
        if self.stage=='hold':
            good=bilateral and state['z']-self.baseline>=.08
            self.dwell=t if good and self.dwell is None else self.dwell if good else None
            if self.dwell is not None and t-self.dwell>=.8:
                self.transition(('place' if self.release_mode=='place' else 'release') if self.case=='grasp' else 'route1',t,state);return self.command(t,state)
            if elapsed>3.:
                self.record_failure(dict(stage='hold',reason='lift/contact not sustained',t_s=float(t)))
                self.transition('place' if (self.release_mode=='place' or self.case=='insert') and self.stage!='place' else 'release',t,state);return self.command(t,state)
            return self.high,.002,self.phase
        if self.stage in ('release','done'):
            opening=.002+(.04-.002)*np.clip(elapsed/.5,0,1) if self.stage=='release' else .04
            retreat=np.array([0,0,.12])*np.clip((elapsed-1)/1.5,0,1) if self.release_mode=='place' or (self.case=='insert' and self.failure is None) else 0.
            goal=self.release_point+retreat
            return goal,float(opening),'release'
        raise RuntimeError('Unknown controller stage '+self.stage)
