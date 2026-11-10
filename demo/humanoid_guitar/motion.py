"""URDF frame kinematics with constrained hand mimics and prescribed arm IK."""
import numpy as np


def motion(axis, value, kind):
    t = np.eye(4)
    axis = np.asarray(axis, float)
    if kind == "prismatic":
        t[:3, 3] = axis * value
    elif kind == "revolute":
        x, y, z = axis
        cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        t[:3, :3] = np.eye(3) + np.sin(value) * cross + (1 - np.cos(value)) * cross @ cross
    return t


def resolve_joints(model, values):
    """Resolve mimic chains, then reject any actuated or mimic limit violation."""
    values={**model.get("preset",{}),**values}
    pending=[j for j in model["joints"] if j.get("mimic")]
    for j in model["joints"]:
        if not j.get("mimic"):values.setdefault(j["name"],0.)
    while pending:
        ready=[]
        for j in pending:
            m=j["mimic"];parent=m.get("joint",m.get("parent"))
            if parent in values:
                values[j["name"]]=float(m.get("multiplier",1))*values[parent]+float(m.get("offset",0));ready.append(j)
        if not ready:raise ValueError("Unresolved mimic chain")
        for j in ready:pending.remove(j)
    for j in model["joints"]:
        v=values[j["name"]]
        if not np.isfinite(v) or v<float(j.get("lower",-np.inf))-1e-8 or v>float(j.get("upper",np.inf))+1e-8:
            raise ValueError("Joint limit violation: "+j["name"]+"="+str(v))
    return values


class Kinematics:
    def __init__(self,model):
        self.model=model;self.placement=np.asarray(model["placement"],float)
        self.roots={n:np.asarray(t,float) for n,t in model["roots"].items()}
        self.joints=[];known=set(self.roots);pending=list(model["joints"])
        while pending:
            ready=[j for j in pending if j["parent"] in known]
            if not ready:raise ValueError("Unresolved link tree")
            for j in ready:
                if j["child"] in known:raise ValueError("Repeated child link")
                self.joints.append(j);known.add(j["child"]);pending.remove(j)
        self.names=model["control_joints"]
        default=resolve_joints(model,dict(zip(self.names,model.get("seed",[0.]*len(self.names)))))
        dynamic=set(self.names)
        while True:
            additional={j["name"] for j in self.joints if j.get("mimic") and j["mimic"].get("joint",j["mimic"].get("parent")) in dynamic}
            if additional.issubset(dynamic):break
            dynamic.update(additional)
        self.dynamic_joints=dynamic
        self.prepared=[]
        for j in self.joints:
            parent=np.asarray(j["parent_frame"]);child=np.linalg.inv(np.asarray(j["child_frame"]))
            fixed=None if j["name"] in self.dynamic_joints else parent@motion(j["axis"],default[j["name"]],j["kind"])@child
            self.prepared.append((j,parent,child,fixed))

    def forward(self,q):
        q=np.asarray(q,float)
        if q.shape!=(len(self.names),) or not np.isfinite(q).all():raise ValueError("Invalid joint vector")
        values=resolve_joints(self.model,dict(zip(self.names,q)))
        frames={n:self.placement@t for n,t in self.roots.items()}
        for j,parent,child,fixed in self.prepared:
            local=fixed if fixed is not None else parent@motion(j["axis"],values[j["name"]],j["kind"])@child
            frames[j["child"]]=frames[j["parent"]]@local
        tcp=frames[self.model["tcp_link"]]@np.asarray(self.model["tcp_transform"])
        return frames,tcp


def solve_targets(model,targets,rotations=None,seed=None,tolerance=.001):
    from scipy.optimize import least_squares
    kin=Kinematics(model);jm={j["name"]:j for j in model["joints"]}
    lo=np.array([jm[n].get("lower",-np.pi) for n in kin.names]);hi=np.array([jm[n].get("upper",np.pi) for n in kin.names])
    q=np.clip(np.asarray(seed if seed is not None else model.get("seed",np.zeros(len(lo))),float),lo+1e-7,hi-1e-7)
    qs=[];frames=[];errors=[];angles=[];names=list(kin.forward(q)[0])
    for i,target in enumerate(targets):
        target=np.asarray(target,float);prior=q.copy();rot=None if rotations is None else np.asarray(rotations[i],float)
        def residual(x):
            _,tcp=kin.forward(x)
            return np.r_[10*(tcp[:3,3]-target),([] if rot is None else .7*(tcp[:3,:3]-rot).ravel()),.001*(x-prior)]
        fit=least_squares(residual,q,bounds=(lo+1e-8,hi-1e-8),max_nfev=150,ftol=1e-10,xtol=1e-10,gtol=1e-10)
        q=fit.x;f,tcp=kin.forward(q);err=float(np.linalg.norm(tcp[:3,3]-target))
        if err>tolerance:raise RuntimeError(f"Unreachable pick target {i}: {err:.6f}m")
        qs.append(q.copy());frames.append([f[n] for n in names]);errors.append(err)
        angles.append(0. if rot is None else float(np.arccos(np.clip((np.trace(rot.T@tcp[:3,:3])-1)/2,-1,1))))
    return dict(joints=np.asarray(qs),link_transforms=np.asarray(frames),link_names=np.asarray(names),ik_error=np.asarray(errors),orientation_error=np.asarray(angles))
