import numpy as np
import inspect
from pxr import UsdSkel, Gf, Usd, UsdGeom, UsdPhysics
try:
    from pxr import PhysxSchema
except Exception:
    PhysxSchema = None

class SkeletonRodDriver:
    def __init__(self, stage, skeleton_path, assume_chain=True):
        self.stage = stage
        self.skeleton_path = skeleton_path
        self.skel_prim = None
        self.anim_prim = None
        self.assume_chain = assume_chain
        self.num_joints = 0
        self.skel_cache = None

    def _find_skel_root_prim(self):
        if self.skel_prim is None or not self.skel_prim.IsValid():
            return None
        p = self.skel_prim
        while p and p.IsValid():
            if p.GetTypeName() == "SkelRoot":
                return p
            p = p.GetParent()
        return None

    def _remove_physics_api(self, prim):
        removed = False
        # Best-effort removal for any applied PhysX/Physics schemas by token name.
        if hasattr(prim, "GetAppliedSchemas"):
            try:
                for schema_name in prim.GetAppliedSchemas():
                    if ("Physx" in schema_name) or ("Physics" in schema_name):
                        if hasattr(prim, "RemoveAppliedSchema"):
                            prim.RemoveAppliedSchema(schema_name)
                            print(f"  Removed applied schema {schema_name} from {prim.GetPath()}")
                            removed = True
            except Exception:
                pass

        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
            print(f"  Removed RigidBodyAPI from {prim.GetPath()}")
            removed = True
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            prim.RemoveAPI(UsdPhysics.CollisionAPI)
            print(f"  Removed CollisionAPI from {prim.GetPath()}")
            removed = True
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            print(f"  Removed ArticulationRootAPI from {prim.GetPath()}")
            removed = True
        if PhysxSchema is not None:
            # Remove any applied PhysX API classes we can discover
            for name in dir(PhysxSchema):
                if not name.endswith("API"):
                    continue
                api_cls = getattr(PhysxSchema, name, None)
                if not inspect.isclass(api_cls):
                    continue
                try:
                    if prim.HasAPI(api_cls):
                        prim.RemoveAPI(api_cls)
                        print(f"  Removed {name} from {prim.GetPath()}")
                        removed = True
                except Exception:
                    continue
        return removed

    def _disable_physics_on_skeleton(self):
        """
        Disable physics on the skeleton and all of its descendants.
        """
        if self.skel_prim is None or not self.skel_prim.IsValid():
            raise RuntimeError("Skeleton prim is not set; cannot disable physics.")

        root_prim = self._find_skel_root_prim() or self.skel_prim
        print(f"Checking physics on skeleton: {root_prim.GetPath()} (skel: {self.skel_prim.GetPath()})")

        # Walk every prim under the skeleton root.
        for p in Usd.PrimRange(root_prim):
            self._remove_physics_api(p)

        # Also check ancestors: the rigid body sometimes sits on a parent Xform.
        parent = root_prim.GetParent()
        while parent and parent.IsValid():
            self._remove_physics_api(parent)
            parent = parent.GetParent()
        
        # Isaac Sim specific: opt the prim out of physics entirely.
        # from pxr import PhysxSchema
        # if not p.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
        #     PhysxSchema.PhysxRigidBodyAPI.Apply(p)
        # PhysxSchema.PhysxRigidBodyAPI(p).CreateDisableSimulationAttr(True)

    def load_asset(self, asset_usd_path):
        """
        Load an external skinned-skeleton USD asset into the stage.
        """
        # 1. Create a container prim and reference the asset into it.
        prim = self.stage.DefinePrim(self.skeleton_path, "Xform")
        prim.GetReferences().AddReference(asset_usd_path)
        
        # Recursively search for a Skeleton prim: the asset may nest it at any depth.
        found_skel = None
        # PrimRange walks every descendant under the container prim.
        for p in Usd.PrimRange(self.stage.GetPrimAtPath(self.skeleton_path)):
            if p.IsA(UsdSkel.Skeleton):
                found_skel = p
                break
        
        if found_skel:
            self.skel_prim = found_skel
            # Point at the skeleton we actually found so animation binding targets it.
            self.skeleton_path = found_skel.GetPath().pathString
        else:
            # Nothing found: dump the subtree so the mismatch is easy to spot.
            print(f"[debug] prims found under {self.skeleton_path}:")
            for p in Usd.PrimRange(self.stage.GetPrimAtPath(self.skeleton_path)):
                print(f"  - {p.GetPath()} [{p.GetTypeName()}]")
            raise RuntimeError(f"No valid Skeleton prim found in {asset_usd_path}")

        # 2. Initialize the animation binding.
        self._setup_animation()
        print(f"Loaded asset and initialized skeleton: {self.skeleton_path}")

    def _setup_animation(self):
        """Create the SkelAnimation prim and bind it to the skeleton."""
        self._disable_physics_on_skeleton()
        self.skel = UsdSkel.Skeleton(self.skel_prim)
        self.joints = self.skel.GetJointsAttr().Get() or []
        self.num_joints = len(self.joints)
        self.parent_indices = None
        topo_cls = getattr(UsdSkel, "Topology", None) or getattr(UsdSkel, "SkelTopology", None)
        if topo_cls is not None:
            try:
                topo = topo_cls(self.joints)
                self.parent_indices = topo.GetParentIndices()
            except Exception:
                self.parent_indices = None
        
        # Rest transforms are kept so the original per-joint scale is preserved.
        self.rest_transforms = self.skel.GetRestTransformsAttr().Get()
        
        # Create the SkelAnimation prim under /World to dodge instancing/reference limits.
        anim_parent = self.stage.DefinePrim("/World/SkelAnimations", "Xform")
        self.anim_path = anim_parent.GetPath().AppendChild(
            f"{self.skel_prim.GetName()}_RealtimeAnim"
        ).pathString

        if self.stage.GetPrimAtPath(self.anim_path).IsValid():
            self.stage.RemovePrim(self.anim_path)
        
        self.anim_prim = self.stage.DefinePrim(self.anim_path, "SkelAnimation")
        binding = UsdSkel.BindingAPI.Apply(self.skel_prim)
        binding.CreateAnimationSourceRel().SetTargets([self.anim_prim.GetPath()])
        skel_root = self._find_skel_root_prim()
        if skel_root and skel_root != self.skel_prim:
            UsdSkel.BindingAPI.Apply(skel_root).CreateAnimationSourceRel().SetTargets([self.anim_prim.GetPath()])
        print(
            "Skeleton instance:",
            self.skel_prim.IsInstance(),
            "instanceable:",
            self.skel_prim.IsInstanceable(),
            "SkelRoot:",
            skel_root.GetPath() if skel_root else None,
            "SkelRoot instance:",
            skel_root.IsInstance() if skel_root else None,
            "SkelRoot instanceable:",
            skel_root.IsInstanceable() if skel_root else None,
        )

        self._ensure_mesh_bindings()

        self.anim = UsdSkel.Animation(self.anim_prim)
        self.anim.CreateJointsAttr().Set(self.joints)
        
        # Cache the attribute handles written every frame.
        self.rot_attr = self.anim.CreateRotationsAttr()
        self.trans_attr = self.anim.CreateTranslationsAttr()
        self.scale_attr = self.anim.CreateScalesAttr()
        self.joint_xforms_attr = None

        # SkelCache handle, used to force a refresh after writing new samples.
        self.skel_cache = UsdSkel.Cache()

    def _ensure_mesh_bindings(self):
        root_prim = self._find_skel_root_prim() or self.skel_prim
        skel_path = self.skel_prim.GetPath()
        bound = 0
        candidates = 0
        for p in Usd.PrimRange(root_prim):
            if not p.IsA(UsdGeom.Mesh):
                continue
            bind = UsdSkel.BindingAPI(p)
            # Only bind meshes that already have joint indices/weights
            ji = bind.GetJointIndicesPrimvar()
            jw = bind.GetJointWeightsPrimvar()
            ji_attr = ji.GetAttr() if ji else None
            jw_attr = jw.GetAttr() if jw else None
            ji_ok = bool(ji_attr and ji_attr.IsValid() and ji_attr.HasAuthoredValue())
            jw_ok = bool(jw_attr and jw_attr.IsValid() and jw_attr.HasAuthoredValue())
            if not (ji_ok and jw_ok):
                continue
            candidates += 1
            rel = bind.GetSkeletonRel()
            targets = rel.GetTargets() if rel else []
            print(f"  Skinned mesh: {p.GetPath()} skeleton targets: {targets}")
            if (not targets) or (skel_path not in targets):
                UsdSkel.BindingAPI.Apply(p).CreateSkeletonRel().SetTargets([skel_path])
                bound += 1
        print(f"Skel mesh bindings: candidates={candidates}, newly_bound={bound}")

    def _orthonormalize(self, R):
        U, _, Vt = np.linalg.svd(R)
        Rn = U @ Vt
        if np.linalg.det(Rn) < 0:
            U[:, -1] *= -1
            Rn = U @ Vt
        return Rn

    def _mat3_to_quat(self, R):
        """Matrix to Gf.Quatf"""
        # Standard branch on the matrix trace for numerical stability.
        tr = np.trace(R)
        if tr > 0:
            s = np.sqrt(tr + 1.0) * 2
            w, x, y, z = 0.25 * s, (R[2,1]-R[1,2])/s, (R[0,2]-R[2,0])/s, (R[1,0]-R[0,1])/s
        else:
            # Trace is non-positive: pivot on the largest diagonal entry.
            i = np.argmax([R[0,0], R[1,1], R[2,2]])
            j, k = (i+1)%3, (i+2)%3
            s = np.sqrt(R[i,i] - R[j,j] - R[k,k] + 1.0) * 2
            q = [0,0,0,0]
            q[i+1] = 0.25 * s
            q[0] = (R[k,j]-R[j,k])/s
            q[j+1] = (R[i,j]+R[j,i])/s
            q[k+1] = (R[i,k]+R[k,i])/s
            w, x, y, z = q
        return Gf.Quatf(float(w), float(x), float(y), float(z))

    def update_skeleton(self, rod_pos, rod_dir, time_code = None):
        """
        Drive the skeleton from rod_pos (3, N_nodes) and rod_dir (3, 3, N_elems).
        """
        tc = Usd.TimeCode.Default() if time_code is None else time_code
        # Convert world-space inputs into skeleton local space if needed
        try:
            skel_xf = UsdGeom.Xformable(self.skel_prim)
            skel_l2w = skel_xf.ComputeLocalToWorldTransform(tc)
            w2s = skel_l2w.GetInverse()
            R_w2s = np.array(w2s.ExtractRotationMatrix())
        except Exception:
            w2s = None
            R_w2s = None
        R_world = []
        t_world = []
        
        # 1. Build the world-space transform of every joint.
        for i in range(self.num_joints):
            p0 = rod_pos[:, i].astype(np.float64)
            if w2s is not None:
                p0 = np.array(w2s.Transform(Gf.Vec3d(float(p0[0]), float(p0[1]), float(p0[2]))))
            t_world.append(p0)
            
            # Build the rotation from the rod director frame.
            D = rod_dir[:, :, i].astype(np.float64) 
            Rw = np.zeros((3, 3), dtype=np.float64)
            Rw[:, 1] = D[2] # Tangent
            Rw[:, 0] = D[1] # Normal
            Rw[:, 2] = D[0] # Binormal
            Rw = self._orthonormalize(Rw)
            if R_w2s is not None:
                Rw = R_w2s @ Rw
            R_world.append(self._orthonormalize(Rw))

        rotations, translations, scales = [], [], []

        # 2. Convert world-space transforms into joint-local space.
        for i in range(self.num_joints):
            # Scale
            rest_matrix = self.rest_transforms[i]
            s = Gf.Vec3h(rest_matrix.GetRow(0).GetLength(), 
                         rest_matrix.GetRow(1).GetLength(), 
                         rest_matrix.GetRow(2).GetLength())
            scales.append(s)

            # Parent hierarchy
            if self.parent_indices is not None and i < len(self.parent_indices):
                parent = self.parent_indices[i]
            else:
                parent = (i - 1) if (self.assume_chain and i > 0) else -1
            
            if parent < 0:
                R_loc = R_world[i]
                t_loc = t_world[i]
            else:
                Rp = R_world[parent]
                tp = t_world[parent]
                R_loc = Rp.T @ R_world[i]
                t_loc = Rp.T @ (t_world[i] - tp)

            rotations.append(self._mat3_to_quat(self._orthonormalize(R_loc)))
            translations.append(Gf.Vec3f(float(t_loc[0]), float(t_loc[1]), float(t_loc[2])))

        # 3. Write the sampled attributes at this time code.
        self.rot_attr.Set(rotations, tc)
        self.trans_attr.Set(translations, tc)
        self.scale_attr.Set(scales, tc)
    # Force the skeleton cache to refresh.
        self.skel_cache.Clear()
