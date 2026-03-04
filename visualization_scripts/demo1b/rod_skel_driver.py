import numpy as np
from pxr import UsdSkel, Gf, Usd, UsdGeom, UsdPhysics
try:
    from pxr import PhysxSchema
except Exception:
    PhysxSchema = None

class SkeletonRodDriver:
    def __init__(self, stage, skeleton_path, assume_chain=True): # 添加参数
        self.stage = stage
        self.skeleton_path = skeleton_path
        self.skel_prim = None
        self.anim_prim = None
        self.assume_chain = assume_chain  # 保存该属性
        self.num_joints = 0
        self.parent_indices = None

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
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
            removed = True
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            prim.RemoveAPI(UsdPhysics.CollisionAPI)
            removed = True
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            removed = True
        if PhysxSchema is not None:
            for name in dir(PhysxSchema):
                if not name.endswith("API"):
                    continue
                api_cls = getattr(PhysxSchema, name, None)
                try:
                    if api_cls is not None and prim.HasAPI(api_cls):
                        prim.RemoveAPI(api_cls)
                        removed = True
                except Exception:
                    continue
        return removed

    def _disable_physics_on_skeleton(self):
        if self.skel_prim is None or not self.skel_prim.IsValid():
            raise RuntimeError("Skeleton prim is not set; cannot disable physics.")
        root_prim = self._find_skel_root_prim() or self.skel_prim
        removed_count = 0
        for p in Usd.PrimRange(root_prim):
            if self._remove_physics_api(p):
                removed_count += 1
        parent = root_prim.GetParent()
        while parent and parent.IsValid():
            if self._remove_physics_api(parent):
                removed_count += 1
            parent = parent.GetParent()
        print(f"Disabled physics schemas on skeleton hierarchy: {removed_count} prim(s)")

    def load_asset(self, asset_usd_path):
        """
        集成步骤：将外部骨骼模型 USD 载入到场景中
        """
        # 1. 创建容器并引用模型
        prim = self.stage.DefinePrim(self.skeleton_path, "Xform")
        prim.GetReferences().AddReference(asset_usd_path)
        
        # --- 核心改进：递归搜索 Skeleton 类型的节点 ---
        found_skel = None
        # 使用 PrimRange 遍历 /World/CableAssembly 下的所有子孙节点
        for p in Usd.PrimRange(self.stage.GetPrimAtPath(self.skeleton_path)):
            if p.IsA(UsdSkel.Skeleton):
                found_skel = p
                break
        
        if found_skel:
            self.skel_prim = found_skel
            # 更新路径为实际找到的骨骼路径，确保后续动画绑定正确
            self.skeleton_path = found_skel.GetPath().pathString
        else:
            # 调试信息：如果没找到，打印出该路径下的结构，帮你定位问题
            print(f"调试：在 {self.skeleton_path} 路径下发现的节点有：")
            for p in Usd.PrimRange(self.stage.GetPrimAtPath(self.skeleton_path)):
                print(f"  - {p.GetPath()} [{p.GetTypeName()}]")
            raise RuntimeError(f"在 {asset_usd_path} 中没找到有效的 Skeleton Prim")

        # 3. 初始化动画
        self._setup_animation()
        print(f"成功加载模型并初始化骨骼: {self.skeleton_path}")

    def _setup_animation(self):
        """内部方法：设置动画绑定"""
        self._disable_physics_on_skeleton()
        self.skel = UsdSkel.Skeleton(self.skel_prim)
        self.joints = self.skel.GetJointsAttr().Get() or []
        self.num_joints = len(self.joints)
        topo_cls = getattr(UsdSkel, "Topology", None) or getattr(UsdSkel, "SkelTopology", None)
        if topo_cls is not None:
            try:
                topo = topo_cls(self.joints)
                self.parent_indices = topo.GetParentIndices()
            except Exception:
                self.parent_indices = None
        
        # 获取 Rest Transforms 保持初始比例
        self.rest_transforms = self.skel.GetRestTransformsAttr().Get()
        
        # 创建 SkelAnimation 节点
        self.anim_path = self.skeleton_path + "/RealtimeAnim"
        if self.stage.GetPrimAtPath(self.anim_path).IsValid():
            self.stage.RemovePrim(self.anim_path)
        
        self.anim_prim = self.stage.DefinePrim(self.anim_path, "SkelAnimation")
        UsdSkel.BindingAPI.Apply(self.skel_prim).CreateAnimationSourceRel().SetTargets([self.anim_path])
        
        self.anim = UsdSkel.Animation(self.anim_prim)
        self.anim.CreateJointsAttr().Set(self.joints)
        
        # 创建属性句柄
        self.rot_attr = self.anim.CreateRotationsAttr()
        self.trans_attr = self.anim.CreateTranslationsAttr()
        self.scale_attr = self.anim.CreateScalesAttr()

    def _orthonormalize(self, R):
        U, _, Vt = np.linalg.svd(R)
        Rn = U @ Vt
        if np.linalg.det(Rn) < 0:
            U[:, -1] *= -1
            Rn = U @ Vt
        return Rn

    def _mat3_to_quat(self, R):
        """Matrix to Gf.Quatf"""
        # 简单转换逻辑
        tr = np.trace(R)
        if tr > 0:
            s = np.sqrt(tr + 1.0) * 2
            w, x, y, z = 0.25 * s, (R[2,1]-R[1,2])/s, (R[0,2]-R[2,0])/s, (R[1,0]-R[0,1])/s
        else:
            # 简化版逻辑，实际生产建议用你原脚本中更健壮的判断
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

    def update_skeleton(self, rod_pos, rod_dir, time_code):
        """
        根据 rod_pos (3, N_nodes) 和 rod_dir (3, 3, N_elems) 更新骨骼
        """
        tc = Usd.TimeCode.Default() if time_code is None else time_code
        # Convert world-space rod state into skeleton-local frame.
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
        
        # 1. 计算世界坐标系下的变换
        for i in range(self.num_joints):
            p0 = rod_pos[:, i].astype(np.float64)
            if w2s is not None:
                p0 = np.array(
                    w2s.Transform(
                        Gf.Vec3d(float(p0[0]), float(p0[1]), float(p0[2]))
                    )
                )
            t_world.append(p0)
            
            # 使用 rod_dir 构建旋转 (对应你 animate 脚本中的逻辑)
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

        # 2. 转换为局部坐标系 (Local Space)
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

        # 3. 写入 USD 属性
        self.rot_attr.Set(rotations, tc)
        self.trans_attr.Set(translations, tc)
        self.scale_attr.Set(scales, tc)
