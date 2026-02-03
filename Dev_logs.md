20260202 commnet: base on 0128

20260203 TODO: 
1. 数据读取部分，增加符合OmniVGGT的输入。如masks, depth_indices, camera_indices。需要check读入的extrinsics, intrinsics格式是否正确。

输入的depth_indices, camera_indices为List，extrinsics（cam2world）在读入src/dataset/dataset_nuscenes.py后先归一化到第一个相机，再inverse到world2cam形式[3, 4]。

2. 模型蒸馏部分，初始化OmniVGGT并copy到当前模型。（暂时使用OG gs head。）

