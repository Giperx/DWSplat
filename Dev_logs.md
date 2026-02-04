20260202 commnet: base on 0128

20260203 TODO: 
1. 数据读取部分，增加符合OmniVGGT的输入。如masks, depth_indices, camera_indices。需要check读入的extrinsics, intrinsics格式是否正确。

OmniVGGT：
OmniVGGT输入model的intrinsics没有归一化，在extri_intri_to_pose_encoding中利用image.shape进行encode；camera_head返回到pose_encoding_to_extri_intri里面利用image.shape得到正常的intrinsics；
现在修改extri_intri_to_pose_encoding为处理归一化版本。

OmniVGGT：extrinsics在读入的时候只需[3, 4]，然后进行inverse得到world2cam形式。
extrinsics_selected在ZeroAggregator中进行归一化到cam0上。

OmniVGGT：输入的depth_indices, camera_indices为List。

src/dataset/shims/augmentation_shim.py的reflect_views需要适配现在的3x4形式的外参。

now：现在输入depth_indices, camera_indices先作为tensor，在ZeroAggregator中进一步处理转为List。depth_indices中tensor状态时全部设置为-1；     extrinsics在读入后先归一化cam0，inverse为w2c，再取[3, 4]形式。


2. 模型蒸馏部分，初始化OmniVGGT并copy到当前模型。

为了适配Gaussian Head需要DINO特征，进一步修改ZeroAggregator部分。额外输出DINO concat后的特征。


tar -czvf DWSplat0203.tar.gz --exclude=./DWSplat_0202/anysplat_hfog_1108 --exclude=./DWSplat_0202/datasets --exclude=./DWSplat_0202/outputs --exclude=./DWSplat_0202/.git --exclude=./DWSplat_0202/.vscode ./DWSplat_0202