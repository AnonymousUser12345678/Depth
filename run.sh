# python main.py --data kitti --student_arch resnet34 --loss mse --student_input_size 64
# python main.py --data kitti --student_arch resnet34 --loss mse --student_input_size 32

# python main.py --data kitti --student_arch resnet18 --loss mse --student_input_size 64 --global_info
# py/thon main.py --data kitti --student_arch resnet18 --loss mse --student_input_size 32 --global_info --epochs 30

# python main.py --data kitti --student_arch resnet34 --loss mse --student_input_size 64 --global_info --epochs 30
# python main.py --data kitti --student_arch resnet34 --loss mse --student_input_size 32 --global_info --epochs 30

# python test.py --data kitti --loss mse --student_arch resnet18 --student_input 64
# python test.py --data kitti --loss mse --student_arch resnet18 --student_input 32

# python test.py --data kitti --loss mse --student_arch resnet34 --student_input 64 --global_info
# python test.py --data kitti --loss mse --student_arch resnet34 --student_input 32 --global_info
# python test.py --data kitti --loss mse --student_arch resnet34 --student_input 64
# python test.py --data kitti --loss mse --student_arch resnet34 --student_input 32

# python time_energy.py --student_arch resnet18 --student_input_size 64 --data kitti
# python time_energy.py --student_arch resnet18 --student_input_size 32 --data kitti
# python time_energy.py --student_arch resnet18 --student_input_size 64 --global_info --data kitti
# python time_energy.py --student_arch resnet18 --student_input_size 32 --global_info --data kitti

# python time_energy.py --student_arch resnet34 --student_input_size 64 --data kitti
# python time_energy.py --student_arch resnet34 --student_input_size 32 --data kitti
# python time_energy.py --student_arch resnet34 --student_input_size 64 --global_info --data kitti
# python time_energy.py --student_arch resnet34 --student_input_size 32 --global_info --data kitti
