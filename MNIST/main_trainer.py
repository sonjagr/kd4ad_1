import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import subprocess

# Run the other script
cmd = "python3 3_train_student.py --size 84 --digit 0"
process = subprocess.Popen(cmd, shell=True)
process.wait()
print('finished')

cmd = "python3 3_train_student.py --size 84 --digit 1"
process = subprocess.Popen(cmd, shell=True)
process.wait()
print('finished')

cmd = "python3 3_train_student.py --size 84 --digit 2"
process = subprocess.Popen(cmd, shell=True)
process.wait()
print('finished')

cmd = "python3 3_train_student.py --size 84 --digit 3"
process = subprocess.Popen(cmd, shell=True)
process.wait()
print('finished')

cmd = "python3 3_train_student.py --size 84 --digit 4"
process = subprocess.Popen(cmd, shell=True)
process.wait()
print('finished')

cmd = "python3 3_train_student.py --size 84 --digit 5"
process = subprocess.Popen(cmd, shell=True)
process.wait()
print('finished')

cmd = "python3 3_train_student.py --size 84 --digit 6"
process = subprocess.Popen(cmd, shell=True)
process.wait()
print('finished')

cmd = "python3 3_train_student.py --size 84 --digit 7"
process = subprocess.Popen(cmd, shell=True)
process.wait()
print('finished')
cmd = "python3 3_train_student.py --size 84 --digit 8"
process = subprocess.Popen(cmd, shell=True)
process.wait()
print('finished')

cmd = "python3 3_train_student.py --size 84 --digit 9"
process = subprocess.Popen(cmd, shell=True)
process.wait()
print('finished')