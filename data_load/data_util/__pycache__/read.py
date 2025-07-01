import dis
import marshal

#
pyc_path = '/home/SHIH0020/robustlearn/DI2SDiff/data_load/data_util/__pycache__/raw_aug_loader.cpython-39.pyc'

# 输出
output_path = '/home/SHIH0020/robustlearn/DI2SDiff/data_load/data_util/raw_aug_loader_disassembled.py'

# 
with open(pyc_path, 'rb') as f:
    f.read(16)  #
    code = marshal.load(f)
    disassembled_code = dis.code_info(code) + '\n\n' + dis.Bytecode(code).dis()

# 
with open(output_path, 'w') as out_file:
    out_file.write("# This is disassembled bytecode, not original source code\n")
    out_file.write("# pyc file: {}\n\n".format(pyc_path))
    out_file.write(str(disassembled_code))

print(f"Disassembled code written to {output_path}")
