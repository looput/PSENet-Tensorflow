#encoding = utf-8

def cmd(cmd):
    try:
        import  commands
        return commands.getoutput(cmd)
    except:
        import subprocess
        return subprocess.check_output(cmd, shell = True)
#         return os.system(cmd)
