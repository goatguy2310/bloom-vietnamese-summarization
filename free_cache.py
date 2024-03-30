import os
def deletebuffer():
    cmd = "sudo sh -c 'echo 1 >  /proc/sys/vm/drop_caches'"
    os.system(cmd)
deletebuffer()