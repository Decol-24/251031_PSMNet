import subprocess


cmd = [
    "python", "Test_img.py",
    "--loadmodel", "pretrained_sceneflow_new.tar",
    "--leftimg", "./0054_left.png",
    "--rightimg", "./0054_right.png",
]
subprocess.run(cmd, check=True)