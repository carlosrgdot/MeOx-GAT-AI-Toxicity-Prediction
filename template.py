import os,sys
from pathlib import Path

project_name = 'MeOx'

files = [
    '.github/workflows/gitkeep',
    f'{project_name}/__init__.py',
    f'{project_name}/components/__init__.py',
    f'{project_name}/utils/__init__.py',
    f'{project_name}/utils/common.py',
    f'{project_name}/logging/__init__.py',
    f'{project_name}/exception/__init__.py',
    f'{project_name}/entity/__init__.py',
    f'{project_name}/config/__init__.py',
    f'{project_name}/cloud/__init__.py',
    f'{project_name}/constant/__init__.py',
    'main.py',
    'setup.py',
    'app.py',
    'Dockerfile',
    'requirements.txt',
]


for filepath in files:
    filepath=Path(filepath)
    filedir,filename=os.path.split(filepath)
    if filedir!="":
        os.makedirs(filedir,exist_ok=True)

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath)==0):
        with open (filepath,'w') as f:
            pass

    else:
        print(f'{filepath} is empty')