from setuptools import find_packages,setup
from typing import List
hypen="-e ."
def get_req(file_path:str)->List[str]:
    req=[]
    with open(file_path) as obj:
        req=obj.readlines()
        req=[i.replace("\n"," ")for i in req]
        if hypen in req:
            req.remove(hypen)
    return req        


setup(
    name="Mlproject",
    version="0.0.1",
    author="Aditi Rao",
    packages=find_packages(),
    install_requires=get_req('requirements.txt')

)