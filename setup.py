import setuptools

from pathlib import Path
from os import scandir

def scan_packages(path, prefix=None):
    if prefix is None:
        prefix = ''

    for entry in scandir(path):
        if entry.is_dir(follow_symlinks=False) and \
           entry.name[0] != '.' and \
           entry.name not in ['__pycache__']:
            yield prefix+entry.name
            yield from scan_packages(entry.path, prefix= prefix+entry.name+'.')


setuptools.setup(
    name="litenn-apps",
    version="2020.11.83",
    author="iperov",
    author_email="lepersorium@gmail.com",
    description="",
    install_requires=['numpy','litenn','opencv-python'],
    include_package_data=True,
    long_description="Ready to work ML models using LiteNN.",
    url="https://github.com/iperov/litenn-apps",
    packages=['.', *scan_packages('.')],
    license = 'MIT',
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Environment :: GPU",
        "Topic :: Scientific/Engineering",
    ],
    python_requires='>=3.6',
)