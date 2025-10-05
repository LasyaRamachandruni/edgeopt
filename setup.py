from setuptools import setup, find_packages

setup(
    name="edgeopt",
    version="0.1.0",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'edgeopt=edgeopt.cli:main',
        ],
    },
    install_requires=[
        'torch',
        'torchvision', 
        'onnx',
        'onnxruntime',
    ],
)
