import setuptools

INSTALL_REQUIRES = [
    'numpy>=1.20.1',
    'jax>=0.2.16',
    'jaxlib>=0.1.61',
    'jax-md>=0.1.12',
    'scikit-learn>=0.24.2',
    'plum-dispatch==1.5.6',
]

setuptools.setup(
    name="gdml-jax",
    version="0.2.0",
    author="Niklas Schmitz",
    author_email="n.schmitz@tu-berlin.de",
    install_requires=INSTALL_REQUIRES,
    description="Automatic Differentiation of Gaussian Processes for learning Molecular Force Fields, powered by jax",
    url="https://github.com/niklasschmitz/gdml-jax",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
