import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="iPLAN",
    version="0.1.0",
    author="Xiyang Wu",
    author_email="wuxiyang1996@gmail.com",
    description="Codebase for iPLAN: Intent-Aware Planning in Heterogeneous Traffic via "
                "Distributed Multi-Agent Reinforcement Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wuxiyang1996/iPLAN",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)