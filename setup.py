from setuptools import find_packages, setup
import os

version = {}
with open(os.path.join("omniwatermask", "__version__.py")) as fp:
    exec(fp.read(), version)


setup(
    name="omniwatermask",
    version=version["__version__"],
    description="""Python library for water segmentation in high to moderate resolution remotely sensed imagery""",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Nick Wright",
    author_email="nicholas.wright@dpird.wa.gov.au",
    url="https://github.com/DPIRD-DMA/OmniWaterMask",
    python_requires=">=3.7",
    packages=find_packages(),
    install_requires=[
        "omnicloudmask>=1.0.7",
        "c2v>=4.9",
        "geopandas>=0.14.4",
        "osmnx>=2.0.0",
        "scipy>=1.10.0",
    ],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    package_data={"omniwatermask": ["models/model_download_links.csv"]},
)