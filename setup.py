from setuptools import setup, find_packages
import os
import re
import codecs

here = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    with codecs.open(os.path.join(here, *parts), "r") as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as f:
    required = f.read().splitlines()

setup(
    version=find_version("largelookuputils", "__init__.py"),
    name="LargeLookupUtils",
    description="Utils for large lookups",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Sven Dorkenwald",
    author_email="sven.dorkenwald@alleninstitute.org",
    url="https://github.com/CAVEconnectome/LargeLookupUtils",
    packages=find_packages(),
    include_package_data=True,
    install_requires=required,
    setup_requires=["pytest-runner"],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
    ),
)
