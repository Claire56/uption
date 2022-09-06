# #!/usr/bin/env python
# # -*- coding: utf-8 -*-

# from pathlib import Path

# from setuptools import find_packages, setup

# # Package meta-data.
# NAME = 'startup-classifier'
# DESCRIPTION = "Classifier model for startups from Train In Data."
# URL = "https://github.com/claire56/uption"
# EMAIL = "kclamu26@gmail.com"
# AUTHOR = "Zimba- Claire MK"
# REQUIRES_PYTHON = ">=3.6.0"


# # What packages are required for this module to be executed?
# def list_reqs(fname="requirements.txt"):
#     with open(fname) as fd:
#         return fd.read().splitlines()


# long_description = DESCRIPTION

# # Load the package's VERSION file as a dictionary.
# about = {}
# ROOT_DIR = Path(__file__).resolve().parent
# PACKAGE_DIR = ROOT_DIR / 'src'
# with open(PACKAGE_DIR / "VERSION") as f:
#     _version = f.read().strip()
#     about["__version__"] = _version


# # Where the magic happens:
# setup(
#     name=NAME,
#     version=about["__version__"],
#     description=DESCRIPTION,
#     long_description=long_description,
#     long_description_content_type="text/markdown",
#     author=AUTHOR,
#     author_email=EMAIL,
#     python_requires=REQUIRES_PYTHON,
#     url=URL,
#     packages=find_packages(exclude=("tests",)),
#     package_data={"gb_classifier": ["VERSION"]},
#     install_requires=list_reqs(),
#     extras_require={},
#     include_package_data=True,
#     license="BSD-3",
#     classifiers=[
#         # Trove classifiers
#         # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
#         "License :: OSI Approved :: MIT License",
#         "Programming Language :: Python",
#         "Programming Language :: Python :: 3",
#         "Programming Language :: Python :: 3.6",
#         "Programming Language :: Python :: 3.7",
#         "Programming Language :: Python :: 3.8",
#         "Programming Language :: Python :: Implementation :: CPython",
#         "Programming Language :: Python :: Implementation :: PyPy",
#     ],
# )
