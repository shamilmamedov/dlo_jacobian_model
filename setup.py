from setuptools import setup, find_packages

NAME = "dlo_jacobian_model"
VERSION = "0.0.1"
AUTHORS = ""
MAINTAINER_EMAIL = ""
DESCRIPTION = "Fast control of DLO with MPC"

setup(
    name=NAME,
    version=VERSION,
    author=AUTHORS,
    author_email=MAINTAINER_EMAIL,
    packages=find_packages()
    # packages=['training', 'FEIN']
)