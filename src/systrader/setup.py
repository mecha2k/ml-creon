from setuptools import find_namespace_packages, setup

setup(
    name='systrader',
    version='1.0',
    description='System trading for Quantylab',
    author='Quantylab',
    author_email='quantylab@gmail.com',
    url='https://github.com/quantylab/systrader',
    # packages=find_packages(exclude=['quantylab', 'systrader']),
    packages=find_namespace_packages(include=['quantylab.*']),
    install_requires=[
        'django', 'pywinauto'
    ]
)
