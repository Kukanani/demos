from setuptools import setup

setup(
    name='picky_robot',
    version='0.0.0',
    packages=[],
    py_modules=[
        'nodes.picky_robot',
        'nodes.ur5_motion'],
    install_requires=['setuptools'],
    author='Adam Allevato',
    author_email='adam.d.allevato@gmail.com',
    maintainer='Adam Allevato',
    maintainer_email='adam.d.allevato@gmail.com',
    keywords=['ROS'],
    classifiers=[],
    description=(''),
    license='Apache License, Version 2.0',
    entry_points={
        'console_scripts': [
            'picky_robot = nodes.picky_robot:main',
            'ur5_motion = nodes.ur5_motion:main',
        ],
    },
)
