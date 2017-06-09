from setuptools import setup

setup(
    name='picky_robot',
    version='0.0.0',
    packages=[],
    py_modules=[
        'nodes.picky_robot',
        'nodes.ur5_pusher'],
    install_requires=['setuptools'],
    author='Adam Allevato',
    author_email='adam.d.allevato@gmail.com',
    maintainer='Adam Allevato',
    maintainer_email='adam.d.allevato@gmail.com',
    keywords=['ROS'],
    classifiers=[],
    description=('Uses the UR5 robot to do linear motions ("pushes")'
        ' at specified X-positions. Designed for extremely basic manipulation '
        ' demos.'),
    license='Apache License, Version 2.0',
    entry_points={
        'console_scripts': [
            'picky_robot = nodes.picky_robot:main',
            'ur5_pusher = nodes.ur5_pusher:main',
        ],
    },
)
