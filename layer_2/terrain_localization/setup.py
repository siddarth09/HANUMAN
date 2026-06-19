from glob import glob

from setuptools import find_packages, setup

package_name = 'terrain_localization'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/rviz', glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sid',
    maintainer_email='siddarth.dayasagar@gmail.com',
    description='HANUMAN HiRISE orbital prior: terrain-relative localization.',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'terrain_matcher_node = terrain_localization.terrain_matcher_node:main',
        ],
    },
)
