from glob import glob

from setuptools import find_packages, setup

package_name = 'navigation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/rviz', glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sid',
    maintainer_email='siddarth.dayasagar@gmail.com',
    description='HANUMAN navigation: HiRISE DEM geometric cost map + A* global planner.',
    license='MIT',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'global_planner_node = navigation.global_planner_node:main',
            'mppi_node = navigation.mppi_node:main',
            'dashboard_qt = navigation.dashboard_qt_node:main',
            'dashboard_mock = navigation.mock_publisher:main',
        ],
    },
)
