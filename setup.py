from distutils.core import setup

setup(
      name='comgra',
      version='0.1.0',
      description='',
      url='https://github.com/FlorianDietz/comgra',
      author='Florian Dietz',
      author_email='floriandietz44@gmail.com',
      packages=['comgra'],
      install_requires=[
            'dash==2.6.2',
            'dash-svg==0.0.12',
            'msgpack==1.0.4',
            'numpy==1.24.2',
            'pandas==1.5.3',
      ],
)
