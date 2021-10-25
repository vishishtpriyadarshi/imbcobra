from distutils.core import setup
setup(
  name = 'cobraclassifier',
  packages = ['cobraclassifier'],
  version = '0.1',
  license='MIT', 
  description = 'COBRA for classification tasks (on Imbalanced Data)',
  author = 'Vishisht Priyadarshi',
  author_email = 'vishishtpriyadarshi867@gmail.com',
  url = 'https://github.com/vishishtpriyadarshi/MA691-COBRA-6',
  download_url = 'https://github.com/vishishtpriyadarshi/MA691-COBRA-6/archive/refs/tags/v0.1.tar.gz',
  keywords = ['Classification', 'Imbalanced Data', 'Machine Learning'],
  install_requires=[
          'numpy',
          'scipy',
          'scikit-learn',
          'matplotlib',
          'seaborn',
          'pandas'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Science/Research',      
    'Topic :: Scientific/Engineering',
    'License :: OSI Approved :: MIT License',   
    'Programming Language :: Python :: 3',      
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)