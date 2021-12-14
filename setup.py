from distutils.core import setup
setup(
  name = 'imbcobra',
  packages = ['imbcobra'],
  version = '2.1',
  license = 'MIT', 
  description = 'COBRA for classification tasks (on Imbalanced Data)',
  author = ['Dr. Arabin Kumar Dey', 'Vishisht Priyadarshi', 'Aadi Gupta', 'Tejus Singla', 'Shashank Goyal'],
  author_email = 'vishishtpriyadarshi867@gmail.com',
  url = 'https://github.com/vishishtpriyadarshi/MA691-COBRA-6',
  download_url = 'https://github.com/vishishtpriyadarshi/MA691-COBRA-6/archive/refs/tags/v2.1.tar.gz',
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
    'Programming Language :: Python :: 3.4',
  ],
)