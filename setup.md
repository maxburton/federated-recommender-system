#How to set up the project
*Install Python3 and make sure it's added to your PATH
*Install anaconda3 (and add it to PATH)


## Automatic Installation (with pycharm) (recommended):
*Add the anaconda3 interpreter to pycharm
*Go to terminal settings and change shell path to: cmd.exe "/K" C:\ProgramData\Anaconda3\Scripts\activate.bat C:\ProgramData\Anaconda3
NOTE: Anaconda3 path may be different depending on installation, check your Anaconda3 path first
*Install existing conda env from environment.yml:
`conda env create -f environment.yml`
`conda activate FRS-conda`

These steps should work on other IDEs too

## Manual Installation:
*Create a new conda env:
`conda create --name FRS-conda`
`conda activate FRS-conda`
*Install numpy on conda
*Install pandas on conda
*Install pytorch on conda
*Install h5py on conda
*Install scikit-learn on conda
*Install matplotlib on conda
`conda install -c conda-forge fuzzywuzzy`
`conda install -c conda-forge scikit-surprise`
`conda install -c conda-forge lightfm`
*clone the spotlight repo: https://github.com/maciejkula/spotlight.git
*Install it in your conda env using: 
`python setup.py build`
`python setup.py install`

## Troubleshooting:
*If you get a pip install error, you may need to install VS Build Tools: http://go.microsoft.com/fwlink/?LinkId=691126&fixForIE=.exe.
