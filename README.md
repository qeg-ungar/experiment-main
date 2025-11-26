# experiment-main
NV experiment control code with qudi and qm-qua

---

# Development Setup
This repo works alongside two sibling repositories:
* [qudi-iqo-modules](https://github.com/qeg-ungar/qudi-iqo-modules) ([fork of the Ulm-IQO/qudi-iqo-modules repository](https://github.com/Ulm-IQO/qudi-iqo-modules))
* [qua-libs](https://github.com/qeg-ungar/qua-libs) ([fork of the qua-platform/qua-libs repository](https://github.com/qua-platform/qua-libs))

All three share a single **Python 3.10** virtual environment. Fork both repositories along with this one before proceding to the setup.

## Clone the Repositories
Clone all three repos into the **same parent folder**:

```bash
cd ~/parent_folder

git clone https://github.com/your_user/experiment-main # your forked repo
git clone https://github.com/your_user/qudi-iqo-modules # your forked repo
git clone https://github.com/your_user/qua-libs # your forked repo
```

Your folder structure should look like:
```
parent_folder/
- experiment-main/
- qudi-iqo-modules/
- qua-libs/
```

## Create the Python 3.10 Virtual Environment
Go into `experiment-main`:

```bash
cd experiment-main
python -m venv .venv   # create venv
.\.venv\Scripts\activate  # Windows CMD
python --version  # should show Python 3.10.x
pip install -r requirements.txt # installs local qudi-iqo-modules as a package, qudi-core, and qua-libs requirements
```
## Set up qudi config path
Running the command `qudi` will load an empty Qudi manager if qudi has not been previously installed, or will load the most recent config file.  We have copied the default config (from `qudi-iqo-modules\src\qudi\default.cfg`) into `experiment-main\qudi_configs\`. The default path for loading a new config file is `C:\Users\<USER>\qudi\config`. To load a new config from the local directory `experiment-main\qudi_configs\` Start qudi, and then load (via File -> Load configuration) and select the local config file.

