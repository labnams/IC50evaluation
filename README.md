# IC50evaluation
An evaluation of drug responsiveness prediction models for cell viability inhibitory concentrations (IC50 values)

# Usage
- Jupyter notebook files of python coding for each scenarios are stored in Model_generation_validation
- All the dataset and pre-defined models of each scenarios could be downloaded from [here (IC50evaluation/Dataset, IC50evaluation/Pre-defined_models)](https://mega.nz/#F!CeYGDKyS!uqkmWJ4E2XSGJp_C2VO2gg)
- you should download each dataset and change the path in each code (path for dataset, model_output_folder, result_output_folder, etc)
- please read Code_and_Data_description_IC50evaluation.pdf to understand data repository and code repository.
- please refer to the annotations in each code


# Code description
- *.ipynb : the jupyter notebook files (python 3) to construct prediction model and show validation result
- *.h5, *.json : these files are constructed model architecture and model weight from AI-method (CNN and ResNet)
- *.pkl : these files are constructed model from ML-method (linearSVR, random forest, XGBoost, etc)

# Computer specification
- all models were constructed under the computer specification below.
- Windows 10
- Python 3
- Keras 2.3.0
- tensorflow-gpu 1.14.0
- Geforce GTX 1080 Ti 11GB and Titan RTX 24GB
- RAM 64GB

# Contact
### If you have any questions, please contact below.
- Mr. Aron Park (parkar13@gmail.com)
- Prof. Seungyoon Nam (nams@gachon.ac.kr)
