# IC50evaluation
An evaluation of drug responsiveness prediction models for cell viability inhibitory concentrations (IC50 values)

# Usage
- python codes for each scenarios are stored in Model_generation_validation
- All the dataset of each scenarios could be downloaded from [here (IC50evaluation/Dataset)](https://mega.nz/#F!CeYGDKyS!uqkmWJ4E2XSGJp_C2VO2gg)
- you should download each dataset and change the path in each code (path for dataset, model_output_folder, result_output_folder, etc)
- please refer to the annotations in each code


# Code description
- *.ipynb : the files construct prediction model and show validation result
- *.h5, *.json : these files are constructed model architecture and model weight from AI-method (CNN and ResNet)
- *.pkl : these files are constructed model from ML-method (linearSVR, random forest, XGBoost, etc)
- Computer specification with model generation : Windows 10, Keras 2.3.0, tensorflow-gpu 1.14.0, Geforce GTX 1080 Ti, Titan RTX, RAM 64GB

# Contact
### If you have any questions, please contact below.
- Mr. Aron Park (parkar13@gmail.com)
- Prof. Seungyoon Nam (nams@gachon.ac.kr)
