# Using a model from the model hub is very version version specific.
# Make sure you have sagemaker version sagemaker==2.48.0 Earlier versions won't work and 2.53 doesn't work.


# Resources
# https://sagemaker.readthedocs.io/en/stable/
# https://github.com/huggingface/notebooks/blob/master/sagemaker/11_deploy_model_from_hf_hub/deploy_transformer_model_from_hf_hub.ipynb
# https://huggingface.co/blog/deploy-hugging-face-models-easily-with-amazon-sagemaker

from sagemaker.huggingface.model import HuggingFaceModel #Need .model
import sagemaker

role = sagemaker.get_execution_role()

# Hub Model configuration. https://huggingface.co/models
hub = {
  'HF_MODEL_ID':'distilbert-base-uncased-distilled-squad', # model_id from hf.co/models
  'HF_TASK':'question-answering' # NLP task you want to use for predictions
}

# create Hugging Face Model Class
huggingface_model = HuggingFaceModel(
   env=hub,
   role=role, # iam role with permissions to create an Endpoint
   transformers_version="4.6", # transformers version used
   pytorch_version="1.7", # pytorch version used
   py_version="py36", # python version of the DLC
)

# deploy model to SageMaker Inference
predictor = huggingface_model.deploy(
   initial_instance_count=1,
   instance_type="ml.m5.xlarge"
)


# example request, you always need to define "inputs"
data = {
"inputs": {
    "question": "What is used for inference?",
    "context": "My Name is Philipp and I live in Nuremberg. This model is used with sagemaker for inference."
    }
}

# request
predictor.predict(data)


breakpoint()
# delete endpoint
predictor.delete_endpoint()