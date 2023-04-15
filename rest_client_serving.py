import json
import requests
import sys
import numpy as np
from argparse import ArgumentParser
from Dev.utils import load_test_data
import os
import matplotlib.pyplot as plt


NUM_CLASSES = 4
IMAGE_SIZE = 128

def get_model_endpoint(model_name,host='127.0.0.1',
                       port='8501',
                       version=None,
                       verb="predict"):
  url = "http://{host}:{port}/v1/models/{model_name}"\
        .format(host=host, 
                port=port, 
                model_name=model_name)
  if version:
    url += 'versions/{version}'.format(version=version)
  url += ':{verb}'.format(verb=verb)
  return url


def get_model_prediction(model_input, model_name='mcpnet',
                         signature_name='serving_default'):
    """
    Args:
      model_input: numpy array, containing a batch of inputs
                    in the following format: (nb_samples,128,128,1)
    """
    
    url = get_model_endpoint(model_name)
    
    data = json.dumps({"signature_name": "serving_default",
                       "instances": model_input.tolist()})
    headers = {"content-type": "application/json"}
    
    response = requests.post(url, data=data, headers=headers)
    if response.status_code != requests.codes.ok:
        response.raise_for_status()
    
    nb_samples = model_input.shape[0]
    predictions = json.loads(response.text)['predictions']
    predictions = np.array(predictions).reshape(
                                            (nb_samples,
                                            IMAGE_SIZE,
                                            IMAGE_SIZE,
                                            NUM_CLASSES))
    predictions = np.argmax(predictions,axis=-1)
    return predictions

def display_image(idx, title,input_image,prediction):
  plt.figure(figsize=(8,8))

  # Input image
  plt.subplot(1,2,1)
  plt.axis('off')
  plt.imshow(input_image[idx,:,:,0])
  plt.title('\n\n{}'.format(title), fontdict={'size': 12})

  # Prediction
  plt.subplot(1,2,2)
  plt.axis('off')
  plt.imshow(prediction[idx,:,:])  
  plt.title('\n\n{}'.format(title), fontdict={'size': 12})
  plt.show()


if __name__ == '__main__':

    print("Generate REST model endpoint ...")
    url = get_model_endpoint(model_name='mcpnet')
    print(url)
    
    parser = ArgumentParser(
                    prog='CardiacMRISegmentation',
                    description='Automatically segments \
                              LV,RV and MYO in Cine-MRIs')
    
    parser.add_argument('nb_samples',
                        type=int,
                        help="Specifies the number\
                          of test images to load")
    args = parser.parse_args()

    nb_samples = int(args.nb_samples)
    data_path =os.path.dirname(__file__)+ "/Test/data/"
    test_data,files,_,_ = load_test_data(data_path,nb_samples)

    for idx,model_input in enumerate(test_data):
        model_prediction = get_model_prediction(model_input)
        display_image(3,f'{files[idx]}',model_input,model_prediction)
