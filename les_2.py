import warnings
import json
from unstructured_client import UnstructuredClient
from unstructured_client.models import shared
from unstructured_client.models.errors import SDKError
from unstructured.partition.html import partition_html
from unstructured.partition.pptx import partition_pptx
from unstructured.staging.base import dict_to_elements, elements_to_json
from Utils import Utils
from IPython.display import JSON
from IPython.display import Image

warnings.filterwarnings('ignore')

# download nltk punkt
# import os
# import nltk
# import certifi
# os.environ['SSL_CERT_FILE'] = certifi.where()
# nltk.download('punkt')

utils = Utils()

DLAI_API_KEY = utils.get_dlai_api_key()
DLAI_API_URL = utils.get_dlai_url()

s = UnstructuredClient(
    api_key_auth=DLAI_API_KEY,
    server_url=DLAI_API_URL,
)

# Example Document: Medium Blog HTML Page

Image(filename="images/HTML_demo.png", height=600, width=600)
filename = "example_files/medium_blog.html"
elements = partition_html(filename=filename)
element_dict = [el.to_dict() for el in elements]
example_output = json.dumps(element_dict, indent=2)
print("The example output is", example_output)

# Example Doc: MSFT PowerPoint on OpenAI

Image(filename="images/pptx_slide.png", height=600, width=600)
filename = "example_files/msft_openai.pptx"
elements = partition_pptx(filename=filename)
element_dict = [el.to_dict() for el in elements]
output = json.dumps(element_dict[:], indent=2)
print("The pptx output is", output)

# Example Document: PDF on Chain-of-Thought
Image(filename="images/cot_paper.png", height=600, width=600)
filename = "example_files/CoT.pdf"
with open(filename, "rb") as f:
    files = shared.Files(
        content=f.read(),
        file_name=filename,
    )

req = shared.PartitionParameters(
    files=files,
    strategy='hi_res',
    pdf_infer_table_structure=True,
    languages=["eng"],
)
try:
    resp = s.general.partition(req)
    print(json.dumps(resp.elements[:3], indent=2))
except SDKError as e:
    print(e)

# output = json.dumps(resp.elements, indent=2)
# print("The .pdf output is", output)
