# Warning control
import warnings

warnings.filterwarnings('ignore')

from unstructured_client import UnstructuredClient
from unstructured_client.models import shared
from unstructured_client.models.errors import SDKError
from unstructured.partition.html import partition_html
from unstructured.partition.pdf import partition_pdf
from unstructured.staging.base import dict_to_elements
from Utils import Utils

utils = Utils()

DLAI_API_KEY = utils.get_dlai_api_key()
DLAI_API_URL = utils.get_dlai_url()

s = UnstructuredClient(
    api_key_auth=DLAI_API_KEY,
    server_url=DLAI_API_URL,
)

filename = "example_files/el_nino.html"
html_elements = partition_html(filename=filename)

for element in html_elements[:10]:
    print(f"{element.category.upper()}: {element.text}")

# Process the Document with Document Layout Detection

filename = "example_files/el_nino.pdf"
pdf_elements = partition_pdf(filename=filename, strategy="fast")

# use the api
for element in pdf_elements[:10]:
    print(f"{element.category.upper()}: {element.text}")

with open(filename, "rb") as f:
    files = shared.Files(
        content=f.read(),
        file_name=filename,
    )

req = shared.PartitionParameters(
    files=files,
    strategy="hi_res",
    hi_res_model_name="yolox",
)

try:
    resp = s.general.partition(req)
    dld_elements = dict_to_elements(resp.elements)
except SDKError as e:
    print(e)

for element in dld_elements[:10]:
    print(f"{element.category.upper()}: {element.text}")

import collections

print(len(html_elements))

html_categories = [el.category for el in html_elements]
collections.Counter(html_categories).most_common()

print(len(dld_elements))

dld_categories = [el.category for el in dld_elements]
collections.Counter(dld_categories).most_common()
