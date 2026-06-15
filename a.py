import sys, inspect, transformers

print("python:", sys.executable)
print("transformers:", transformers.__version__)
print("transformers file:", transformers.__file__)

from transformers.masking_utils import create_causal_mask
print("create_causal_mask file:", inspect.getsourcefile(create_causal_mask))
print("create_causal_mask sig:", inspect.signature(create_causal_mask))