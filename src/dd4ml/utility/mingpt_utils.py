import json
import os
import random
import sys
from ast import literal_eval

import numpy as np
import torch


# -----------------------------------------------------------------------------
# From minGPT utils
# -----------------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def setup_logging(config):
    """ monotonous bookkeeping """
    work_dir = config.system.work_dir
    # create the work directory if it doesn't already exist
    os.makedirs(work_dir, exist_ok=True)
    # log the args (if any)
    with open(os.path.join(work_dir, 'args.txt'), 'w') as f:
        f.write(' '.join(sys.argv))
    # log the config itself
    with open(os.path.join(work_dir, 'config.json'), 'w') as f:
        dict_config = config.to_dict()
        f.write(json.dumps(config.to_dict(), indent=4))

class CfgNode:
    """ a lightweight configuration class inspired by yacs """
    # TODO: convert to subclass from a dict like in yacs?
    # TODO: implement freezing to prevent shooting of own foot
    # TODO: additional existence/override checks when reading/writing params?

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __str__(self):
        return self._str_helper(0)

    def _str_helper(self, indent):
        """ need to have a helper to support nested indentation for pretty printing """
        parts = []
        for k, v in self.__dict__.items():
            if isinstance(v, CfgNode):
                parts.append("%s:\n" % k)
                parts.append(v._str_helper(indent + 1))
            else:
                parts.append("%s: %s\n" % (k, v))
        parts = [' ' * (indent * 4) + p for p in parts]
        return "".join(parts)

    def to_dict(self):
        """Return a dict representation of the config with JSON-serializable values."""
        result = {}

        for k, v in self.__dict__.items():
            if isinstance(v, CfgNode):
                # Recursively call to_dict for nested CfgNode instances
                result[k] = v.to_dict()
            elif isinstance(v, type):  
                # Convert class references to string
                result[k] = f"{v.__module__}.{v.__name__}"
            elif v is None:
                # Replace None with a JSON-serializable equivalent
                result[k] = "null"
            elif v is Ellipsis:  
                # Replace ellipses (...) with a placeholder string
                result[k] = "..."
            else:
                # Keep other values as is
                result[k] = v

        return result


    def merge_from_dict(self, d):
        self.__dict__.update(d)

    def merge_from_args(self, args):
        """
        update the configuration from a list of strings that is expected
        to come from the command line, i.e. sys.argv[1:].

        The arguments are expected to be in the form of `--arg=value`, and
        the arg can use . to denote nested sub-attributes. Example:

        --model.n_layer=10 --trainer.batch_size=32
        """
        for arg in args:

            keyval = arg.split('=')
            assert len(keyval) == 2, "expecting each override arg to be of form --arg=value, got %s" % arg
            key, val = keyval # unpack

            # first translate val into a python object
            try:
                val = literal_eval(val)
                """
                need some explanation here.
                - if val is simply a string, literal_eval will throw a ValueError
                - if val represents a thing (like an 3, 3.14, [1,2,3], False, None, etc.) it will get created
                """
            except ValueError:
                pass

            # find the appropriate object to insert the attribute into
            assert key[:2] == '--'
            key = key[2:] # strip the '--'
            keys = key.split('.')
            obj = self
            for k in keys[:-1]:
                obj = getattr(obj, k)
            leaf_key = keys[-1]

            # ensure that this attribute exists
            assert hasattr(obj, leaf_key), f"{key} is not an attribute that exists in the config"

            # overwrite the attribute
            print("command line overwriting config attribute %s with %s" % (key, val))
            setattr(obj, leaf_key, val)

    def merge_and_cleanup(self, keys_to_look=["system", "data", "model", "trainer"]):
        """
        Overwrite subdictionary values with corresponding upper-level values 
        and remove redundant upper-level keys.
        """
        
        # Get dictionary keys
        keys = list(self.__dict__.keys())

        # Keys other than keys_to_look
        other_keys = [k for k in keys if k not in keys_to_look]

        # Merge and cleanup
        for ok in other_keys:
            delete_ok = False
            for k in keys_to_look:
                # Check if ok is a key in self.__dict__[k]
                if ok in self.__dict__[k].__dict__:
                    # Overwrite the value
                    self.__dict__[k].__dict__[ok] = self.__dict__[ok]
                    delete_ok = True
            # Remove the key
            if delete_ok: del self.__dict__[ok]