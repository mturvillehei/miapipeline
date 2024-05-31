### Model Imports

There are two steps to adding a new model to the pipeline:

   1) Adding a .py script, containing functions for the encoding, decoding, and generation steps. 
        (I'll add an optional arg to the cli call for kwaargs that can be passed to the model)
    2) Add the model to the MODEL_MAP and MODEL_TYPE dicts.


        