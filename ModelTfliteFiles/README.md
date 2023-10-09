Model naming conventions
========================
The model naming convention is done to make it easier to identify the model and its properties.
The following attributes must be present in the model name:
- Date (DDMM)
- Whether the model is Convoluted or not (basic or conv)
- Whether the model is trained on top of an existing model or not (top(#) or notop)
- Which datasets were used to train the model, using dataset ID. UNK is used for unknown datasets.
- The first 4 decimals of the models loss
- extra information (if needed)

For example:
A basic model trained from scratch with datasets V0, V1 and V2
- 0910_basic_notop_V0V1V2_0.578.tflite

