#!/bin/bash

echo "======================================================="

export PYTHONPATH=${CM_ML_MODEL_CODE_WITH_PATH}:${PYTHONPATH}
      
${CM_PYTHON_BIN_WITH_PATH} ${CM_TMP_CURRENT_SCRIPT_PATH}/src/run_model.py \
  --ckpt "${CM_ML_MODEL_FILE_WITH_PATH}" \
  --dataset ${CM_ABTF_DATASET} \
  ${CM_ABTF_EXTRA_CMD}
test $? -eq 0 || exit $?
