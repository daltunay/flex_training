# Adapted from:
# https://github.com/mlcommons/training_results_v4.0/blob/main/Oracle/benchmarks/llama2_70b_lora/implementations/BM.GPU.H100.8/Dockerfile

FROM nvcr.io/nvidia/pytorch:24.04-py3
# nvcr.io/nvdlfwea is used in the original Dockerfile
# access is denied, so we use image from nvcr.io/nvidia instead

WORKDIR /workspace/ft-llm

RUN git config --global user.email "you@example.com" && \
    git config --global user.name "Your Name"
# required for below git commands

## NeMo
ARG NEMO_REVISION=r2.0.0.rc0.beta
ARG NEMO_REVISION_FIX=r2.0.0.rc0.beta.a2a_tp_fix

RUN git clone --single-branch --branch ${NEMO_REVISION} https://github.com/NVIDIA/NeMo.git && \
    git -C NeMo remote add anmolgupt https://github.com/anmolgupt/NeMo.git && \
    git -C NeMo fetch anmolgupt && \
    git -C NeMo merge --no-edit anmolgupt/anmolgupt/${NEMO_REVISION_FIX} && \
    pip install --no-build-isolation -e ./NeMo[nlp]

## Megatron-LM
ARG MEGATRON_REVISION=core_r0.7.0.beta

RUN pip uninstall -y megatron-core && \
    git clone --depth 1 --single-branch --branch ${MEGATRON_REVISION} https://github.com/NVIDIA/Megatron-LM.git && \
    pip install ./Megatron-LM && \
    make -C Megatron-LM/megatron/core/datasets

ENV PYTHONPATH="${PYTHONPATH}:/workspace/ft-llm/megatron-lm"

## TransformerEngine
ARG TE_REVISION=v1.6rc2
ARG CUDNN_FRONTEND_REVISION=1b0b5eac540b7f8fd19b18f1e6b8427c95503348
ARG GTEST_REVISION=f8d7d77c06936315286eb55f8de22cd23c188571

RUN git clone --depth 1 --single-branch --branch ${TE_REVISION} https://github.com/NVIDIA/TransformerEngine.git && \
    git -C TransformerEngine submodule init && git -C TransformerEngine submodule update && \
    git -C TransformerEngine/3rdparty/cudnn-frontend checkout ${CUDNN_FRONTEND_REVISION} && \
    git -C TransformerEngine/3rdparty/googletest checkout ${GTEST_REVISION} && \
    NVTE_FRAMEWORK=pytorch NVTE_WITH_USERBUFFERS=1 MPI_HOME=/usr/local/mpi pip install --force-reinstall --no-deps ./TransformerEngine
# custom cudnn-frontend and googletest revisions were not used in the original Dockerfile
# cf. https://github.com/mlcommons/training_results_v4.0/issues/5#issuecomment-2360615866

ADD . /workspace/ft-llm
RUN rm -rf NeMo && mv libraries/NeMo NeMo && \
    rm -rf Megatron && mv libraries/Megatron-LM Megatron-LM && \
    rm -rf TransformerEngine && mv libraries/TransformerEngine TransformerEngine && \
    rm -rf libraries
# the NeMo, Megatron-LM, and TransformerEngine repositories are copied from Oracle libraries folder
# this overrides the above cloned repositories with the local files instead
# this seems weird, but using the cloned repositories instead causes errors during training

RUN pip install -r requirements.txt
# requirements.txt contains some fixes for huggingface-hub and transformers libraries
# cf. https://github.com/NVIDIA/NeMo/issues/9793, https://github.com/NVIDIA/NeMo/issues/9272