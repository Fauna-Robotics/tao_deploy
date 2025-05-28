# syntax=docker/dockerfile:1.11-labs
ARG BASE_IMAGE=faunarobot/ros2-torch:jetson
FROM ${BASE_IMAGE}

# NVIDIA-related environment variables
ENV NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics


WORKDIR /workspace/third_party

RUN git clone https://github.com/Fauna-Robotics/tao_deploy.git -b shruthi/peoplenet_reid_inferencer

# This uses an "intelligent" installer to handle versions
RUN pip3 install -r tao_deploy/docker/requirements-l4t.txt
RUN cd tao_deploy && python3 setup_l4t.py develop && cd ../
RUN apt update && apt install -y python3-opencv libatlas3-base

WORKDIR /workspace/fauna

# Copy project sources
COPY --parents \
  projects/human-detection \
  ./

# Install Python dependencies
ENV POETRY_VIRTUALENVS_PATH=/root/.cache/poetry-virtualenvs
RUN --mount=type=cache,target=/root/.cache/pypoetry \
  cd projects/human-detection && poetry lock && poetry install --sync --without dev --without desktop

WORKDIR /workspace/fauna/projects/human-detection
CMD ["bash"]
