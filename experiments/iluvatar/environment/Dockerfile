# Reproducible derivative of the environment currently used on ub39.
# Build after pushing the desired source commits to their repositories.
FROM harbor.baai.ac.cn/plugin/iluvatar-corex4.5.0-flagtree0.6.0-triton3.6.0-cxnone-vllm_fl0.24.0@sha256:cc8ed9969977a06996c86c483851567f212ab9761bb9147e85082b4a34ce88b8

ARG VLLM_PLUGIN_REF=13eb9be69ecc5b5ca4f79c44e9ee40081eaa1bf0
ARG FLAGGEMS_REF=a7620cc191a0b42e040194622c5758b22a7a25dc

RUN git clone https://github.com/wyNomad/vllm-plugin-FL.git /opt/flagos/vllm-plugin-FL \
    && git -C /opt/flagos/vllm-plugin-FL checkout --detach "${VLLM_PLUGIN_REF}" \
    && git clone https://github.com/flagos-ai/FlagGems.git /opt/flagos/FlagGems \
    && git -C /opt/flagos/FlagGems checkout --detach "${FLAGGEMS_REF}"

ENV VLLM_PLUGINS=fl \
    PYTHONPATH=/opt/flagos/vllm-plugin-FL:/opt/flagos/FlagGems/src

RUN python3 -c 'import importlib.util as u; assert u.find_spec("vllm_fl").origin.startswith("/opt/flagos/vllm-plugin-FL/"); assert u.find_spec("flag_gems").origin.startswith("/opt/flagos/FlagGems/src/")'

WORKDIR /workspace
