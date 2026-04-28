FROM tensorflow/tensorflow:2.16.1-gpu

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ARG SSH_PUBLIC_KEY=""

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    cmake \
    libopenmpi-dev \
    python3-dev \
    zlib1g-dev \
    swig \
    ffmpeg \
    openssh-client \
    openssh-server \
    tmux \
    ripgrep \
    less \
    vim \
    && rm -rf /var/lib/apt/lists/*

COPY workspace_ssh_sync.sh /usr/local/bin/workspace_ssh_sync.sh
COPY update_bash_prompt.sh /usr/local/bin/update_bash_prompt.sh
COPY sshd_entrypoint.sh /usr/local/bin/sshd_entrypoint.sh
COPY setup_project_venv.sh /usr/local/share/comp579/setup_project_venv.sh
RUN chmod +x /usr/local/bin/workspace_ssh_sync.sh /usr/local/bin/update_bash_prompt.sh /usr/local/bin/sshd_entrypoint.sh /usr/local/share/comp579/setup_project_venv.sh && \
    TARGET_HOME=/root /usr/local/bin/update_bash_prompt.sh && \
    TARGET_HOME=/etc/skel /usr/local/bin/update_bash_prompt.sh && \
    mkdir -p /root/.ssh && chmod 700 /root/.ssh && \
    touch /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys && \
    if [ -n "${SSH_PUBLIC_KEY}" ]; then printf '%s\n' "${SSH_PUBLIC_KEY}" >> /root/.ssh/authorized_keys; fi && \
    mkdir -p /run/sshd && \
    sed -ri 's/^#?PasswordAuthentication .*/PasswordAuthentication no/' /etc/ssh/sshd_config && \
    sed -ri 's/^#?PermitRootLogin .*/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -ri 's|^#?AuthorizedKeysFile .*|AuthorizedKeysFile .ssh/authorized_keys|' /etc/ssh/sshd_config && \
    grep -q '^PubkeyAuthentication yes' /etc/ssh/sshd_config || echo 'PubkeyAuthentication yes' >> /etc/ssh/sshd_config

WORKDIR /workspace

EXPOSE 22
ENTRYPOINT ["/usr/local/bin/sshd_entrypoint.sh"]
