"""Oracle Cloud Always Free deployer.

Generates deployment artifacts for Oracle Cloud ARM instances:
- cloud-init script for automated provisioning
- systemd service unit for persistent spore process
- setup script for dependencies

Oracle Always Free gives us:
- 4 ARM A1 instances (4 OCPU, 24 GB RAM total)
- 2 AMD micro instances (1/8 OCPU, 1 GB each)
- 200 GB block storage
- 10 TB/month egress

Each VM can run 20-50 lightweight spores as separate processes.
"""

from __future__ import annotations

import textwrap
from pathlib import Path


def generate_oracle_deployment(
    output_dir: str,
    spore_id: str,
    mesh_seeds: list[str],
    provider_keys: dict[str, str],
    spores_per_vm: int = 20,
    base_port: int = 8470,
) -> dict[str, str]:
    """Generate Oracle Cloud deployment files."""
    files = {}

    seeds_str = ",".join(mesh_seeds)
    env_exports = "\n".join(f'export {k}="{v}"' for k, v in provider_keys.items())

    # Cloud-init user data script
    files["cloud-init.yaml"] = textwrap.dedent(f"""\
        #cloud-config
        package_update: true
        packages:
          - python3
          - python3-pip
          - python3-venv
          - git

        write_files:
          - path: /opt/synapse/setup.sh
            permissions: '0755'
            content: |
              #!/bin/bash
              cd /opt/synapse
              python3 -m venv venv
              source venv/bin/activate
              pip install synapse-brain
              {env_exports}
              export SYNAPSE_MESH_SEEDS="{seeds_str}"

              # Launch multiple spores on sequential ports
              for i in $(seq 0 {spores_per_vm - 1}); do
                port=$(({{base_port}} + $i))
                spore_id="{spore_id}-${{i}}"
                export SYNAPSE_SPORE_ID="${{spore_id}}"
                export SYNAPSE_PORT="${{port}}"
                nohup python3 -m synapse_brain.spore.runtime \\
                  --port $port --id $spore_id \\
                  > /var/log/synapse/spore-${{i}}.log 2>&1 &
              done

        runcmd:
          - mkdir -p /opt/synapse /var/log/synapse
          - bash /opt/synapse/setup.sh
    """)

    # Systemd service template
    files["synapse-spore@.service"] = textwrap.dedent("""\
        [Unit]
        Description=Synapse Brain Spore %i
        After=network-online.target
        Wants=network-online.target

        [Service]
        Type=simple
        User=synapse
        WorkingDirectory=/opt/synapse
        Environment=PATH=/opt/synapse/venv/bin:/usr/bin
        EnvironmentFile=/opt/synapse/.env
        ExecStart=/opt/synapse/venv/bin/python -m synapse_brain.spore.runtime --id %i
        Restart=always
        RestartSec=10

        [Install]
        WantedBy=multi-user.target
    """)

    # Multi-spore launcher script
    files["launch.sh"] = textwrap.dedent(f"""\
        #!/bin/bash
        # Launch {spores_per_vm} spores on this Oracle Cloud VM
        set -e
        source /opt/synapse/.env

        for i in $(seq 0 {spores_per_vm - 1}); do
            PORT=$(({base_port} + $i))
            ID="{spore_id}-$i"
            systemctl enable synapse-spore@$ID
            systemctl start synapse-spore@$ID
            echo "Started spore $ID on port $PORT"
        done

        echo "All {spores_per_vm} spores running."
    """)

    # Environment file
    env_lines = [f'{k}={v}' for k, v in provider_keys.items()]
    env_lines.append(f'SYNAPSE_MESH_SEEDS={seeds_str}')
    files[".env"] = "\n".join(env_lines) + "\n"

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for fname, content in files.items():
            (out / fname).write_text(content)

    return files
