import os
import shutil


def install_plugin(force=True):
    """Copy the double_helix PYME plugin YAML to the PYME user config directory.

    Called automatically on first import Call it explicitly — or run 
    `double-helix-install-plugin` from a shell — to force a re-install.
    """
    from PYME import config
    import importlib.resources

    dest_dir = os.path.join(config.user_config_dir, 'plugins')
    dest = os.path.join(dest_dir, 'double_helix.yaml')

    if not force and os.path.exists(dest):
        return

    os.makedirs(dest_dir, exist_ok=True)
    yaml_ref = importlib.resources.files('double_helix').joinpath('double_helix.yaml')
    with importlib.resources.as_file(yaml_ref) as src_path:
        shutil.copy2(str(src_path), dest)


def _ensure_plugin_registered():
    try:
        install_plugin(force=False)
    except Exception:
        pass


_ensure_plugin_registered()
