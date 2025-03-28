from GaussianLSS.common import setup_config
import hydra
from pathlib import Path

CONFIG_PATH = Path.cwd() / 'config'
CONFIG_NAME = 'config.yaml'

@hydra.main(version_base="1.3", config_path=str(CONFIG_PATH), config_name=CONFIG_NAME)
def setup(cfg):
    setup_config(cfg)
    print(cfg)

if __name__ == '__main__':
    print(Path.cwd())
    setup()