import pytest
from cifar100classifier.utils import read_yaml   # testing this particular method
from pathlib import Path                         
from box import ConfigBox                        
from ensure.main import EnsureError


class Test_read_yaml:
    yaml_files = [
        "configs/config.yaml"
    ]

    def test_read_yaml_empty(self):
        with pytest.raises(ValueError):
            read_yaml(Path(self.yaml_files[0]))

    def test_read_yaml_return_type(self):
        respone = read_yaml(Path(self.yaml_files[-1]))
        assert isinstance(respone, ConfigBox)

    @pytest.mark.parametrize("path_to_yaml", yaml_files)   # passing multiple inputs to test
    def test_read_yaml_bad_type(self, path_to_yaml):
        with pytest.raises(EnsureError):
            read_yaml(path_to_yaml)

