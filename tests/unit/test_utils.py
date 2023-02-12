import pytest
from cifar100classifier.utils import read_yaml   # testing this particular method
from pathlib import Path                         
from box import ConfigBox                        
from ensure.main import EnsureError


class Test_read_yaml:
    yaml_files = ["tests/data/demo.yaml",
        "configs/config.yaml"
    ]

    def test_read_yaml_empty(self):
        
        """This test method tests if the read_yaml function raises a ValueError 
        when reading an empty YAML file. It uses the pytest.raises 
        context manager to catch the expected exception."""
        
        with pytest.raises(ValueError):
            read_yaml(Path(self.yaml_files[0]))

    def test_read_yaml_return_type(self):
        
        """This test method tests if the read_yaml function returns a ConfigBox object. 
        It calls the read_yaml function on the second file in the yaml_files list 
        and checks if the return value is of type ConfigBox using the isinstance function."""
        
        respone = read_yaml(Path(self.yaml_files[-1]))
        assert isinstance(respone, ConfigBox)

    @pytest.mark.parametrize("path_to_yaml", yaml_files)   # passing multiple inputs to test
    def test_read_yaml_bad_type(self, path_to_yaml):
        
        """This test method tests if the read_yaml function raises an EnsureError 
        when reading a YAML file with incorrect data types. 
        The method is decorated with pytest.mark.parametrize which allows 
        multiple inputs to be passed to the test. The path_to_yaml argument 
        is passed in from the yaml_files list, and the test checks if the read_yaml 
        function raises an EnsureError when reading each file. It uses the pytest.raises context manager 
        to catch the expected exception.
        """
        with pytest.raises(EnsureError):
            read_yaml(path_to_yaml)

