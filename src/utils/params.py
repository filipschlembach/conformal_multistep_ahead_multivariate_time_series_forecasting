import json
import logging
import os


class Params(dict):
    """
    Example:
    m = Params({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    source: https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary
    """

    def __init__(self, *args, **kwargs):
        super(Params, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Params, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Params, self).__delitem__(key)
        del self.__dict__[key]

    def __str__(self):
        out = ''
        max_key_len = 0

        for k in self.keys():
            key_len = len(str(k))
            if key_len > max_key_len:
                max_key_len = key_len

        for k, v in self.items():
            key_delta = max_key_len - len(str(k))
            out += f'{k}: {" " * key_delta}{v}\n'
        return out

    def save(self, root_dir: str = None) -> None:
        """
        Serializes the object as JSON and writes it to the file system.
        :param root_dir: directory where the file will be written
        :return: None
        """
        if root_dir is None:
            root_dir = self.root_dir
        filename = f'{self.name}.json'
        filepath = os.path.join(os.path.join(root_dir, self.experiment), filename)
        if os.path.exists(filepath):
            logging.warning(f'The file {filepath} already exists and will be overwritten.')
        else:
            logging.info(f'Saving {filepath}')
        with open(filepath, 'w') as out_file:
            json.dump(self, out_file, indent=4)

    def cp(self, name: str = None, experiment: str = None, root_dir: str = None) -> 'Params':
        """
        Returns a shallow copy of the params object.
        :param name: description of the stored parameters
        :param experiment: description of the experiment they are used for
        :param root_dir: directory for storing a serialized params file
        :return: returns a shallow copy of the params object
        """
        copy = Params()
        for k, v in self.items():
            copy[k] = v
        if name is not None:
            copy.name = name
        if experiment is not None:
            copy.experiment = experiment
        if root_dir is not None:
            copy.root_dir = root_dir
        return copy

    @staticmethod
    def load(filepath: str):
        """
        Loads a serialized Params object from the file system
        :param filepath: path to json file
        :return: deserialized object
        """
        with open(filepath, 'r') as input_file:
            json_obj = json.load(input_file)
        return Params(json_obj)

    @staticmethod
    def new(name: str, experiment: str, root_dir: str = 'assets/out'):
        """
        Generates new Params object with the attributes name, experiment and root_dir
        :param name: description of the stored parameters
        :param experiment: description of the experiment they are used for
        :param root_dir: directory for storing a serialized params file
        :return: Params object
        """
        return Params(name=name, experiment=experiment, root_dir=root_dir)
