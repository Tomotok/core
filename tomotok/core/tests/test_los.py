# Licensed under the EUPL-1.2 or later.
import io
import json
import unittest
from unittest.mock import patch, mock_open

from tomotok.core import Dsystem
from tomotok.core.geometry.los import generate_los, save_los


class LineOfSightTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.diag = Dsystem(0)
        return

    def test_save_los(self):
        """Test input handling for save_los function"""
        sp, ep = generate_los((1,0,0), axis=(-1, 0, 0))
        with patch('builtins.open', mock_open()):
            save_los('filename', sp, ep)
            save_los('filename', sp, ep, 'test')
            save_los('filename', [sp, sp], [ep, ep])
            save_los('filename', [sp, sp], [ep, ep], ['test1', 'test2'])

            self.assertRaises(ValueError, save_los, 'filename', [sp, sp], ep)
            self.assertRaises(ValueError, save_los, 'filename', [sp, sp], [ep, ep], 'test1')
            self.assertRaises(ValueError, save_los, 'filename', [sp, sp], [ep, ep], ['test1'])

    def test_write_read_los(self):
        """Test reading and writing los file"""
        sp, ep = generate_los((1,0,0), axis=(-1, 0, 0))
        with patch('builtins.open', mock_open()) as open_patch:
            string = io.StringIO()
            open_mock = open_patch()
            open_mock.write = string.write
            open_mock.read = string.getvalue
            save_los('foo', sp, ep)
            written = string.getvalue()
            loaded = self.diag.load_los()
            self.assertDictEqual(loaded, json.loads(written))

            # single detector saving with name not in list
            string.truncate(0)
            string.seek(0)
            name = 'test_1'
            save_los('foo', sp, ep, name)
            written = string.getvalue()
            loaded = self.diag.load_los()
            self.assertDictEqual(loaded, json.loads(written))
            self.assertListEqual(list(loaded.keys()), [name])

            # single detector saving with name in list
            string.truncate(0)
            string.seek(0)
            name_2 = ['test_2']
            save_los('foo', sp, ep, name_2)
            written = string.getvalue()
            loaded = self.diag.load_los()
            self.assertDictEqual(loaded, json.loads(written))
            self.assertListEqual(list(loaded.keys()), name_2)

            # multiple detectors saving without names
            string.truncate(0)
            string.seek(0)
            save_los('foo', [sp, sp], [ep, ep])
            written = string.getvalue()
            loaded = self.diag.load_los()
            self.assertDictEqual(loaded, json.loads(written))

            # multiple detectors saving with names
            string.truncate(0)
            string.seek(0)
            names = ['test1', 'test2']
            save_los('foo', [sp, sp], [ep, ep], names)
            written = string.getvalue()
            loaded = self.diag.load_los()
            self.assertDictEqual(loaded, json.loads(written))
            self.assertListEqual(list(loaded.keys()), names)
        return

# TODO: make tests for data checking in load_los method
